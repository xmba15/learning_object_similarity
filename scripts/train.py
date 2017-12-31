#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from training_pairs_generator import NetworkDataset
from triplet_network import TripletNetWork, loss_random_sampling, global_orthogonal_regularization, CorrelationPenaltyLoss
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from torch import optim

import model_net

from config import Config
from utils import imshow, show_plot

import matplotlib.pyplot as plt



def adjust_learning_rate(optimizer):
    """Updates the learning rate given the learning rate decay.
    The routine has been implemented according to the original Lua SGD optimizer
    """
    for group in optimizer.param_groups:
        if 'step' not in group:
            group['step'] = 0.
        else:
            group['step'] += 1.
        group['lr'] = Config.lr * (
        1.0 - float(group['step']) * float(Config.batch_size) / (Config.n_triplets * float(Config.epochs)))
    return

def create_optimizer(model, new_lr = Config.lr):
    # setup optimizer
    if Config.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.9, dampening=0.9,
                              weight_decay=args.wd)
    elif Config.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=new_lr,
                               weight_decay=args.wd)
    else:
        raise Exception('Not supported optimizer: {0}'.format(args.optimizer))
    return optimizer

def train(train_loader, model, optimizer, epoch):

    model.train()
    pbar = tqdm(enumerate(train_loader))
    
    for batch_idx, data in pbar:
        a, p, n = data
        a, p , n = Variable(a).cuda(), Variable(p).cuda() , Variable(n).cuda()
        e_a, e_p, e_n = model(a, p, n)

        loss = loss_random_sampling(e_a, e_p, e_n, 
                                    margin = Config.margin,
                                    anchor_swap = True,
                                    loss_type = "triplet_margin")

        loss += CorrelationPenaltyLoss()(e_a)
        loss += Config.alpha * global_orthogonal_regularization(e_a, e_n)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    adjust_learning_rate(optimizer)

    torch.save(model.state_dict(), Config.model_dir + Config.cnn_model + "_triplet_" + str(epoch) + ".pth")
    
        
if __name__=="__main__":

    _dataset = NetworkDataset(data_path = Config.training_dir,
                              train = True)
    train_dataloader = DataLoader(_dataset,
                        shuffle = False,
                        num_workers = 8,
                        batch_size = Config.train_batch_size)

    Net = model_net.ResnetBased
    model = Net(normalize = True)
    tnet = TripletNetWork(embedding_net = model, ngpu = 3).cuda()
    model.cuda()

    optimizer1 = create_optimizer(model.features, Config.lr)
    
    for epoch in range(0, Config.train_number_epochs):

        train(train_dataloader, tnet, optimizer1, epoch)
