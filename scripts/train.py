#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from training_pairs_generator import NetworkDataset
from triplet_network import TripletNetWork,TripletLoss, global_orthogonal_regularization, CorrelationPenaltyLoss
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from torch import optim

import model_net

from config import Config
from utils import imshow, show_plot

import matplotlib.pyplot as plt

if __name__=="__main__":

    cnn_model = "resnet_fc256_"
    _dataset = NetworkDataset(data_path = Config.training_dir)
    train_dataloader = DataLoader(_dataset,
                        shuffle = True,
                        num_workers = 8,
                        batch_size = Config.train_batch_size)

    Net = model_net.ResnetBased
    model = Net(normalize = True)
    tnet = TripletNetWork(embedding_net = model, ngpu = 3).cuda()
    model.cuda()
    
    criterion = TripletLoss()
    optimizer = optim.Adam(tnet.parameters(),lr = 0.0005 )

    counter = []
    loss_history = [] 
    iteration_number= 0

    for epoch in range(0, Config.train_number_epochs):
        for i, data in enumerate(train_dataloader, 0):

            a, p , n = data
            a, p , n = Variable(a).cuda(), Variable(p).cuda() , Variable(n).cuda()

            e_a, e_p, e_n = tnet(a, p, n)


            _loss = criterion(e_a, e_p, e_n)
            _loss += CorrelationPenaltyLoss()(e_a)
            _loss += Config.gor_alpha * global_orthogonal_regularization(e_a, e_n)
            optimizer.zero_grad()            
            _loss.backward()
            optimizer.step()

            if i % 10 == 0 :

                print("Epoch number {}\n Current loss {}\n".format(epoch, _loss.data[0]))
                iteration_number +=10
                counter.append(iteration_number)
                loss_history.append(_loss.data[0])
        
        torch.save(tnet.state_dict(), Config.model_dir + cnn_model + "_triplet_" + str(epoch) + ".pth")
