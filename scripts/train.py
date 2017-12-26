#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from training_pairs_generator import NetworkDataset
from triplet_network import TripletNetwork, DBLLoss
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from torch import optim

import model_net

from config import Config
from utils import imshow, show_plot

import matplotlib.pyplot as plt

if __name__=="__main__":

    cnn_model = "resnet"
    _dataset = NetworkDataset(data_path = Config.training_dir)
    train_dataloader = DataLoader(_dataset,
                        shuffle = True,
                        num_workers=8,
                        batch_size = Config.train_batch_size)

    Net = model_net.ResnetBased
    model = Net()
    tnet = TripletNetwork(model).cuda()
    model.cuda()
    
    criterion = DBLLoss()
    optimizer = optim.Adam(net.parameters(),lr = 0.0005 )

    counter = []
    loss_history = [] 
    iteration_number= 0

    for epoch in range(0, Config.train_number_epochs):
        for i, data in enumerate(train_dataloader, 0):

            a, p , n = data
            a, p , n = Variable(a).cuda(), Variable(p).cuda() , Variable(n).cuda()

            e_a, e_p, e_n = tnet(a, p, n)

            optimizer.zero_grad()
            _loss = criterion(e_a, e_p, e_n)
            _loss.backward()
            optimizer.step()

            if i % 10 == 0 :

                print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.data[0]))
                iteration_number +=10
                counter.append(iteration_number)
                loss_history.append(_loss.data[0])
        
        torch.save(net.state_dict(), Config.model_dir + cnn_model + "_triplet_" + str(epoch) + ".pth")

    # plt.plot(counter, loss_history)
    # plt.title("Loss Value Over Time for Training Siamese Dense Net for Object Similarity")
    # plt.savefig(Config.log_dir + "/loss_data.png")
    # plt.close()
