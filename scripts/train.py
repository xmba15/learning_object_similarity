#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from training_pairs_generator import SiameseNetworkDataset
from siamese_network import SiameseNetwork, ContrastiveLoss
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from torch import optim

from config import Config
from utils import imshow, show_plot

if __name__=="__main__":

    cnn_model = "densenet"
    siamese_dataset = SiameseNetworkDataset(data_path = Config.training_dir)
    train_dataloader = DataLoader(siamese_dataset,
                        shuffle = True,
                        num_workers=8,
                        batch_size = Config.train_batch_size)
    
    net = SiameseNetwork(pretrained = True).cuda()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(),lr = 0.0005 )

    counter = []
    loss_history = [] 
    iteration_number= 0

    for epoch in range(0, Config.train_number_epochs):
        for i, data in enumerate(train_dataloader, 0):

            img0, img1 , label = data
            img0, img1 , label = Variable(img0).cuda(), Variable(img1).cuda() , Variable(label).cuda()

            output1, output2 = net(img0, img1)

            optimizer.zero_grad()
            loss_contrastive = criterion(output1,output2,label)
            loss_contrastive.backward()
            optimizer.step()

            if i % 10 == 0 :

                print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.data[0]))
                iteration_number +=10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.data[0])
        
        torch.save(net.state_dict(), Config.model_dir + cnn_model + "_siamese_" + str(epoch) + ".pth")
    # show_plot(counter,loss_history)
