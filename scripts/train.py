#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from training_pairs_generator import NetworkDataset
from triplet_network import TripletNetwork,TripletLoss, global_orthogonal_regularization
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from torch import optim

from config import Config
from utils import imshow, show_plot

if __name__=="__main__":

    dataset = NetworkDataset()
    train_dataloader = DataLoader(dataset,
                        shuffle = True,
                        num_workers = 8,
                        batch_size = Config.train_batch_size)

    tnet = TripletNetwork().cuda()    
    tnet = torch.nn.DataParallel(tnet, [0, 1])
    criterion = TripletLoss()
    optimizer = optim.Adam(tnet.parameters(),lr = 0.0005)

    counter = []
    loss_history = []
    iteration_number= 0

    for epoch in range(0, Config.train_number_epochs):
        for i, data in enumerate(train_dataloader, 0):

            view1, view2, view3, p, n = data
            view1, view2, view3, p, n = Variable(view1).cuda(), Variable(view2).cuda() , Variable(view3).cuda(), Variable(p).cuda(), Variable(n).cuda()

            e_a, e_p, e_n = tnet(view1, view2, view3, p, n)

            optimizer.zero_grad()
            _loss = criterion(e_a, e_p, e_n)
            _loss += Config.gor_alpha * global_orthogonal_regularization(e_a, e_n)
            _loss.backward()
            optimizer.step()

            if i % 10 == 0 :

                print("Epoch number {}\n Current loss {}\n".format(epoch, _loss.data[0]))
                iteration_number +=10

        torch.save(tnet.state_dict(), Config.model_dir + Config.feature_extract_model + "_triplet_" + str(epoch) + ".pth")
