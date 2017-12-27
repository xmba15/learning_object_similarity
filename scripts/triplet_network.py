#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class TripletNetWork(nn.Module):

    def __init__(self, embedding_net):

        super(TripletNetWork, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, a, p, n):

        embedded_a = self.embedding_net(a)
        embedded_p = self.embedding_net(p)
        embedded_n = self.embedding_net(n)
        
        return embedded_a, embedded_p, embedded_n

class DBLLoss(torch.nn.Module):

    def __init__(self, margin = 1.0):
        super(DBLLoss, self).__init__()
        self.margin = margin

    def forward(self, embedded_a, embedded_p, embedded_n):
        
        # ap = F.pairwise_distance(embedded_a, embedded_p)
        # an = F.pairwise_distance(embedded_a, embedded_n)
        # pn = F.pairwise_distance(embedded_p, embedded_n)
        # an = torch.min(an, pn)

        # ap = ap.mul(self.k)
        # an = an.mul(self.k)
        
        # p = 1 / (1.0 +  torch.exp(ap - an))
        # loss = - torch.log(p)

        # if torch.gt(loss, 10e10):
        #     loss = ap - an

        return F.triplet_margin_loss(embedded_a, embedded_p, embedded_n, margin = self.margin, p = 2, eps = 1e-6, swap = True)
        
class ContrastiveLoss(torch.nn.Module):


    def __init__(self, margin = 1.0):

        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive        
