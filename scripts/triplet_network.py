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

class TripletLoss(torch.nn.Module):

    def __init__(self, margin = 1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, embedded_a, embedded_p, embedded_n):
        
        return F.triplet_margin_loss(embedded_a, embedded_p, embedded_n, margin = self.margin, p = 2, eps = 1e-6, swap = True)
