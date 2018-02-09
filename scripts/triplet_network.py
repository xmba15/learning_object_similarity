#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class TripletNetWork(nn.Module):

    def __init__(self, embedding_net, ngpu = 1):

        super(TripletNetWork, self).__init__()
        self.embedding_net = embedding_net
        self.ngpu = ngpu

    def forward(self, a):
        
        if isinstance(a.data, torch.cuda.FloatTensor) and \
           self.ngpu > 1:
            embedded_a = nn.parallel.data_parallel(self.embedding_net, a, range(self.ngpu))
        else:
            embedded_a = self.embedding_net(a)
        
        return embedded_a

class TripletLoss(torch.nn.Module):

    def __init__(self, margin = 1.0):

        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, embedded_a, embedded_p, embedded_n):
        
        return F.triplet_margin_loss(embedded_a, embedded_p, embedded_n, margin = self.margin, p = 2, eps = 1e-6, swap = False)
