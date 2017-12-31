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

    def forward(self, a, p, n):
        
        if isinstance(a.data, torch.cuda.FloatTensor) and \
           isinstance(p.data, torch.cuda.FloatTensor) and \
           isinstance(a.data, torch.cuda.FloatTensor) and \
           self.ngpu > 1:
            embedded_a = nn.parallel.data_parallel(self.embedding_net, a, range(self.ngpu))
            embedded_p = nn.parallel.data_parallel(self.embedding_net, p, range(self.ngpu))
            embedded_n = nn.parallel.data_parallel(self.embedding_net, n, range(self.ngpu))
        else:
            embedded_a = self.embedding_net(a)
            embedded_p = self.embedding_net(p)
            embedded_n = self.embedding_net(n)
        
        return embedded_a, embedded_p, embedded_n

class TripletLoss(torch.nn.Module):

    def __init__(self, margin = 1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, embedded_a, embedded_p, embedded_n):
        
        return F.triplet_margin_loss(embedded_a, embedded_p, embedded_n, margin = self.margin, p = 2, eps = 1e-8, swap = True)

def global_orthogonal_regularization(anchor, negative):

    neg_dis = torch.sum(torch.mul(anchor,negative),1)
    dim = anchor.size(1)
    gor = torch.pow(torch.mean(neg_dis),2) + torch.clamp(torch.mean(torch.pow(neg_dis,2))-1.0/dim, min=0.0)

    return gor

class CorrelationPenaltyLoss(nn.Module):

    def __init__(self):
        super(CorrelationPenaltyLoss, self).__init__()

    def forward(self, input):
        mean1 = torch.mean(input, dim=0)
        zeroed = input - mean1.expand_as(input)
        cor_mat = torch.bmm(torch.t(zeroed).unsqueeze(0), zeroed.unsqueeze(0)).squeeze(0)
        d = torch.diag(torch.diag(cor_mat))
        no_diag = cor_mat - d
        d_sq = no_diag * no_diag

        return torch.sqrt(d_sq.sum())/input.size(0)
