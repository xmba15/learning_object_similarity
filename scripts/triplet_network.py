#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

resnet = torchvision.models.resnet50(pretrained=True)

class TripletNetwork(nn.Module):

    def __init__(self, feature_size = 512, multiple_view = True, ngpu = 1):

        super(TripletNetwork, self).__init__()
        self.multiple_view = multiple_view
        self.ngpu = ngpu
        self.feature_size = feature_size
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.maxpool1 = nn.MaxPool2d(kernel_size = 7)
        self.fc1 = nn.Linear(2048, self.feature_size)

        self.maxpool2 = nn.MaxPool2d(kernel_size = 7)
        self.fc2 = nn.Linear(2048, self.feature_size)

        nn.init.xavier_normal(self.fc1.weight)
        nn.init.xavier_normal(self.fc2.weight)

    def forward(self, view1, view2, view3, p, n):

        if self.multiple_view == True:
            view1 = self.resnet(view1)
            view2 = self.resnet(view2)
            view3 = self.resnet(view3)
            view = torch.max(view1, view2)
            view = torch.max(view, view3)
            view = self.maxpool2(view)
            view = self.fc2(view)
            view = F.normalize(view, p = 2, dim = 1)

            p = self.resnet(p)
            p = self.maxpool1(p)
            p = self.fc1(p)
            p = F.normalize(p, p = 2, dim = 1)

            n = self.resnet(n)
            n = self.maxpool1(n)
            n = self.fc1(n)
            n = F.normalize(n, p = 2, dim = 1)

            if self.ngpu > 1:
                view = nn.DataParallel(view)
                p = nn.DataParallel(p)
                n = nn.DataParallel(n)

            return view, p, n

        else:
            p = self.resnet(p)
            p = self.maxpool1(p)
            p = self.fc1(p)
            p = F.normalize(p, p = 2, dim = 1)

            n = self.resnet(n)
            n = self.maxpool1(n)
            n = self.fc1(n)
            n = F.normalize(n, p = 2, dim = 1)

            if self.ngpu > 1:
                p = nn.DataParallel(p)
                n = nn.DataParallel(n)

            return p, n

class TripletLoss(torch.nn.Module):

    def __init__(self, margin = 1.0):

        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, embedded_a, embedded_p, embedded_n):

        return F.triplet_margin_loss(embedded_a, embedded_p, embedded_n, margin = self.margin, p = 2, eps = 1e-6, swap = False)
