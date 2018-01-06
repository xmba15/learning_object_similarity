#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class ResnetBased(nn.Module):
    
    def __init__(self, feature_size = 1000, im_size = 224, normalize = True):

        super(ResnetBased, self).__init__()
        self.normalize = normalize
        self.im_size = 224
        self.feature_size = feature_size
        self.resnet = torchvision.models.resnet50(pretrained=True)
        fc = nn.Linear(self.resnet.fc.in_features, feature_size)
        self.resnet.fc = fc
        nn.init.xavier_normal(self.resnet.fc.weight)

    def forward(self, x):
        x = self.resnet(x)
        if self.normalize:
            return F.normalize(x, p = 2, dim = 1)
        else:
            return x
