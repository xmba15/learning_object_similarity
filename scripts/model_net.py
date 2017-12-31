#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class ResnetBased(nn.Module):
    
    def __init__(self, feature_size = 256, im_size = 224, normalize = True):

        super(ResnetBased, self).__init__()
        self.normalize = normalize
        self.im_size = 224
        self.feature_size = feature_size
        self.resnet = torchvision.models.resnet18(pretrained=True)
        fc = nn.Linear(2048, feature_size)
        self.resnet.fc = fc
        self.features = nn.Sequential(self.resnet)
        nn.init.xavier_normal(self.resnet.fc.weight)

    def forward(self, x):
        x = self.features(x)
        if self.normalize:
            return F.normalize(x, p = 2, dim = 1)
        else:
            return x

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal(m.weight.data, gain = 0.7)
        try:
            nn.init.constant(m.bias.data, 0.01)
        except:
            pass
    return
