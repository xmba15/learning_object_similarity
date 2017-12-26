#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from config import Config

lapsrn_model_path = Conifg.lapsrn_dir + "model/model_epoch_100.pth"
lapsrn_model = torch.load(model_path)["model"]
for param in lapsrn_model.parameters():
    param.requires_grad = False

resnet_model = torchvision.models.resnet50(pretrained = True)
for param in resnet_model.parameters():
    param.requires_grad = False

class SiameseNetwork(nn.Module):

    def __init__(self, pretrained = False, embed_size = 1000):

        super(SiameseNetwork, self).__init__()
        
        self.embed_size = embed_size
        self.pretrained = pretrained
        self.lapsrn_model = lapsrn_model
        self.adaptive_avg_pool2d = nn.AdaptiveAvgPool2d((224, 224))
        modules = list(resnet_model.children())[:-1]
        self.resnet_model = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet_model.fc.in_features, self.embed_size)
        self.bn = nn.BatchNorm1d(self.embed_size, momentum = 0.01)

        self.linear2 = nn.Linear(2000, 100)        
        self.init_weights()
        
    def init_weights(self):
        
        """Initialize the weights."""
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

        self.linear2.weight.data.normal_(0.0, 0.02)
        self.linear2.bias.data.fill_(0)
        
    def forward_once(self, x):

        HR_2x, HR_4x = self.lapsrn_model(x)
        output = torch.cat((HR_4x, HR_4x, HR_4x), 1)
        output = self.adaptive_avg_pool2d(output)
        output = self.resnet_model(output)
        output = self.bn(self.linear(output))

        return output

    def forward(self, input1, input2):

        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output = torch.cat((output1, output2), 1)
        output = self.linear2(output)
        return output

class ContrastiveLoss(torch.nn.Module):

    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin = 1.0):

        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive        
