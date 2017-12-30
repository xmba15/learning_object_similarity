#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from config import Config

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable

from PIL import Image

from new_augmentation import SquareZeroPadding

img = Image.open(Config.image_dir + "/roi_1.jpg")
img1 = Image.open(Config.image_dir + "/roi_3.jpg")
img2 = Image.open(Config.image_dir + "/roi_2.jpg")

_preprocess = transforms.Compose([
    SquareZeroPadding(),
    transforms.Resize((224, 224), 2),
    # transforms.ToTensor(),
])

img = _preprocess(img)
# img = img.unsqueeze(0)
img.show()

# print img
# img1 = _preprocess(img1)
# img1 = img1.unsqueeze(0)

# img2 = _preprocess(img2)
# img2 = img2.unsqueeze(0)

# densnet = torchvision.models.densenet201(pretrained = True)
# densnet_cnn = nn.Sequential(densnet).cuda()
# img = Variable(img).cuda()
# img1 = Variable(img1).cuda()
# img2 = Variable(img2).cuda()

from time import time

start = time()
# output = F.normalize(densnet_cnn(img), p = 2, dim = 1)

# euclidean_distance = F.pairwise_distance(output1, output2)
# print euclidean_distance

