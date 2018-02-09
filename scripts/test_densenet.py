#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
directory_root = os.path.dirname(os.path.realpath(__file__)) + "/../"
data_path = directory_root + "/data/raw_training_data/"
test_data_path = directory_root + "/data/raw_testing_data/"
model_path = directory_root + "/models/"
image_path = directory_root + "/images/"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable

from PIL import Image



img = Image.open(image_path + "/noodle.jpg")

_preprocess = transforms.Compose([
    transforms.Resize((224, 224), 2),
    transforms.ToTensor(),
])

img = _preprocess(img)
img = img.unsqueeze(0)

densnet = torchvision.models.vgg19_bn(pretrained = True).cuda().eval()
densnet_cnn = densnet
# densnet_cnn = nn.Sequential(densnet).cuda()
img = Variable(img).cuda()

from time import time

start = time()
output = densnet_cnn(img)
print time() - start
print output
