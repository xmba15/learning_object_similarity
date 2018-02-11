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



img = Image.open(image_path + "/roi_1.jpg")
img1 = Image.open(image_path + "/roi_3.jpg")
img2 = Image.open(image_path + "/roi_2.jpg")

_preprocess = transforms.Compose([
    transforms.Resize((224, 224), 2),
    transforms.ToTensor(),
])

img = _preprocess(img)
img = img.unsqueeze(0)

img1 = _preprocess(img1)
img1 = img1.unsqueeze(0)

img2 = _preprocess(img2)
img2 = img2.unsqueeze(0)

densnet = torchvision.models.densenet201(pretrained = True)
densnet_cnn = nn.Sequential(densnet).cuda()
img = Variable(img).cuda()
img1 = Variable(img1).cuda()
img2 = Variable(img2).cuda()

from time import time

start = time()
# output = F.normalize(densnet_cnn(img), p = 2, dim = 1)
output1 = F.normalize(densnet_cnn(img1), p = 2, dim = 1)
output2 = F.normalize(densnet_cnn(img2), p = 2, dim = 1)

euclidean_distance = F.pairwise_distance(output1, output2)
print euclidean_distance

# print time() - start
# print output
