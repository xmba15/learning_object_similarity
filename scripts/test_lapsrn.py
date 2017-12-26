#!/usr/bin/env python
# -*- coding: utf-8 -*-

from config import Config
import sys
sys.path.insert(0, Config.lapsrn_dir)
from lapsrn import Net as LapSRN

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

from PIL import Image
img = Image.open(Config.image_dir + "/noodle.jpg")

_preprocess = transforms.Compose([
    transforms.ToTensor(),
])

img = _preprocess(img)
img = img.unsqueeze(0)

_lapsrn = LapSRN()
_lapsrn_cnn = _lapsrn.cuda()
# print _lapsrn_cnn
lapsrn_net = torch.load(Config.lapsrn_dir + "/model/model_epoch_100.pth")["model"]
for param in lapsrn_net.parameters():
    param.requires_grad = False

# _lapsrn_cnn.load_state_dict(net)
