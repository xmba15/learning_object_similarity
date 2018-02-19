#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from config import Config
from triplet_network import TripletNetwork
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchvision
from utils import imshow
from PIL import Image

import cv2
import numpy as np

from new_augmentation import SquareZeroPadding
from config import Config
from utils import imshow, show_plot


import torchvision.transforms as transforms
normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)

_preprocess = transforms.Compose([
    SquareZeroPadding(),
    transforms.Resize((224, 224), 2),
    transforms.ToTensor(),
    normalize
])

def load_image(img_path):

    img = Image.open(img_path)
    img = _preprocess(img)

    return img.unsqueeze(0)

if __name__=="__main__":

    tnet = TripletNetwork().cuda()
    tnet = torch.nn.DataParallel(tnet)
    tnet.eval()
    tnet.load_state_dict(torch.load(Config.model_dir + "/resnet50_triplet_9.pth"))
    
    img1_path = Config.image_dir + "/noodle.jpg"
    img2_path = Config.image_dir + "/noodle3.jpg"
    img3_path = Config.image_dir + "/noodle4.jpg"    
    p_path = Config.image_dir + "/cold_brew.jpg"
    n_path = Config.image_dir + "/noodle2.jpg"
    
    img1 = load_image(img1_path)
    img2 = load_image(img2_path)
    img3 = load_image(img3_path)
    p = load_image(p_path)
    n = load_image(n_path)
    
    output1, output2, output3 = tnet(Variable(img1).cuda(), Variable(img2).cuda(), Variable(img3).cuda(), Variable(p).cuda(), Variable(n).cuda())

    print F.pairwise_distance(output1, output2)    
    print F.pairwise_distance(output1, output3)    
