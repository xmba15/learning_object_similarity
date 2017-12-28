#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from config import Config
from training_pairs_generator import NetworkDataset
from triplet_network import TripletNetWork
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchvision
from utils import imshow
from PIL import Image

import cv2
import numpy as np

import model_net

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
    # normalize
])

def load_image(img_path):

    img = Image.open(img_path)
    img = _preprocess(img)

    return img.unsqueeze(0)

if __name__=="__main__":

    
    Net = model_net.ResnetBased
    model = Net(normalize = True)
    tnet = TripletNetWork(model).cuda()
    tnet.eval()
    tnet.load_state_dict(torch.load(Config.model_dir + "/resnet_triplet_triplet_69.pth"))
    
    img1_path = Config.image_dir + "/apple.jpg"
    img2_path = Config.image_dir + "/fast_mask_roi_8.jpg"
    img3_path = Config.image_dir + "/tea.jpg"    
    
    img1 = load_image(img1_path)
    img2 = load_image(img2_path)
    img3 = load_image(img3_path)

    output1, output2, output3 = tnet(Variable(img1).cuda(), Variable(img2).cuda(), Variable(img3).cuda())

    # _output1 = output1.cpu().data.numpy()[0]
    # _output2 = output2.cpu().data.numpy()[0]
    # _output1 = _output1.reshape((1, 1000))
    # _output2 = _output2.reshape((1, 1000))
    # _output1 = np.array(_output1)
    # _output1 /= _output1.max()
    # _output2 = np.array(_output2)
    # _output2 /= _output2.max()
    
    # from matplotlib import pyplot as plt
    # plt.hist(_output1.ravel(), 1000, [_output1.min(), _output1.max()])
    # plt.hist(_output2.ravel(), 1000, [_output2.min(), _output2.max()])    
    # plt.show()
        
    # tmp = cv2.compareHist(_output1, _output2, HISTCMP_BHATTACHARYYA)
    # tmp = cv2.compareHist(_output1, _output2, cv2.HISTCMP_CHISQR)    

    concatenated = torch.cat((img2, img3),0)
    print F.pairwise_distance(output1, output2)
    euclidean_distance = F.pairwise_distance(output2, output3)
    print euclidean_distance
    imshow(torchvision.utils.make_grid(concatenated),'Dissimilarity: {:.2f}'.format(euclidean_distance.cpu().data.numpy()[0][0]))
