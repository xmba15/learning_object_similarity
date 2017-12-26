#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from config import Config
from training_pairs_generator import SiameseNetworkDataset
from siamese_network_old import SiameseNetwork
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchvision
from utils import imshow
from PIL import Image

from sklearn.cluster import KMeans

import cv2
import numpy as np

import torchvision.transforms as transforms
normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)

_preprocess = transforms.Compose([
    transforms.Resize((224, 224), 2),
    transforms.ToTensor(),
    normalize
])

def load_image(img_path):

    img = Image.open(img_path)
    img = _preprocess(img)

    return img.unsqueeze(0)

if __name__=="__main__":

    net = SiameseNetwork().cuda()
    model_name = "siamese_98.pth"
    net.load_state_dict(torch.load(Config.model_dir + model_name))

    img1_path = Config.image_dir + "/output3.jpg"
    img2_path = Config.image_dir + "/output2.jpg"

    img1 = load_image(img1_path)
    img2 = load_image(img2_path)
    
    output1, output2 = net(Variable(img1).cuda(),Variable(img2).cuda())
    _output1 = output1.cpu().data.numpy()[0]
    _output2 = output2.cpu().data.numpy()[0]
    _output1 = _output1.reshape((1, 1000))
    _output2 = _output2.reshape((1, 1000))
    _output1 = np.array(_output1)
    _output1 /= _output1.max()
    _output2 = np.array(_output2)
    _output2 /= _output2.max()
    
    # from matplotlib import pyplot as plt
    # plt.hist(_output1.ravel(), 1000, [_output1.min(), _output1.max()])
    # plt.hist(_output2.ravel(), 1000, [_output2.min(), _output2.max()])    
    # plt.show()
        
    tmp = cv2.compareHist(_output1, _output2, HISTCMP_BHATTACHARYYA)
    # tmp = cv2.compareHist(_output1, _output2, cv2.HISTCMP_CHISQR)    
    # print tmp

    concatenated = torch.cat((img1, img2),0)    
    euclidean_distance = F.pairwise_distance(output1, output2)
    print euclidean_distance
    imshow(torchvision.utils.make_grid(concatenated),'Dissimilarity: {:.2f}'.format(euclidean_distance.cpu().data.numpy()[0][0]))
