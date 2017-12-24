#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from config import Config
from training_pairs_generator import SiameseNetworkDataset
from siamese_network import SiameseNetwork
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchvision
from utils import imshow
from PIL import Image

from sklearn.cluster import KMeans

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
    model_name = "densenet_siamese_42.pth"
    net.load_state_dict(torch.load(Config.model_dir + model_name))

    img1_path = Config.image_dir + "/noodle2.jpg"
    img2_path = Config.image_dir + "/dove.jpg"

    img1 = load_image(img1_path)
    img2 = load_image(img2_path)
    
    output1, output2 = net(Variable(img1).cuda(),Variable(img2).cuda())    
    concatenated = torch.cat((img1, img2),0)    
    euclidean_distance = F.pairwise_distance(output1, output2)

    imshow(torchvision.utils.make_grid(concatenated),'Dissimilarity: {:.2f}'.format(euclidean_distance.cpu().data.numpy()[0][0]))
