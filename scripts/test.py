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

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)

_preprocess = transforms.Compose([
    transforms.Resize((100, 100), 2),
    transforms.ToTensor(),
    normalize
])


if __name__=="__main__":

    test_siamese_dataset = SiameseNetworkDataset(data_path = Config.testing_dir, preprocess = _preprocess)
    test_dataloader = DataLoader(test_siamese_dataset,
                                 num_workers = 6,
                                 batch_size = 1,
                                 shuffle = True)

    dataiter = iter(test_dataloader)
    x0,_,_ = next(dataiter)

    net = SiameseNetwork().cuda()
    net.load_state_dict(torch.load(Config.model_dir + "/siamese.pth"))

    for i in range(10):

        _,x1,label2 = next(dataiter)
        concatenated = torch.cat((x0,x1),0)
        output1,output2 = net(Variable(x0).cuda(),Variable(x1).cuda())
        euclidean_distance = F.pairwise_distance(output1, output2)
        imshow(torchvision.utils.make_grid(concatenated),'Dissimilarity: {:.2f}'.format(euclidean_distance.cpu().data.numpy()[0][0]))

