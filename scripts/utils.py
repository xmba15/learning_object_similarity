#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torch, torchvision
import numpy as np

from training_pairs_generator import NetworkDataset

def imshow(img, text, should_save=False):

    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

def show_plot(iteration,loss):

    plt.plot(iteration,loss)
    plt.show()

if __name__=="__main__":

    directory_root = os.path.dirname(os.path.realpath(__file__)) + "/../"
    _data_path = directory_root + "/data/raw_training_data/"
    test_data_path = directory_root + "/data/raw_testing_data/"
    
    # siamese_dataset = NetworkDataset(data_path = _data_path)
    siamese_dataset = NetworkDataset(data_path = test_data_path)
    
    vis_dataloader = DataLoader(siamese_dataset,
                        shuffle = True,
                        num_workers = 8,
                        batch_size = 8)
    dataiter = iter(vis_dataloader)
    example_batch = next(dataiter)
    concatenated = torch.cat((example_batch[0], example_batch[1]), 0)

    print example_batch[2]    
    imshow(torchvision.utils.make_grid(concatenated), None)
