#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, glob
import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
from PIL import Image

from config import Config

class SiameseNetworkDataset(Dataset):

    def __init__(self, data_path, preprocess = Config.transforms):

        self.data_path = data_path
        self.preprocess = preprocess
        self.num_categories = None
        self.image_lists = []
        self.num_img = 0
        self.get_all_images()
        
    def get_all_images(self):

        dir_list = glob.glob(self.data_path + "/*")
        self.num_categories = len(dir_list)
        
        for _dir in dir_list:
            image_each_cat = []
            for root, dirs, files in os.walk(_dir):

                self.num_img += len(files)
                for name in files:
                    image_each_cat.append(os.path.join(root, name))

            self.image_lists.append(image_each_cat)

    def get_random_two_images(self, list1, list2):

        img_path1 = np.random.choice(list1)
        img_path2 = np.random.choice(list2)

        while (img_path1 == img_path2):
            img_path2 = np.random.choice(list2)

        return img_path1, img_path2
        
    def __getitem__(self, index):

        same_class = random.randint(0, 1)
        first_categor_num = np.random.choice(np.arange(self.num_categories))

        if same_class:
            label = 0
            img_path1, img_path2 = self.get_random_two_images(self.image_lists[first_categor_num], self.image_lists[first_categor_num])
        else:
            label = 1
            negative_categor_list = np.delete(np.arange(self.num_categories), first_categor_num)
            second_categor_num = np.random.choice(negative_categor_list)
            img_path1, img_path2 = self.get_random_two_images(self.image_lists[first_categor_num], self.image_lists[second_categor_num])

        img1 = Image.open(img_path1)
        img2 = Image.open(img_path2)

        if self.preprocess is not None:

            img1 = self.preprocess(img1)
            img2 = self.preprocess(img2)

        return img1, img2, torch.from_numpy(np.array([label], dtype = np.float32))

    def __len__(self):

        return self.num_img
        
if __name__=="__main__":

    data_set = SiameseNetworkDataset(Config.training_dir)
    print data_set.num_categories
    print data_set.image_lists[0][0]
    import cv2
    img = cv2.imread(data_set.image_lists[0][0])
    print img.shape
    data_set.__getitem__(1)
    print data_set.num_img
