#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, glob
import random
import numpy as np
from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
from PIL import Image
from config import Config

class NetworkDataset(Dataset):

    def __init__(self, data_path, preprocess = Config.transforms, n_triplets = 5000000):

        self.data_path = data_path
        self.preprocess = preprocess
        self.num_categories = None
        self.image_lists = []
        self.num_img = 0
        self.n_triplets = n_triplets
        self.get_all_images()
        self.triplets = self.generate_triplets()
        
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
        
    def generate_triplets(self):

        triplets = []

        for _ in tqdm(range(self.n_triplets)):
                      
            first_categor_num = np.random.choice(np.arange(self.num_categories))
            a_path, p_path = self.get_random_two_images(self.image_lists[first_categor_num], self.image_lists[first_categor_num])
            negative_categor_list = np.delete(np.arange(self.num_categories), first_categor_num)   
            second_categor_num = np.random.choice(negative_categor_list)      
            n_path = np.random.choice(self.image_lists[second_categor_num])
            triplets.append([a_path, p_path, n_path])

        return torch.LongTensor(np.array(triplets))
            
    def __getitem__(self, index):

        t = self.triplets[index]
        a = Image.open(t[0])
        p = Image.open(t[1])
        n = Image.open(t[2])

        if self.preprocess is not None:

            a = self.preprocess(a)
            p = self.preprocess(p)
            n = self.preprocess(n)

        return a, p, n

    def __len__(self):

        return self.num_img
        
if __name__=="__main__":

    data_set = NetworkDataset(Config.training_dir)
    print data_set.num_categories
    print len(data_set.image_lists[0])
    import cv2
    img = cv2.imread(data_set.image_lists[0][0])
    print img.shape
    print data_set.num_img
