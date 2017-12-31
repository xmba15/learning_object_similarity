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
from tqdm import tqdm

from config import Config

class NetworkDataset(Dataset):

    def __init__(self, data_path, preprocess = Config.transforms, batch_size = Config.train_batch_size, n_triplets = Config.n_triplets):

        self.data_path = data_path
        self.preprocess = preprocess
        self.num_categories = None
        self.image_lists = []
        self.num_img = 0
        self.batch_size = batch_size
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
        already_idxs = set()

        for _ in tqdm(range(self.n_triplets)):
            if len(already_idxs) >= self.batch_size:
                already_idxs = set()
            first_categor_num = np.random.choice(np.arange(self.num_categories))

            while first_categor_num in already_idxs:
                first_categor_num = np.random.choice(np.arange(self.num_categories))
                already_idxs.add(first_categor_num)
                
            a_path, p_path = self.get_random_two_images(self.image_lists[first_categor_num], self.image_lists[first_categor_num])
            negative_categor_list = np.delete(np.arange(self.num_categories), first_categor_num)
            second_categor_num = np.random.choice(negative_categor_list)
            n_path = np.random.choice(self.image_lists[second_categor_num])
            triplets.append([a_path, p_path, n_path])

        triplets = np.array(triplets)

        return triplets
                
    def __getitem__(self, index):

        # same_class = random.randint(0, 1)
        # first_categor_num = np.random.choice(np.arange(self.num_categories))

        # a_path, p_path = self.get_random_two_images(self.image_lists[first_categor_num], self.image_lists[first_categor_num])

        # negative_categor_list = np.delete(np.arange(self.num_categories), first_categor_num)
        # second_categor_num = np.random.choice(negative_categor_list)
        # n_path = np.random.choice(self.image_lists[second_categor_num])

        t = self.triplets[index]
        a_path, p_path, n_path = t[0], t[1], t[2]
        a = Image.open(a_path)
        p = Image.open(p_path)
        n = Image.open(n_path)

        if self.preprocess is not None:

            a = self.preprocess(a)
            p = self.preprocess(p)
            n = self.preprocess(n)

        return a, p, n

    def __len__(self):

        return self.n_triplets
        
if __name__=="__main__":

    data_set = NetworkDataset(Config.training_dir)
    print data_set.num_categories
    print data_set.image_lists[0][0]
    import cv2
    img = cv2.imread(data_set.image_lists[0][0])
    print img.shape
    data_set.__getitem__(1)
    print data_set.num_img
