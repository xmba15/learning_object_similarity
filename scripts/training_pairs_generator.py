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

    def __init__(self, data_path = Config.training_dir, preprocess = Config.transforms, \
                 batch_size = Config.train_batch_size, n_triplets = Config.n_triplets):

        self.data_path = data_path
        self.preprocess = preprocess
        self.num_categories = None
        self.image_lists = []
        self.num_img = 0
        self.batch_size = batch_size
        self.n_triplets = n_triplets
        self.get_all_images()
        self.triplets = self.generate_triplets_anchor_multi_view()

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

    def get_multiple_view_and_p_images(self, l_img):

        max_step = 20
        num_view = 3
        step = np.random.randint(max_step)

        view1_index = np.random.randint(len(l_img) - max_step * (num_view - 1))

        v1_path = l_img[view1_index]
        v2_path = l_img[view1_index + step * 1]
        v3_path = l_img[view1_index + step * 2]

        p_img_list = np.delete(l_img, [view1_index, view1_index + step * 1, view1_index + step * 2])

        p_path = np.random.choice(p_img_list)

        return v1_path, v2_path, v3_path, p_path

    def generate_triplets_anchor_multi_view(self):

        triplets = []
        anchor_categories = np.arange(self.num_categories)

        for _ in tqdm(range(self.n_triplets)):

            if len(anchor_categories) == 0:
                anchor_categories = np.arange(self.num_categories)

            random_index = np.random.randint(len(anchor_categories))
            first_categor_num = anchor_categories[random_index]
            anchor_categories = np.delete(anchor_categories, random_index)

            negative_categor_list = np.delete(np.arange(self.num_categories), first_categor_num)
            second_categor_num = np.random.choice(negative_categor_list)
            n_path = np.random.choice(self.image_lists[second_categor_num])

            v1_path, v2_path, v3_path, p_path = self.get_multiple_view_and_p_images(self.image_lists[first_categor_num])

            triplets.append([v1_path, v2_path, v3_path, p_path, n_path])

        triplets = np.array(triplets)

        return triplets

    def __getitem__(self, index):

        t = self.triplets[index]

        v1_path, v2_path, v3_path, p_path, n_path = t[0], t[1], t[2], t[3], t[4]

        v1 = Image.open(v1_path)
        v2 = Image.open(v2_path)
        v3 = Image.open(v3_path)
        p = Image.open(p_path)
        n = Image.open(n_path)

        if self.preprocess is not None:
            v1 = self.preprocess(v1)
            v2 = self.preprocess(v2)
            v3 = self.preprocess(v3)
            p = self.preprocess(p)
            n = self.preprocess(n)

        return v1, v2, v3, p, n

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
