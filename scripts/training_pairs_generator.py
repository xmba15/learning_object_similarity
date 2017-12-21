#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, glob
from torch.utils.data import DataLoader,Dataset

directory_root = os.path.dirname(os.path.realpath(__file__)) + "/../"
data_path = directory_root + "/data/raw_training_data/"

class SiameseNetworkDataset(Dataset):

    def __init__(self, data_path):

        self.data_path = data_path
        self.num_categories = None
        self.image_lists = []
        self.get_all_images()
        
    def get_all_images(self):

        dir_list = glob.glob(self.data_path + "/*")
        self.num_categories = len(dir_list)
        for _dir in dir_list:
            image_each_cat = []
            for root, dirs, files in os.walk(_dir):
                for name in files:
                    image_each_cat.append(os.path.join(root, name))
            self.image_lists.append(image_each_cat)

if __name__=="__main__":

    data_set = SiameseNetworkDataset(data_path)
    print data_set.num_categories
