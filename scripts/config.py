#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
directory_root = os.path.dirname(os.path.realpath(__file__)) + "/../"
data_path = directory_root + "/data/raw_training_data/"
test_data_path = directory_root + "/data/raw_testing_data/"
model_path = directory_root + "/models/"

import torchvision.transforms as transforms

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)

_preprocess = transforms.Compose([
    transforms.Resize((224, 224), 2),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    normalize
])

class Config():

    training_dir = data_path
    testing_dir = test_data_path
    model_dir = model_path
    train_batch_size = 50
    train_number_epochs = 100
    transforms = _preprocess
