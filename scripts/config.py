#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random

directory_root = os.path.dirname(os.path.realpath(__file__)) + "/../"
data_path = directory_root + "/data/raw_training_data/"
test_data_path = directory_root + "/data/raw_testing_data/"
model_path = directory_root + "/models/"
image_path = directory_root + "/images/"
log_path = directory_root + "/logs/"

import torchvision.transforms as transforms
from new_augmentation import SquareZeroPadding

Normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)

RandomColorJitter = transforms.Lambda(

    lambda x: transforms.ColorJitter(brightness = 0.1, contrast = 0.1, hue = 0.01)(x) if random.random() < 0.5 else x)

RandomZoom = transforms.Lambda(

    lambda x: transforms.Resize((224, 224), 2)(transforms.CenterCrop((220, 220))(x)) if random.random() < 0.5 else x)

_preprocess = transforms.Compose([
    SquareZeroPadding(),
    transforms.Resize((224, 224), 2),
    RandomZoom,
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees = 10),
    RandomColorJitter,
    transforms.ToTensor(),
    Normalize
])

class Config():

    training_dir = data_path
    testing_dir = test_data_path
    model_dir = model_path
    image_dir = image_path
    log_dir = log_path
    train_batch_size = 64
    train_number_epochs = 100
    transforms = _preprocess
    feature_extract_model = "resnet50"
    n_triplets = 1280000
