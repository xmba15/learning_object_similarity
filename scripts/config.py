#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
directory_root = os.path.dirname(os.path.realpath(__file__)) + "/../"
data_path = directory_root + "/data/raw_training_data/"
test_data_path = directory_root + "/data/raw_testing_data/"
model_path = directory_root + "/models/"

class Config():

    training_dir = data_path
    testing_dir = test_data_path
    model_dir = model_path
    train_batch_size = 500
    train_number_epochs = 10
