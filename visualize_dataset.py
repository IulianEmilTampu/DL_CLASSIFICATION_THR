'''
Script that, given the folder where the TFR detaset is located, prints out examples
of testing and training datasets as they come out from the data loader
'''

import os
import sys
import cv2
import glob
import json
import pickle
import random
import pathlib
# import imutils
import argparse
import importlib
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.utils import to_categorical
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

# custom imports
import utilities
import utilities_models_tf


## 1 - load the data that the model used for training and testing
# get dataset info from the configuration file
from_configuration_file = True

if from_configuration_file:
    model_name = "M4_c1_BatchNorm_dr0.2_lr0.00001_wcce_weights_batch64"
    trained_models_path = '/flush/iulta54/Research/P3-THR_DL/trained_models'
    dataset_path = '/flush/iulta54/Research/Data/OCT/Thyroid_2019_refined_DeepLearning/'

    # load configuration file
    with open(os.path.join(trained_models_path, model_name,'config.json')) as json_file:
        config = json.load(json_file)

    # take one testing. training and validation images (tr and val for fold specific fold)
    # make sure that the files point to this system dataset
    fold = 0
    test_img = [os.path.join(dataset_path, pathlib.Path(f).parts[-2], pathlib.Path(f).parts[-1]) for f in config['test']]
    tr_img = [os.path.join(dataset_path, pathlib.Path(f).parts[-2], pathlib.Path(f).parts[-1]) for f in config['training'][fold]]
    val_img = [os.path.join(dataset_path, pathlib.Path(f).parts[-2], pathlib.Path(f).parts[-1]) for f in config['validation'][fold]]

    # some other settings
    crop_size = config['input_size'] # (h, w)
else:
    # specify manually the files to show
    test_img = []
    tr_img = []
    val_img = []
    crop_size = []

examples_to_show = 80

## 2 create dataset and augmentation layer
importlib.reload(utilities)

# build tf generator
test_dataloader = utilities.TFR_2D_dataset(test_img,
                dataset_type = 'train',
                batch_size=examples_to_show,
                buffer_size=5000,
                crop_size=crop_size)


## 3 take out data
importlib.reload(utilities)
importlib.reload(utilities_models_tf)

x, y = next(iter(test_dataloader))

# augment
# x = utilities_models_tf.augmentor(x)
# fix labels
y = utilities_models_tf.fix_labels_v2(y.numpy(), classification_type='c3', unique_labels=config['unique_labels'], categorical=False)

sample = (x.numpy(), y.numpy())

print(f'{"Mean:":5s}{x.numpy().mean():0.2f}')
print(f'{"STD:":5s}{x.numpy().std():0.2f}')

utilities.show_batch_2D(sample, img_per_row=10, title='Without augmentation')

# show examples with histogram

utilities.show_batch_2D_with_histogram(sample, title='Without augmentation')

## 4 apply augmentation (check histograms)
importlib.reload(utilities)
importlib.reload(utilities_models_tf)

# create agumentation layer
def augmentor(inputs):
    aug = tf.keras.Sequential([
            layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
            layers.experimental.preprocessing.RandomRotation(0.02)],
            name='Augmentation')
    return aug(inputs)
# augment

# layers.experimental.preprocessing.RandomContrast(factor=0.8)

x = augmentor(x)

sample = (x.numpy(), y.numpy())

print(f'{"Mean:":5s}{x.numpy().mean():0.2f}')
print(f'{"STD:":5s}{x.numpy().std():0.2f}')

utilities.show_batch_2D(sample, img_per_row=10, title='With augmentation')

# show examples with histogram

utilities.show_batch_2D_with_histogram(sample, title='With augmentation')



