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
from tensorflow.keras.models import Sequential
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

# custom imports
import utilities


## 1 specify the path to the TFR record database
importlib.reload(utilities)

# data to work on
dataset_path = '/flush/iulta54/Research/Data/OCT/Thyroid/NewDataset/2D_classification_dataset_anisotropic_TFR'
training_dataset = os.path.join(dataset_path, 'Train')
testing_dataset = os.path.join(dataset_path, 'Test')

# get training and testing files
train_val_file_list = []
test_filenames = []

unique_labels = ['class_0',
                 'class_1',
                 'class_2',
                 'class_3',
                 'class_4',
                 'class_5'
                 ]

for idx1, c in enumerate(unique_labels):
    if type(c) is list:
        aus = []
        for cc in c:
            train_val_file_list.extend(glob.glob(os.path.join(training_dataset, cc,'*.tfrecords')))
            test_filenames.extend(glob.glob(os.path.join(testing_dataset, cc,'*.tfrecords')))
    else:
        train_val_file_list.extend(glob.glob(os.path.join(training_dataset, c,'*.tfrecords')))
        test_filenames.extend(glob.glob(os.path.join(testing_dataset, c,'*.tfrecords')))

# shuffle files
random.shuffle(train_val_file_list)
random.shuffle(test_filenames)

# some other settings
crop_size = (200,200) # (h, w)

test_examples_to_show = 50
train_examples_to_show = 100

## create data augmentation layers


augmentor = tf.keras.Sequential([
                layers.experimental.preprocessing.Normalization(),
                layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
                layers.experimental.preprocessing.RandomRotation(0.02, fill_mode='reflect', interpolation='nearest')],
            name='NormalizationAugmentationCropping')


## 2 print testing images
importlib.reload(utilities)

# build tf generator
test_dataloader = utilities.TFR_2D_dataset(test_filenames,
                dataset_type = 'train',
                batch_size=test_examples_to_show,
                buffer_size=1000,
                crop_size=(crop_size[0], crop_size[1]))

x, y = next(iter(test_dataloader))

x = augmentor(x, training=True)
sample = (x.numpy(), y.numpy())

print("Features mean: %.2f" % (x.numpy().mean()))
print("Features std: %.2f" % (x.numpy().std()))

utilities.show_batch_2D(sample, title='Test data')

## 3 print training images
importlib.reload(utilities)

# build tf generator
train_dataloader = utilities.TFR_2D_dataset(train_val_file_list,
                dataset_type = 'train',
                batch_size=train_examples_to_show,
                buffer_size=5000,
                crop_size=(crop_size[0], crop_size[1]))
x, y = next(iter(train_dataloader))
x = augmentor(x, training=True)
sample = (x.numpy(), y.numpy())

print("Features mean: %.2f" % (x.numpy().mean()))
print("Features std: %.2f" % (x.numpy().std()))

utilities.show_batch_2D(sample, title='Training data')



















