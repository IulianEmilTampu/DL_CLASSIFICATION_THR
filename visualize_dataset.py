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
from tensorflow.keras.utils import to_categorical
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

# custom imports
import utilities
import utilities_models_tf


## 1 specify the path to the TFR record database
importlib.reload(utilities)
# data to work on
dataset_path = '/home/iulta54/Desktop/Testing/TH_DL_dummy_dataset/Created/2D_isotropic_TFR'
file_names = glob.glob(os.path.join(dataset_path, '*'))
c_type='c1'
file_names, labels, organized_files = utilities.get_organized_files(file_names, c_type, categorical=False)

aus = list(zip(file_names, labels))
random.shuffle(aus)
file_names, labels = zip(*aus)

# some other settings
crop_size = (200,200) # (h, w)

examples_to_show = 50

## create data augmentation layers

augmentor = tf.keras.Sequential([
                layers.experimental.preprocessing.Normalization(),
                layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
                layers.experimental.preprocessing.RandomRotation(0.02)],
            name='NormalizationAugmentationCropping')


## 2 print testing images
importlib.reload(utilities)

# build tf generator
test_dataloader = utilities.TFR_2D_dataset(file_names,
                dataset_type = 'train',
                batch_size=examples_to_show,
                buffer_size=1000,
                crop_size=(crop_size[0], crop_size[1]))

x, y = next(iter(test_dataloader))

x = augmentor(x, training=True)
y = utilities_models_tf.fix_labels_v2(y.numpy(), classification_type=c_type, unique_labels=[2,3], categorical=False)
sample = (x.numpy(), y)

print("Features mean: %.2f" % (x.numpy().mean()))
print("Features std: %.2f" % (x.numpy().std()))

utilities.show_batch_2D(sample, title='TFR data')













