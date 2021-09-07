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
    model_name = 'LightOCT_rollback'
    trained_models_path = '/flush/iulta54/Research/P3-THR_DL/trained_models'
    dataset_path = '/flush/iulta54/Research/Data/OCT/Thyroid_2019_refined_DeepLearning'

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

examples_to_show = 50

## 2 create dataset and augmentation layer
importlib.reload(utilities)

# build tf generator
test_dataloader = utilities.TFR_2D_dataset(tr_img,
                dataset_type = 'train',
                batch_size=examples_to_show,
                buffer_size=5000,
                crop_size=(crop_size[0], crop_size[1]))

# set normalization layer on the training dataset
tr_feature_ds = test_dataloader.map(tf.autograph.experimental.do_not_convert(lambda x, y: x))
normalizer = layers.experimental.preprocessing.Normalization(axis=-1)
normalizer.adapt(tr_feature_ds)

augmentor = tf.keras.Sequential([layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
                layers.experimental.preprocessing.RandomRotation(0.02)],
            name='NormalizationAugmentationCropping')

## 2 create data augmentation layers
importlib.reload(utilities)

x, y = next(iter(test_dataloader))

# x = normalizer(x)
x = augmentor(x, training=False)
y = utilities_models_tf.fix_labels_v2(y.numpy(), classification_type=config['classification_type'], unique_labels=config['unique_labels'], categorical=False)
sample = (x.numpy(), y.numpy())

print(f'{"Mean:":5s}{x.numpy().mean():0.2f}')
print(f'{"STD:":5s}{x.numpy().std():0.2f}')

utilities.show_batch_2D(sample, img_per_row=5)

## show examples with histogram
importlib.reload(utilities)

utilities.show_batch_2D_with_histogram(sample)






