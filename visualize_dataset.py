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


## 1 specify the path to the TFR record database
importlib.reload(utilities)

def get_organized_files(file_names, classification_type,
                        return_labels=True,
                        categorical=False,
                        custom=False,
                        custom_labels=None):
    '''
    Utility that given a list of file names using the convention described in
    the create_dataset_v2.py script, returns three things:
    1 - list of files that does not contain the file smarked as to be excluded
    2 - list of labels corresponding to the files above (categprical or not)
    3 - a list of lists that contains the files organised per aggregation class

    Parameters
    ----------
    file_names : list of str
        Identifies the file names
    classification_type : str
        Specifies the classification type as described in the
        create_dataset_v2.py script.
    custom : bool
        Specifies if the labels need to be aggregated in a different way from
        default. The default is that every labels is an independent class. If
        False, the labels will be clastered based on the specifications given in
        the custom_labels. Default is False.
    custom_labels : list
        Specifies the way the labels should be clustered. Used if custom
        parameter is set to True.
        # For example:
        [
        0,
        [1, 3],
        [2, 4, 5],
        6
        ]

    will return categorical labels where labels are 0, 1, 2 and 3 with:
        - 0 having images from class 0;
        - 1 having images from classes 1 and 3
        - 2 having images from classes 2, 4, 5
        - 3 having images from class 6

    categorical : bool
        If True, returns the labels in categorical form.
    '''

    # check that the inputs are correct
    if isinstance(file_names, list):
        # loop through all the elements in the list and make sure they are
        # strings and they match the convention
        for file in file_names:
            # try to get the labels
            c1 = int(file[file.find('c1')+3])
            c2 = int(file[file.find('c2')+3])
            c3 = int(file[file.find('c3')+3])
    else:
        raise TypeError(f'file_name expected to be a list, but give {type(file_names)}')

    if not isinstance(classification_type, str):
        raise TypeError(f'classification_type expected to be a list, but give {type(classification_type)}')
    else:
        # chack that it specifies a know classification type
        if not (classification_type=='c1' or classification_type=='c2' or classification_type=='c3'):
            raise ValueError(f'classification_type expected to be c1, c2 or c3. Instead was given {classification_type}')

    if custom:
        # custom label aggregation given, thus checking if custom_labels is given
        if custom_labels:
            # chack that is a list
            if not isinstance(custom_labels, list):
                raise TypeError(f'custom_labels expected to be a list, but given {type(custom_labels)}')
        else:
            raise ValueError('custom was set to True, but no custom_labels specification was given.')

    # get labels for the specified classification type and exclude label 9
    # (flags volumes to not be used)
    raw_labels = []
    filtered_file_names = []
    for file in file_names:
        label = int(file[file.find(classification_type)+3])
        if label != 9:
            raw_labels.append(label)
            filtered_file_names.append(file)

    # aggregate base on the specification
    if custom:
        # use custom aggregation
        organized_files = [[] for i in range(len(custom_labels))]
        labels = np.zeros(len(filtered_file_names))
        for idx, l in enumerate(custom_labels):
            if type(l) is list:
                for ll in l:
                    indexes = [i for i, x in enumerate(raw_labels) if x==ll]
                    organized_files[idx].extend([filtered_file_names[i] for i in indexes])
                    labels[indexes] = idx
            else:
                indexes = [i for i, x in enumerate(raw_labels) if x==l]
                organized_files[idx].extend([filtered_file_names[i] for i in indexes])
                labels[indexes] = idx
    else:
        # use default aggredation
        organized_files = [[] for i in range(np.unique(raw_labels).shape[0])]
        labels = np.zeros((len(filtered_file_names)))
        for idx, l in enumerate(np.unique(raw_labels)):
            indexes = [i for i, x in enumerate(raw_labels) if x == l]
            organized_files[idx].extend([filtered_file_names[i] for i in indexes])
            labels[indexes] = idx

    if categorical == True:
        # convert labels to categorical
        labels = to_categorical(labels, num_classes=np.unique(labels).shape[0])

    return filtered_file_names, labels, organized_files

# data to work on
dataset_path = '/home/iulta54/Desktop/Testing/TH_DL_dummy_dataset/Created/2D_isotropic_TFR'
file_names = glob.glob(os.path.join(dataset_path, '*'))

filtered_file_names, labels, organized_files = get_organized_files(file_names, 'c1', categorical=False)

aus = list(zip(filtered_file_names, labels))
random.shuffle(aus)
filtered_file_names, labels = zip(*aus)

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
test_dataloader = utilities.TFR_2D_dataset(filtered_file_names,
                dataset_type = 'train',
                batch_size=examples_to_show,
                buffer_size=1000,
                crop_size=(crop_size[0], crop_size[1]))

x, y = next(iter(test_dataloader))

x = augmentor(x, training=True)
sample = (x.numpy(), y.numpy()[:,2])

print("Features mean: %.2f" % (x.numpy().mean()))
print("Features std: %.2f" % (x.numpy().std()))

utilities.show_batch_2D(sample, title='TFR data')













