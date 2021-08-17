'''
Main script that runs the training of a deep leaning model for classification
of 2D OCT thyroid images (normal vs diseased or disease type).

Steps
- create trainig/validation/test dataloader with on-the-fly augmentation
- load the CNN model and define loss function and training hyperparamters
- train the model
- save trained model along with training curves.
- run testing and save model performance
'''

import os
import sys
import json
import glob
import types
import time
import random
import argparse
import importlib
import numpy as np
import nibabel as nib
from random import shuffle
from datetime import datetime
import matplotlib.pyplot as plt
from operator import itemgetter
from shutil import copyfile, move
from collections import OrderedDict
from sklearn.model_selection import KFold


import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# custom imports
import models_tf
import utilities
import utilities_models_tf

## parse inline parameters

parser = argparse.ArgumentParser(description='Script that runs a cross-validation training for OCT 2D image classification.')
parser.add_argument('-wd','--working_directory' ,required=False, help='Provide the Working Directory where the models_tf.py, utilities.py and utilities_models_tf.py files are.This folder will also be the one where the trained models will be saved. If not provided, the current working directory is used', default=os.getcwd())
parser.add_argument('-df', '--dataset_folder', required=True, help='Provide the Dataset Folder where the Train and Test folders are present along with the dataset information file.')
parser.add_argument('-mc', '--model_configuration', required=False, help='Provide the Model Configuration (LightOCT, M2, M3, ResNet50, VAE or others if implemented in the models_tf.py file).', default='LightOCT')
parser.add_argument('-mn', '--model_name', required=False, help='Provide the Model Name. This will be used to create the folder where to save the model. If not provided, the current datetime will be used', default=datetime.now().strftime("%H:%M:%S"))
parser.add_argument('-ct', '--classification_type', required=False, help='Provide the Classification Type. Chose between 1 (normal-vs-disease), 2 (normal-vs-enlarged-vs-shrinked) and 3 (normal-vs-all_diseases_available). If not provided, normal-vs-disease will be used.', default=1)
parser.add_argument('-f', '--folds', required=False, help='Number of folds. Default is 3', default='3')
parser.add_argument('-l', '--loss', required=False, help='Loss to use to train the model (cce, wcce or sfce). Default is cce', default='cce')
parser.add_argument('-lr', '--learning_rate', required=False, help='Learning rate.', default=0.001)
parser.add_argument('-bs', '--batch_size', required=False, help='Batch size.', default=50)
parser.add_argument('-is', '--input_size', nargs='+', required=False, help='Model input size.', default=(200,200))
parser.add_argument('-ks', '--kernel_size', nargs='+', required=False, help='Encoder conv kernel size.', default=(5,5))
parser.add_argument('-augment', '--augmentation', required=False, help='Specify if data augmentation is to be performed (True) or not (False)', default=True)
parser.add_argument('-vld', '--vae_latent_dim', required=False, help='Dimension of the VAE latent space', default=128)
parser.add_argument('-vkl', '--vae_kl_weight',required=False, help='KL weight in for the VAE loss', default=0.1)
parser.add_argument('-vrl', '--vae_reconst_weight',required=False, help='Reconstruction weight in for the VAE loss', default=0.1)
parser.add_argument('-v', '--verbose',required=False, help='How much to information to print while training: 0 = none, 1 = at the end of an epoch, 2 = detailed progression withing the epoch.', default=0.1)
parser.add_argument('-ids', '--imbalance_data_strategy', required=False, help='Strategy to use to tackle imbalance data', default='weights')
args = parser.parse_args()

# parse variables
working_folder = args.working_directory
dataset_folder = args.dataset_folder
model_configuration = args.model_configuration
model_save_name = args.model_name
classification_type = args.classification_type
loss = args.loss
learning_rate = float(args.learning_rate)
batch_size = int(args.batch_size)
input_size = [int(i) for i in args.input_size]
data_augmentation = args.augmentation
vae_latent_dim = int(args.vae_latent_dim)
vae_kl_weight = float(args.vae_kl_weight)
vae_reconst_weight = float(args.vae_reconst_weight)
N_FOLDS = int(args.folds)
verbose = int(args.verbose)
imbalance_data_strategy = args.imbalance_data_strategy
kernel_size = [int(i) for i in args.kernel_size]

# # parse variables
# working_folder = '/flush/iulta54/Research/P3-THR_DL/'
# dataset_folder = '/home/iulta54/Desktop/Testing/TH_DL_dummy_dataset/Created/2D_isotropic_TFR'
# model_configuration = 'LightOCT'
# model_save_name = 'Test_new_dataset_LightOCT'
# classification_type = 'c1'
# loss = 'cce'
# learning_rate = 0.0001
# batch_size = 100
# input_size = [200, 200]
# data_augmentation = True
# vae_latent_dim = 128
# vae_kl_weight = 0.1
# vae_reconst_weight = 0.1
# N_FOLDS = 3
# verbose = 2
# imbalance_data_strategy = 'oversampling'
# kernel_size = [5,5]

# check if working folder and dataset folder exist
if os.path.isdir(working_folder):
    # check if the trained_model folder exists, if not create it
    if not os.path.isdir(os.path.join(working_folder, 'trained_models_test')):
        print('trained_model folders does not exist in the working path, creating it...')
        save_path = os.path.join(working_folder, 'trained_models_test')
        os.mkdir(save_path)
else:
    print('The provided working folder does not exist. Input a valid one. Given {}'.format(working_folder))
    sys.exit()

if not os.path.isdir(dataset_folder):
    print(f'The dataset folder provided does not exist. Input a valid one. Given {dataset_folder}')
    sys.exit()

print('working directory - {}'.format(working_folder))
print('model configuration - {}'.format(model_configuration))
if model_configuration == 'VAE':
    print(' - VAE latent space dimension:{}'.format(vae_latent_dim))
    print(' - VAE KL loss weight:{}'.format(vae_kl_weight))
print('model save name - {}'.format(model_save_name))
print('classification type - {}'.format(classification_type))
print('Loss function - {}'.format(loss))
print('Learning rate - {}'.format(learning_rate))
print('Batch size - {}'.format(batch_size))
print('Input size - {}'.format(input_size))
print('Data augmentation - {} \n\n'.format(data_augmentation))

## get all file names, configure based on classification type and unique labels
'''
Based on the classification type, the labels change meaning. For the default
classification types, here are the description:
- c1 : binary classification
    - 0: normal
    - 1: abnormal
- c2 : 4-class classification
    - 0 : normal
    - 1 : enlarged
    - 2 : shrunk
    - 3 : depleted
- c3 : 6-class classification
    - 0 : normal
    - 1 : Goiter
    - 2 : Adenoma
    - 3 : Hashimoto
    - 4 : Graves
    - 5 : Cancer

'''
classification_type_dict = {}

classification_type_dict['c1'] = {}
classification_type_dict['c1']['unique_labels'] = [0,1]
classification_type_dict['c1']['class_labels'] = ['normal', 'abnormal']


classification_type_dict['c2'] = {}
classification_type_dict['c2']['unique_labels'] = [0,1,2,3]
classification_type_dict['c2']['class_labels'] = ['normal', 'enlarged', 'shrunk', 'depleted']

classification_type_dict['c3'] = {}
classification_type_dict['c3']['unique_labels'] = [0,1,2,3,4,5]
classification_type_dict['c3']['class_labels'] = ['normal', 'Goiter', 'Adenoma', 'Hashimoto', 'Graves', 'Cancer']



file_names = glob.glob(os.path.join(dataset_folder, '*'))
file_names, labels, per_class_file_names = utilities.get_organized_files(file_names,
                    classification_type=classification_type,
                    custom=True,
                    custom_labels=classification_type_dict[classification_type]['unique_labels'])
n_classes = len(classification_type_dict[classification_type]['unique_labels'])
## split dataset into train+validation and test
'''
Use the images of 2 volumes for each class as test sample
1 - find unique volumes for each class
2 - randomly select volumes for test
3 - get all the images belonging to those volumes and save them for testing
4 - get all the remaining files for train+validation
'''
# 1
per_class_unique_volumes = []
for c in per_class_file_names:
    # reduce the name to contain only the sample code and scan_code
    aus = [os.path.basename(i[0:i.find('c1')-1]) for i in c]
    per_class_unique_volumes.append(list(dict.fromkeys(aus)))

# 2
n_test_volumes_per_class = 0
random.seed(29)
test_volumes = []
for c in per_class_unique_volumes:
    test_idx = random.sample(range(0, len(c)), n_test_volumes_per_class)
    test_volumes.append([c[i] for i in test_idx])

# 3
test_filenames = []
test_indexes = []
for c in test_volumes:
    for vol in c:
        indexes = [i for i, x in enumerate(file_names) if vol in x]
        test_indexes.extend(indexes)
        test_filenames.extend([file_names[i] for i in indexes])

train_val_filenames = [file_names[i] for i in range(len(file_names)) if i not in test_indexes]

n_train = len(train_val_filenames)
n_test = len(test_filenames)

# make sure that no training file is in the test set
for f in train_val_filenames:
    if f in test_filenames:
        raise ValueError('Train testing split did not go as planned. Check implementation')

## compute class weights on the training dataset and apply imbalance data strategy
train_val_filenames, train_val_labels, per_class_file_names = utilities.get_organized_files(train_val_filenames,
                    classification_type=classification_type,
                    custom=True,
                    custom_labels=classification_type_dict[classification_type]['unique_labels'])

class_weights = np.array([len(i) for i in per_class_file_names])
print(f'Using {imbalance_data_strategy} strategy to handle imbalance data.')
if imbalance_data_strategy == 'oversampling':
    # get the class with highest number of elements
    better_represented_class = np.argmax(class_weights)
    num_sample_to_eversample = [class_weights[better_represented_class] - len(i) for i in per_class_file_names]

    # sample where needed and add to the training file names
    for idx, i in enumerate(num_sample_to_eversample):
        # only oversample where is needed
        if i != 0:
            n_class_samples = len(per_class_file_names[idx])
            train_val_filenames.extend(per_class_file_names[idx]*int(i // n_class_samples))
            train_val_filenames.extend(random.sample(per_class_file_names[idx], int(i % n_class_samples)))

    class_weights = np.ones(n_classes)
    print('Setting loss function to cce given the oversampling strategy')
    loss = 'cce'
elif imbalance_data_strategy == 'weights':
    class_weights = class_weights.sum() / class_weights**1
    class_weights = class_weights / class_weights.sum()

print('Class weights -> {}'.format(class_weights))

## prepare for cross validation
'''
Make sure that images from the same volumes are not in the both the training and
validation sets. So, as before, we take out the volume names, select train and
validation for every fold and then save the images belonging to that volumes.
1 - get unique volumes for each class
2 - split each class independently for cross validation
3 - save file names for each fold
'''
# 1
per_class_unique_volumes = []
for c in per_class_file_names:
    # reduce the name to contain only the sample code and scan_code
    aus = [os.path.basename(i[0:i.find('c1')-1]) for i in c]
    per_class_unique_volumes.append(list(dict.fromkeys(aus)))

# 2
N_FOLDS = 2
kf = KFold(n_splits=N_FOLDS)
per_fold_train_files = [[] for i in range(N_FOLDS)]
per_fold_val_files = [[] for i in range(N_FOLDS)]
for c in per_class_unique_volumes:
    # for all classes
    for idx, (train_volume_index, val_volume_index) in enumerate(kf.split(c)):
        # use the indexes of the unique volumes for split the data
        # training
        for v in train_volume_index:
            tr_vol = c[v]
            per_fold_train_files[idx].extend([f for f in train_val_filenames if tr_vol in f])
        # validation
        for v in val_volume_index:
            val_vol = c[v]
            per_fold_val_files[idx].extend([f for f in train_val_filenames if val_vol in f])

# check that the split is valid
for c in range(N_FOLDS):
    for tr_f in per_fold_train_files[c]:
        if tr_f in per_fold_val_files[c]:
            print(f'File {os.path.basename(tr_f)} in both set for fold {c}')
            raise ValueError('Train validation split did not go as planned \n Some training file are in the validation set. Check implementation')

print('Cross-validation set. Running a {}-fold cross validation'.format(N_FOLDS))


## Save all the information in a configuration file
'''
The configuration file will be used by the training routine to access the
the train-val-test files as well as the different set-up for the model. Having a
separate configuration file helps keeping the training routine more clean.
'''
json_dict = OrderedDict()
json_dict['working_folder'] = working_folder
json_dict['dataset_folder'] = dataset_folder

json_dict['classification_type'] = classification_type
json_dict['unique_labels'] = classification_type_dict[classification_type]['unique_labels']
json_dict['label_description'] =classification_type_dict[classification_type]['class_labels']

json_dict['model_configuration'] = model_configuration
json_dict['model_save_name'] = model_save_name
json_dict['loss'] = loss
json_dict['learning_rate'] = learning_rate
json_dict['batch_size'] = batch_size
json_dict['input_size'] = input_size
json_dict['kernel_size'] = kernel_size
json_dict['data_augmentation'] = data_augmentation

if 'VAE' in model_configuration:
    json_dict['vae_latent_dim'] = vae_latent_dim
    json_dict['vae_kl_weight'] = vae_kl_weight
    json_dict['vae_reconst_weight'] = vae_reconst_weight

json_dict['N_FOLDS'] = N_FOLDS
json_dict['verbose'] = verbose
json_dict['imbalance_data_strategy'] = imbalance_data_strategy

json_dict['training'] = per_fold_train_files
json_dict['validation'] = per_fold_val_files
json_dict['test'] = test_filenames
json_dict['class_weights'] = list(class_weights)

# save file
save_model_path = os.path.join(working_folder, 'trained_models_test', model_save_name)

if not os.path.isdir(save_model_path):
    os.mkdir(save_model_path)

json_dict['save_model_path'] = save_model_path


with open(os.path.join(save_model_path,'config.json'), 'w') as fp:
    json.dump(json_dict, fp)

print(f'Configuration file created. Avvailable at {save_model_path}')






