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
parser.add_argument('-tts', '--train_test_split', required=False, help='Provide the path to the train_test_split.json file specifying the test and training dataset.', default=None)
parser.add_argument('-mc', '--model_configuration', required=False, help='Provide the Model Configuration (LightOCT, M2, M3, ResNet50, VAE or others if implemented in the models_tf.py file).', default='LightOCT')
parser.add_argument('-mn', '--model_name', required=False, help='Provide the Model Name. This will be used to create the folder where to save the model. If not provided, the current datetime will be used', default=datetime.now().strftime("%H:%M:%S"))
parser.add_argument('-ct', '--classification_type', required=False, help='Provide the Classification Type. Chose between 1 (normal-vs-disease), 2 (normal-vs-enlarged-vs-shrinked) and 3 (normal-vs-all_diseases_available). If not provided, normal-vs-disease will be used.', default='c1')
parser.add_argument('-cct', '--custom_classification_type', required=False, help='If the classification type is custom (not one of the dfefault one). If true, training test split will be generated here instead of using the already available one in the dataset folder. Note that all the custom classification arte based on the per-disease class split.', default=False)
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
parser.add_argument('-db', '--debug', required=False, help='True if want to use a smaller portion of the dataset for debugging', default=False)
args = parser.parse_args()

# parse variables
working_folder = args.working_directory
dataset_folder = args.dataset_folder
train_test_split = args.train_test_split
model_configuration = args.model_configuration
model_save_name = args.model_name
classification_type = args.classification_type
custom_classification = args.custom_classification_type == 'True'
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
debug = args.debug == 'True'

# # parse variables
# working_folder = '/flush/iulta54/Research/P3-THR_DL/'
# dataset_folder = '/flush/iulta54/Research/Data/OCT/Thyroid_2019_refined_DeepLearning/2D_isotropic_TFR'
# train_test_split = '/flush/iulta54/Research/Data/OCT/Thyroid_2019_refined_DeepLearning/2D_isotropic_TFR/train_test_split_rollback.json'
# model_configuration = 'LightOCT'
# model_save_name = 'LightOCT_rollback'
# classification_type = 'c1'
# custom_classification = False
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
# imbalance_data_strategy = 'weights'
# kernel_size = [5,5]
# debug = False

# check if working folder and dataset folder exist
if os.path.isdir(working_folder):
    # check if the trained_model folder exists, if not create it
    if not os.path.isdir(os.path.join(working_folder, 'trained_models')):
        print('trained_model folders does not exist in the working path, creating it...')
        save_path = os.path.join(working_folder, 'trained_models')
        os.mkdir(save_path)
else:
    print('The provided working folder does not exist. Input a valid one. Given {}'.format(working_folder))
    sys.exit()

if not os.path.isdir(dataset_folder):
    print(f'The dataset folder provided does not exist. Input a valid one. Given {dataset_folder}')
    sys.exit()

if not custom_classification:
    # check if the train_test_split file is provided.
    if not os.path.isfile(train_test_split):
        raise ValueError('Custom classification is set to false, but give train_test_split file is not specified. Provide a valid one. Given {train_test_split}')

if debug:
    print(f'\n{"-"*70}')
    print(f'{"Configuration file script - running in debug mode (less training data)"}')
    print(f'{"-"*70}\n')
else:
    print(f'\n{"-"*25}')
    print(f'{"Configuration file script"}')
    print(f'{"-"*25}\n')

print(f'{"Working directory":<26s}: {working_folder}')
print(f'{"Model configuration":<26s}: {model_configuration}')
if model_configuration == 'VAE':
    print(f'{"VAE latent space dimension":<26s}: {vae_latent_dim}')
    print(f'{"VAE KL loss weight":<26s}: {vae_kl_weight}')
print(f'{"Model save name":<26s}: {model_save_name}')
print(f'{"Classification type":<26s}: {classification_type}')
print(f'{"Custom classification":<26s}: {custom_classification}')
print(f'{"Loss function":<26s}: {loss}')
print(f'{"Learning rate":<26s}: {learning_rate}')
print(f'{"Batch size":<26s}: {batch_size}')
print(f'{"Input size":<26s}: {input_size}')
print(f'{"Data augmentation":<26s}: {data_augmentation} ')


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

classification_type_dict['c4'] = {}
classification_type_dict['c4']['unique_labels'] = [0,[1,2],3,[4,5]]
classification_type_dict['c4']['class_labels'] = ['normal', 'Goiter', 'Adenoma', 'Hashimoto', 'Graves', 'Cancer']

# check if we are using a default classification type. If yes, use the train_test_split.json file

if custom_classification:
    print(f'\nClassification type is not a default one. Splitting the data accordingly.')
    print(f'{"Unique labels":<26s}: {classification_type_dict[classification_type]["unique_labels"]} ')
    # infere file extention from the dataset files
    _, extension = os.path.splitext(glob.glob(os.path.join(dataset_folder, '*'))[10])
    file_names = glob.glob(os.path.join(dataset_folder, '*'+extension))
    # use less data in debug mode
    if debug:
        print('Running in debug mode - using less training data \n')
        random.seed(29)
        random.shuffle(file_names)
        file_names = file_names[0:10000]
else:
    if (classification_type == 'c1' or classification_type == 'c2' or classification_type == 'c3'):
        if os.path.isfile(train_test_split):
            print(f'\nUsing default training test split (available in the train_test_split file)')
            print(f'{"Unique labels":<26s}: {classification_type_dict[classification_type]["unique_labels"]} ')
            with open(train_test_split) as file:
                split = json.load(file)
                train_val_filenames = split['training']
                print(f'Initial training files: {len(train_val_filenames)}')

                if classification_type == 'c1':
                    test_filenames = split['c1_test']
                if classification_type == 'c2':
                    test_filenames = split['c2_test']
                if classification_type == 'c3':
                    test_filenames = split['c3_test']

                # append basefolder and extention to the files
                # infere file extention from the dataset files
                _, extension = os.path.splitext(glob.glob(os.path.join(dataset_folder, '*'))[10])
                train_val_filenames = [os.path.join(dataset_folder, f+extension) for f in train_val_filenames]
                test_filenames = [os.path.join(dataset_folder, f+extension) for f in test_filenames]

                # debug modeaus = [os.path.basename(i[0:i.find('c1')-1]) for i in c]
                if debug:
                    print('Running in debug mode - using less training data (20000) \n')
                    # random.seed(29)
                    random.shuffle(train_val_filenames)
                    train_val_filenames = train_val_filenames[0:20000]
        else:
            raise ValueError(f'Using default classification type, but not train_test_split.json file found. Run the set_test_set.py first')
    else:
        raise ValueError(f'Custom classification type was set to False, but the given classification type is not a default one. Given {classification_type} expecting c1, c2 or c3.')


## split dataset into train+validation and test if custom classicifation
n_images_per_class = 1000
min_n_volumes = 2

if custom_classification:
    print('Working on the test dataset...')
    importlib.reload(utilities)
    '''
    Use n_images_per_class of at least 2 volumes for each class as test sample
    1 - find unique volumes for each class
    2 - randomly select volumes for test (to reach n_images_per_class images)
    3 - randomly select n_images_per_class from the selected volumes for each class
    4 - get all the remaining files for train+validation
    '''


    file_names, labels, per_class_file_names = utilities.get_organized_files(file_names,
                        classification_type=classification_type,
                        custom=True,
                        custom_labels=classification_type_dict[classification_type]['unique_labels'])
    n_classes = len(classification_type_dict[classification_type]['unique_labels'])

    # 1
    per_class_unique_volumes = []
    for c in per_class_file_names:
        # reduce the name to contain only the sample code and scan_code
        aus = [os.path.basename(i[0:i.find('c1')-1]) for i in c]
        per_class_unique_volumes.append(list(dict.fromkeys(aus)))

    # 2
    random.seed(29)
    per_class_random_files = []
    index_of_selected_files = []

    for c in per_class_unique_volumes:
        # for this class, shuffle the volumes and get all the images untill we reach the limit
        random.shuffle(c)
        count = 0
        idx = 0
        per_class_random_files.append([])
        while count <= n_images_per_class or idx < min_n_volumes:
            # get all the files from that volume
            indexes = [i for i, f in enumerate(file_names) if c[idx] in f]
            per_class_random_files[-1].extend([file_names[i] for i in indexes])
            index_of_selected_files.extend(indexes)
            count += len(indexes)
            idx += 1

    for i, c in enumerate(per_class_random_files):
        print(f'{"Unique labels:"+str(classification_type_dict[classification_type]["unique_labels"][i]):26s}: {len(c):4d} test files')

    # 3 get exactly n_images_per_class from each class and set it to the test set
    test_filenames = []
    for f in per_class_random_files:
        test_filenames.extend(random.sample(f, n_images_per_class))

    # 4 get the remaining training validation files
    train_val_filenames = [f for i, f in enumerate(file_names) if i not in index_of_selected_files]

# make sure that no training file is in the test set
for f in train_val_filenames:
    if f in test_filenames:
        raise ValueError('Train testing split did not go as planned. Check implementation')

## compute class weights on the training dataset and apply imbalance data strategy

train_val_filenames, train_val_labels, per_class_file_names = utilities.get_organized_files(train_val_filenames,
                    classification_type=classification_type,
                    custom= not (classification_type == 'c1' or classification_type == 'c2' or classification_type == 'c3'),
                    custom_labels=classification_type_dict[classification_type]['unique_labels'])

class_weights = np.array([len(i) for i in per_class_file_names])
if imbalance_data_strategy == 'oversampling':
    # get the class with highest number of elements
    better_represented_class = np.argmax(class_weights)
    num_sample_to_eversample = [class_weights[better_represented_class] - len(i) for i in per_class_file_names]

    # check if oversampling is reasonable (not replicate an entire dataset more
    # than 3 times).
    rep = 0

    for idx, i in enumerate(num_sample_to_eversample):
        # only oversample where is needed
        if i != 0:
            if int(i // len(per_class_file_names[idx])) > rep:
                rep = int(i // len(per_class_file_names[idx]))

    if rep < 50:
        # sample where needed and add to the training file names
        for idx, i in enumerate(num_sample_to_eversample):
            # only oversample where is needed
            if i != 0:
                n_class_samples = len(per_class_file_names[idx])
                train_val_filenames.extend(per_class_file_names[idx]*int(i // n_class_samples))
                train_val_filenames.extend(random.sample(per_class_file_names[idx], int(i % n_class_samples)))

        class_weights = np.ones(len(per_class_file_names))
        print(f'\nUsing {imbalance_data_strategy} strategy to handle imbalance data.')
        print(f'Setting loss function to cce given the oversampling strategy')
        loss = 'cce'
    else:
        print(f'Avoiding oversampling strategy since this will imply repeating one of the classes more that 3 times')
        print(f'Using class weights instead. Setting loss function to weighted categorical cross entropy (wcce)')
        imbalance_data_strategy = 'weights'
        class_weights = class_weights.sum() / class_weights**1
        class_weights = class_weights / class_weights.sum()
        loss = 'wcce'

elif imbalance_data_strategy == 'weights':
    print(f'\nUsing {imbalance_data_strategy} strategy to handle imbalance data.')
    print(f'Setting loss function to wcce given the oversampling strategy')
    class_weights = class_weights.sum() / class_weights**1
    class_weights = class_weights / class_weights.sum()
    loss = 'wcce'

n_train = len(train_val_filenames)
n_test = len(test_filenames)

print(f'\nWill train and validate on {n_train} images (some might have been removed since not classifiebly in this task)')
print(f'Will test on {n_test} images ({n_images_per_class} for each class)')
print(f'{"Class weights":<10s}: {class_weights}')

## prepare for cross validation
'''
Make sure that images from the same volumes are not in the both the training and
validation sets. So, as before, we take out the volume names, select train and
validation for every fold and then save the images belonging to that volumes.
1 - get unique volumes for each class
2 - split each class independently for cross validation
3 - save file names for each fold
'''
print(f'\nSetting cross-validation files...')
# 1
per_class_unique_volumes = []
for c in per_class_file_names:
    # reduce the name to contain only the sample code and scan_code
    aus = [os.path.basename(i[0:i.find('c1')-1]) for i in c]
    per_class_unique_volumes.append(list(dict.fromkeys(aus)))
    random.shuffle(per_class_unique_volumes[-1])


# for i, c in enumerate(classification_type_dict[classification_type]['unique_labels']):
#     for v in per_class_unique_volumes[i]:
#         print(v)

N_FOLDS = 1

# 2
if N_FOLDS >= 2:
    kf = KFold(n_splits=N_FOLDS)
    per_fold_train_files = [[] for i in range(N_FOLDS)]
    per_fold_val_files = [[] for i in range(N_FOLDS)]

    for idx1, c in enumerate(per_class_unique_volumes):
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
                # print(f'Fold {idx} - Validation: {val_vol}')
                per_fold_val_files[idx].extend([f for f in train_val_filenames if val_vol in f])

    # shuffle training files (since that there can be many files, the buffer size
    # for the generator should be very large. By shuffling now we can reduce the
    # buffer size).

    for c in range(N_FOLDS):
        random.shuffle(per_fold_train_files[c])
        random.shuffle(per_fold_val_files[c])

else:
    # set 1000 images from each class as validation (like the testing)
    n_images_per_class = 1000
    per_fold_train_files = [[] for i in range(N_FOLDS)]
    per_fold_val_files = [[] for i in range(N_FOLDS)]

    random.seed(29)
    per_class_random_files = []
    index_of_selected_files = []

    # randomly select as many volumes per class as needed to reach n_images_per_class
    for c in per_class_unique_volumes:
        # for this class, shuffle the volumes and get all the images untill we reach the limit
        random.shuffle(c)
        count = 0
        idx = 0
        per_class_random_files.append([])
        while count <= n_images_per_class or idx < min_n_volumes:
            # get all the files from that volume
            indexes = [i for i, f in enumerate(train_val_filenames) if c[idx] in f]
            per_class_random_files[-1].extend([train_val_filenames[i] for i in indexes])
            index_of_selected_files.extend(indexes)
            count += len(indexes)
            idx += 1

    # 3 get exactly n_images_per_class from each class and set it to the test set
    for f in per_class_random_files:
        per_fold_val_files[0].extend(random.sample(f, n_images_per_class))

    # 4 get the remaining training validation files
    per_fold_train_files[0] = [f for i, f in enumerate(train_val_filenames) if i not in index_of_selected_files]

# make sure that no training file is in the test set
for f in per_fold_train_files[0]:
    if f in  per_fold_val_files[0]:
        raise ValueError('Train testing split did not go as planned. Check implementation')

# check that the split is valid
for c in range(N_FOLDS):
    for tr_f in per_fold_train_files[c]:
        if tr_f in per_fold_val_files[c]:
            print(f'File {os.path.basename(tr_f)} in both set for fold {c}')
            raise ValueError('Train validation split did not go as planned \n Some training file are in the validation set. Check implementation')

for c in range(N_FOLDS):
    random.shuffle(per_fold_train_files[c])
    random.shuffle(per_fold_val_files[c])

# check that the split is valid
for c in range(N_FOLDS):
    for tr_f in per_fold_train_files[c]:
        if tr_f in per_fold_val_files[c]:
            print(f'File {os.path.basename(tr_f)} in both set for fold {c}')
            raise ValueError('Train validation split did not go as planned \n Some training file are in the validation set. Check implementation')

print(f'Cross-validation set. Running a {N_FOLDS}-fold cross validation')
print(f'Images from the validation set are taken from volumes not in the training sets')
for f in range(N_FOLDS):
    print(f'Fold {f+1}: training on {len(per_fold_train_files[f]):5d} and validation on {len(per_fold_val_files[f]):5d}')


## Save all the information in a configuration file
'''
The configuration file will be used by the training routine to access the
the train-val-test files as well as the different set-up for the model. Having a
separate configuration file helps keeping the training routine more clean.
'''
print(f'\nSavingconfiguration file...')

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
save_model_path = os.path.join(working_folder, 'trained_models', model_save_name)

if not os.path.isdir(save_model_path):
    os.mkdir(save_model_path)

json_dict['save_model_path'] = save_model_path


with open(os.path.join(save_model_path,'config.json'), 'w') as fp:
    json.dump(json_dict, fp)

print(f'Configuration file created. Avvailable at {save_model_path}')






