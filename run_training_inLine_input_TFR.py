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


def check_dataset(dataset_folder):
    '''
    Ausiliary function that checks that the dataset provided by the user is in
    required format:
    .../dataset_folder/
    ├── Train
    │   ├── class_1
    │   │   ├── TH01_0001_0001_label_1.tfrecords
    │   │   ├── TH01_0002_0001_label_1.tfrecords
    │   │   ├── TH01_0003_0001_label_1.tfrecords
    │   ├── class_2
    │   │   ├── TH01_0001_0001_label_2.tfrecords
    │   │   ├── TH01_0002_0001_label_2.tfrecords
    │   │   ├── TH01_0003_0001_label_2.tfrecords
    │   ├── ...
    ├── Test
    │   ├── class_1
    │   │   ├── TH02_0001_0001_label_1.tfrecords
    │   │   ├── TH02_0002_0001_label_1.tfrecords
    │   │   ├── TH02_0003_0001_label_1.tfrecords
    │   ├── class_2
    │   │   ├── TH03_0001_0001_label_2.tfrecords
    │   │   ├── TH03_0002_0001_label_2.tfrecords
    │   │   ├── TH03_0003_0001_label_2.tfrecords
    │   ├── ...

    Steps:
    1 - check that the main folder exists
    2 - check that the Train, Test and dataset_info.json exist
    3 - in each Train and Test, check the subfolders and that these ara named
        class_integer-value
    4 - check that each file in a sfecific class_x is a nifti file and ends with
        label_x
    '''
    # 1 check main folder
    if not os.path.isdir(dataset_folder):
        print('Dataset folder does not exist. Given {}'.format(dataset_folder))
        sys.exit()
    else:
        # 2 - check the existance of Test, Train and dataset_info.json
        if not os.path.isfile(os.path.join(dataset_folder, 'dataset_info.json')):
            print('dataset infromation file not found. Input a valid dataset directory. Given {}'.format(dataset_folder))
            sys.exit()
        if not os.path.isdir(os.path.join(dataset_folder, 'Train')):
            print('Train folder not found. Input a valid dataset directory. Given {}'.format(dataset_folder))
            sys.exit()
        if not os.path.isdir(os.path.join(dataset_folder, 'Test')):
            print('Test folder not found. Input a valid dataset directory. Given {}'.format(dataset_folder))
            sys.exit()
        # 3 - in Train and Test, check that each subfolder is named correctly -> class_x where x is an integer
        folder1 = next(os.walk(dataset_folder))[1]
        for f1 in folder1:
            folder2 = next(os.walk(os.path.join(dataset_folder, f1)))[1]
            for f2 in folder2:
                # check folder name
                if (f2[0:6] == 'class_' and f2[6::].isdigit()):
                    # all in order, check all files
                    files = next(os.walk(os.path.join(dataset_folder, f1, f2)))[2]
                    for f3 in files:
                        # check that are nifti files and that the the file name
                        # ends with label_x
                        if not (f3[-10::]=='.tfrecords' and f3[f3.index('label_')+len('label_'):-10]==f2[6::]):
                            print('Found invalid file in {} folder. {}'.format(f2, f3))
                            sys.exit()

                else:
                    print('Invalid class folder. Found {} in {}.'.format(f2, f1))
                    sys.exit()
        # all good.
        print('Dataset folder checked. All good!')

check_dataset(dataset_folder)

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


## load dataset file names and compute class weights

train_folder = os.path.join(dataset_folder, 'Train')
test_folder = os.path.join(dataset_folder, 'Test')

'''
¤¤¤¤¤¤¤¤¤¤¤ Class descriptions
class_0: normal
class_1: goiter
class_2: Hashimoto
class_3: adenoma
class_4: cancer
class_5: Graves


Here we create dictionary with possible class combinations:
1 - normal-vs-diseased classification
2 - normal-vs-enlarged-vs-atrophic (3 class classification)
3 - normal-vs-all-disease (6 class classification)
'''
classification_type_dict = {}

classification_type_dict['1'] = {}
classification_type_dict['1']['unique_labels'] = ['class_0',
                 ['class_1','class_2','class_3','class_4','class_5']
                 ]
classification_type_dict['1']['class_labels'] = ['normal', 'disease']


classification_type_dict['2'] = {}
classification_type_dict['2']['unique_labels'] = ['class_0',
                 'class_1',
                 ['class_2','class_3','class_4','class_5']
                 ]
classification_type_dict['2']['class_labels'] = ['normal', 'enlarged', 'shrinked']

classification_type_dict['3'] = {}
classification_type_dict['3']['unique_labels'] = ['class_0',
                 'class_1',
                 'class_2',
                 'class_3',
                 'class_4',
                 'class_5'
                 ]
classification_type_dict['3']['class_labels'] = ['normal', 'goiter', 'Hashimoto', 'adenoma', 'cancer', 'Graves']

# choose one of the classification type
unique_labels = classification_type_dict[str(classification_type)]['unique_labels']
class_labels = classification_type_dict[str(classification_type)]['class_labels']

print('Unique labels {}'.format(unique_labels))
print('Class labels {}'.format(class_labels))

# get file names and class weights based on the unique_label specification
# also over-sample the classes that are underrepresented
train_val_file_list = []
test_filenames = []
class_weights = np.zeros(len(unique_labels))

per_class_training_file_list = {}

for idx1, c in enumerate(unique_labels):
    if type(c) is list:
        aus = []
        for cc in c:
            class_weights[idx1] += len(glob.glob(os.path.join(train_folder, cc,'*.tfrecords')))
            train_val_file_list.extend(glob.glob(os.path.join(train_folder, cc,'*.tfrecords')))
            test_filenames.extend(glob.glob(os.path.join(test_folder, cc,'*.tfrecords')))
            aus.extend(glob.glob(os.path.join(train_folder, cc,'*.tfrecords')))
        per_class_training_file_list[idx1] = aus
    else:
        class_weights[idx1] += len(glob.glob(os.path.join(train_folder, c,'*.tfrecords')))
        train_val_file_list.extend(glob.glob(os.path.join(train_folder, c,'*.tfrecords')))
        test_filenames.extend(glob.glob(os.path.join(test_folder, c,'*.tfrecords')))
        per_class_training_file_list[idx1] = glob.glob(os.path.join(train_folder, c,'*.tfrecords'))

print('Using {} strategy to handle imbalance data.'.format(imbalance_data_strategy))
if imbalance_data_strategy == 'oversampling':
    # get the class with highest number of elements
    better_represented_class = np.argmax(class_weights)
    num_sample_to_eversample = [class_weights[better_represented_class] - len(value) for key, value in per_class_training_file_list.items()]

    # sample where needed and add to the training file names
    for idx, i in enumerate(num_sample_to_eversample):
        # only oversample where is needed
        if i != 0:
            n_class_samples = len(per_class_training_file_list[idx])
            train_val_file_list.extend(per_class_training_file_list[idx]*int(i // n_class_samples))
            train_val_file_list.extend(random.sample(per_class_training_file_list[idx], int(i % n_class_samples)))

    class_weights = np.ones(len(unique_labels))
    print('Setting loss function to cce given the oversampling strategy')
    loss = 'cce'
elif imbalance_data_strategy == 'weights':
    class_weights = class_weights.sum() / class_weights**1
    class_weights = class_weights / class_weights.sum()

print('Class weights -> {}'.format(class_weights))

## prepare for cross validation

n_train = len(train_val_file_list)
n_test = len(test_filenames)

# get ready training dataGenerator
# shuffle dataset
seed = 29
random.seed(seed)
random.shuffle(train_val_file_list)
# train_val_file_list = train_val_file_list[0:1000]

# cross-validation split
kf = KFold(n_splits=N_FOLDS)

train_filenames = []
validation_filenames = []
for train_index, val_index in kf.split(train_val_file_list):
    train_filenames.append(itemgetter(*train_index)(train_val_file_list))
    validation_filenames.append(itemgetter(*val_index)(train_val_file_list))

print('Cross-validation set. Running a {}-fold cross validation'.format(N_FOLDS))

## CROSS_VALIDATION TRAINING

print('Starting cross validation training')

# initialize variable to save test information and others
test_fold_summary = {}

save_model_path = os.path.join(working_folder, 'trained_models', model_save_name)

if not os.path.isdir(save_model_path):
    os.mkdir(save_model_path)

for cv in range(N_FOLDS):
    if not os.path.isdir(os.path.join(save_model_path, 'fold_'+str(cv+1))):
        os.mkdir(os.path.join(save_model_path, 'fold_'+str(cv+1)))

# ################ save lists of training, validation and testing file names in the model folder
train_val_filenames = {
    'Training' : [[os.path.basename(y) for y in x] for x in train_filenames],
    'Validation': [[os.path.basename(y) for y in x] for x in validation_filenames],
    'Testing': [os.path.basename(x) for x in test_filenames],
    'unique_labels': unique_labels,
    'class_labels': class_labels
    }
with open(os.path.join(save_model_path, 'train_val_test_filenames_json.txt'), 'w') as outfile:
    json.dump(train_val_filenames, outfile)

# ############################ TRAINING
for cv in range(N_FOLDS):
    print('Working on fold {}/{}. Start time {}'.format(cv+1,N_FOLDS, datetime.now().strftime("%H:%M:%S")))

    print(' - Creating datasets...')
    # get the file names for training and validation
    X_train = train_filenames[cv]
    X_val = validation_filenames[cv]

    # create datasets
    train_dataset = utilities.TFR_2D_dataset(X_train,
                    dataset_type = 'train',
                    batch_size=batch_size,
                    buffer_size=1000,
                    crop_size=input_size)

    val_dataset = utilities.TFR_2D_dataset(X_val,
                    dataset_type = 'train',
                    batch_size=batch_size,
                    buffer_size=1000,
                    crop_size=input_size)

    # create model based on specification
    if model_configuration == 'LightOCT':
        model = models_tf.LightOCT(number_of_input_channels = 1,
                        model_name='LightOCT',
                        num_classes = len(unique_labels),
                        data_augmentation=data_augmentation,
                        class_weights = class_weights,
                        kernel_size=(5,5),
                        input_size=input_size
                        )
    elif model_configuration == 'M2':
        model = models_tf.M2(number_of_input_channels = 1,
                        model_name='M2',
                        num_classes = len(unique_labels),
                        data_augmentation=data_augmentation,
                        class_weights = class_weights,
                        kernel_size=(5,5)
                        )
    elif model_configuration == 'M3':
        model = models_tf.M3(number_of_input_channels = 1,
                        model_name='M3',
                        num_classes = len(unique_labels),
                        data_augmentation=data_augmentation,
                        class_weights = class_weights,
                        kernel_size=(5,5)
                        )
    elif model_configuration == 'ResNet50':
        model = models_tf.ResNet50(number_of_input_channels = 1,
                        model_name='ResNet50',
                        num_classes = len(unique_labels),
                        data_augmentation=data_augmentation,
                        class_weights = class_weights
                        )
    elif model_configuration == 'VAE_original':
        model = models_tf.VAE_original(number_of_input_channels = 1,
                        model_name='VAE_original',
                        num_classes = len(unique_labels),
                        data_augmentation=data_augmentation,
                        class_weights = class_weights,
                        kernel_size=(5,5),
                        input_size=input_size,
                        vae_latent_dim=vae_latent_dim
                        )
    elif model_configuration == 'VAE1':
        model = models_tf.VAE1(number_of_input_channels = 1,
                        model_name='VAE1',
                        num_classes = len(unique_labels),
                        data_augmentation=data_augmentation,
                        class_weights = class_weights,
                        kernel_size=(5,5),
                        input_size=input_size
                        )
    else:
        model = models_tf.LightOCT(number_of_input_channels = 1,
                        model_name='LightOCT',
                        num_classes = len(unique_labels),
                        data_augmentation=data_augmentation,
                        class_weights = class_weights,
                        kernel_size=(5,5),
                        input_size=input_size
                        )

    # train model
    print(' - Training fold...')
    utilities_models_tf.outsideTrain(model,
                    train_dataset, val_dataset,
                    unique_labels = unique_labels,
                    loss=[loss],
                    start_learning_rate = 0.001,
                    scheduler = 'polynomial',
                    power = 0.3,
                    vae_kl_weight=vae_kl_weight,
                    vae_reconst_weight=vae_reconst_weight,
                    max_epochs=200,
                    early_stopping=True,
                    patience=20,
                    save_model_path=os.path.join(save_model_path, 'fold_'+str(cv+1)),
                    save_model_architecture_figure=True if cv==0 else False,
                    verbose=verbose
                    )

    # test model
    print(' - Testing fold...')
    test_dataset = utilities.TFR_2D_dataset(test_filenames,
                    dataset_type = 'test',
                    batch_size=batch_size,
                    buffer_size=1000,
                    crop_size=input_size)

    test_gt, test_prediction, test_time = utilities_models_tf.test(model, test_dataset)
    test_fold_summary[cv]={
            'ground_truth':np.argmax(test_gt.numpy(), axis=-1),
            'prediction':test_prediction.numpy(),
            'test_time':float(test_time)
            }

    # delete model to free space
    del model
    del train_dataset
    del val_dataset

## CROSS_VALIDATION TESTING
from collections import OrderedDict
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
'''
Saving overall cross validation test results and images:
- Confisuion matrix
- ROC curve
- Precision-Recall curve

- test summary file with the prediction for every test image (test_summary.txt)
    Here add also the inforlation needed to re-plot the ROC and PP curves (fpr,
    tpr, roc_auc, precision and recall - micro and macro average)
    The test_summary.txt file is a dictionary with the following entries:
    - model_name: string
    - labels: list of the true values fot the tested images
    - fold_test_values: list containing the predictions for every fold (list of lists)

    - test_time: string
    - test date: string

    - accuracy: float

    - false_positive_rate: list containing the fpr for every class (list of lists)
    - false_positive_rate_micro_avg: list containing the micro average fpr (used for the overall roc plot)
    - false_positive_rate_macro_avg: list containing the macro average fpr (used for the overall roc plot)

    - true_positive_rate: list containing the tpr for every class (list of lists)
    - true_positive_rate_micro_avg: list containing the micro average tpr (used for the overall roc plot)
    - true_positive_rate_macro_avg: list containing the macro average tpr (used for the overall roc plot)

    - precision: list containing the precision values for every class to plot the PP (list of lists)
    - precision_micro_avg: list of overall micro average of precision
    - average_precision: average precision value computed using micro average

    - recall: list containing the recall value for every class to plot the PP (list of lists)
    - recall_micro_avg: list of overall micro average of recall

    - F1: list of micro and macro average f1-score

Since the full test_summary file is long to open, the scores are also saved in a separate file for easy access
scores_test_summary.txt
'''
# ############# save the information that is already available
test_summary = OrderedDict()

test_summary['model_name'] = model_save_name
test_summary['labels'] = [int(i) for i in test_fold_summary[0]['ground_truth']]
test_summary['folds_test_logits_values'] = [test_fold_summary[cv]['prediction'].tolist() for cv in range(N_FOLDS)]
test_summary['test_time'] = utilities.tictoc_from_time(np.sum([test_fold_summary[cv]['test_time'] for cv in range(N_FOLDS)]))
test_summary['test_date'] = time.strftime("%Y%m%d-%H%M%S")

# ############ plot and save confucion matrix
ensemble_pred_argmax = []
ensemble_pred_logits = []
# compute ensemble
# compute the logits mean along the folds
ensemble_pred_logits = np.array(test_summary['folds_test_logits_values']).mean(axis=0)
# compute argmax prediction
ensemble_pred_argmax = np.argmax(ensemble_pred_logits, axis=1)

acc = utilities.plotConfusionMatrix(test_summary['labels'], ensemble_pred_argmax, classes=class_labels, savePath=save_model_path, draw=False)

# ############ plot and save ROC curve
fpr, tpr, roc_auc = utilities.plotROC(test_summary['labels'], ensemble_pred_logits, classes=class_labels, savePath=save_model_path, draw=False)
# make elements of the dictionary to be lists for saving
for key, value in fpr.items():
    fpr[key]=value.tolist()
for key, value in tpr.items():
    tpr[key]=value.tolist()
for key, value in roc_auc.items():
    roc_auc[key]=value.tolist()

# ############ plot and save ROC curve
precision, recall, average_precision, F1 = utilities.plotPR(test_summary['labels'], ensemble_pred_logits, classes=class_labels, savePath=save_model_path, draw=False)
# make elements of the dictionary to be lists for saving
for key, value in precision.items():
    precision[key]=value.tolist()
for key, value in recall.items():
    recall[key]=value.tolist()

# save all the information in the test summary
test_summary['accuracy'] = acc

# test_summary['false_positive_rate'] = [fpr[i].tolist() for i in range(len(class_labels))]
test_summary['false_positive_rate'] = fpr
# test_summary['false_positive_rate_micro_avg'] = fpr['micro'].tolist()
# test_summary['false_positive_rate_macro_avg'] = fpr['macro'].tolist()

test_summary['true_positive_rate'] = tpr
# test_summary['true_positive_rate'] = [tpr[i].tolist() for i in range(len(class_labels))]
# test_summary['true_positive_rate_micro_avg'] = tpr['micro'].tolist()
# test_summary['true_positive_rate_macro_avg'] = tpr['macro'].tolist()

test_summary['precision'] = precision
# test_summary['precision'] = [precision[i].tolist() for i in range(len(class_labels))]
# test_summary['precision_micro_avg'] = precision['micro'].tolist()

test_summary['recall'] = recall
# test_summary['recall'] = [recall[i].tolist() for i in range(len(class_labels))]
# test_summary['recall_micro_avg'] = recall['micro'].tolist()

test_summary['average_precision'] = average_precision
test_summary['F1'] = F1

# save summary file
with open(os.path.join(save_model_path,'test_summary.txt'), 'w') as fp:
    json.dump(test_summary, fp)

# save score summary

score_test_summary = OrderedDict()

score_test_summary['accuracy'] = acc
score_test_summary['average_precision'] = average_precision
score_test_summary['F1'] = F1

with open(os.path.join(save_model_path,'score_test_summary.txt'), 'w') as fp:
    json.dump(score_test_summary, fp)




