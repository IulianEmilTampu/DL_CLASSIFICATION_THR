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
import tensorflow.keras.layers as layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# custom imports
import models_tf
import utilities
import utilities_models_tf

## parse the configuration file

parser = argparse.ArgumentParser(description='Script that runs a cross-validation training for OCT 2D image classification. It uses the configuration file created using the configure_training.py file. Run the configuration first!')

parser.add_argument('-cf','--configuration_file' ,required=True, help='Provide the path to the configuration file generated using the configure_training.py script.')
parser.add_argument('-db','--debug' ,required=False, help='Set to True if one wants to run the training in debug mode (only 15 epochs with 10 early stop patience).', default=False)
args = parser.parse_args()

configuration_file = args.configuration_file
debug = bool(args.debug)

if not os.path.isfile(configuration_file):
    raise ValueError(f'Configuration file not found. Run the configure_training.py script first. Given {configuration_file}')

if debug is True:
    print(f'\n{"-"*83}')
    print(f'{"Running training routine in debug mode (using less data and lower number of epochs)":^20}')
    print(f'{"-"*83}\n')
else:
    print(f'{"-"*20}')
    print(f'{"Running training routine":^20}')
    print(f'{"-"*20}\n')

with open(configuration_file) as json_file:
    config = json.load(json_file)

## create folders where to save the data and models for each fold

for cv in range(config['N_FOLDS']):
    if not os.path.isdir(os.path.join(config['save_model_path'], 'fold_'+str(cv+1))):
        os.mkdir(os.path.join(config['save_model_path'], 'fold_'+str(cv+1)))

## initialise variables where to save test summary

test_fold_summary = {}

## loop through the folds
importlib.reload(utilities_models_tf)

# ############################ TRAINING
for cv in range(config['N_FOLDS']):
    print('Working on fold {}/{}. Start time {}'.format(cv+1, config['N_FOLDS'], datetime.now().strftime("%H:%M:%S")))

    print(' - Creating datasets...')
    # get the file names for training and validation
    X_train = config['training'][cv]
    X_val = config['validation'][cv]

    # create datasets
    train_dataset = utilities.TFR_2D_dataset(X_train,
                    dataset_type = 'train',
                    batch_size=config['batch_size'],
                    buffer_size=5000,
                    crop_size=config['input_size'])

    # set normalization layer on the training dataset
    tr_feature_ds = train_dataset.map(lambda x, y: x)
    normalizer = layers.experimental.preprocessing.Normalization()
    normalizer.adapt(tr_feature_ds)

    val_dataset = utilities.TFR_2D_dataset(X_val,
                    dataset_type = 'train',
                    batch_size=config['batch_size'],
                    buffer_size=1000,
                    crop_size=config['input_size'])

    # create model based on specification
    if config['model_configuration'] == 'LightOCT':
        model = models_tf.LightOCT(number_of_input_channels = 1,
                        model_name=config['model_configuration'],
                        num_classes = len(config['unique_labels']),
                        data_augmentation=config['data_augmentation'],
                        class_weights = config['class_weights'],
                        kernel_size=config['kernel_size'],
                        input_size=config['input_size'],
                        norm_layer=normalizer
                        )
    elif config['model_configuration'] == 'M2':
        model = models_tf.M2(number_of_input_channels = 1,
                        model_name='M2',
                        num_classes = len(config['unique_labels']),
                        data_augmentation=config['data_augmentation'],
                        class_weights = config['class_weights'],
                        kernel_size=config['kernel_size'],
                        norm_layer=normalizer
                        )
    elif config['model_configuration'] == 'M3':
        model = models_tf.M3(number_of_input_channels = 1,
                        model_name=config['model_configuration'],
                        num_classes = len(config['unique_labels']),
                        data_augmentation=config['data_augmentation'],
                        class_weights = config['class_weights'],
                        kernel_size=config['kernel_size'],
                        norm_layer=normalizer
                        )
    elif config['model_configuration'] == 'ResNet50':
        model = models_tf.ResNet50(number_of_input_channels = 1,
                        model_name=config['model_configuration'],
                        num_classes = len(config['unique_labels']),
                        data_augmentation=config['data_augmentation'],
                        class_weights = config['class_weights'],
                        norm_layer=normalizer
                        )
    elif config['model_configuration'] == 'InceptionV3':
        model = models_tf.InceptionV3(number_of_input_channels = 1,
                        model_name=config['model_configuration'],
                        num_classes = len(config['unique_labels']),
                        data_augmentation=config['data_augmentation'],
                        class_weights = config['class_weights'],
                        input_size=config['input_size'],
                        norm_layer=normalizer
                        )
    elif config['model_configuration'] == 'VAE_original':
        model = models_tf.VAE_original(number_of_input_channels = 1,
                        model_name=config['model_configuration'],
                        num_classes = len(config['unique_labels']),
                        data_augmentation=config['data_augmentation'],
                        class_weights = config['class_weights'],
                        kernel_size=config['kernel_size'],
                        input_size=config['input_size'],
                        vae_latent_dim=config['vae_latent_dim'],
                        norm_layer=normalizer
                        )
    elif config['model_configuration'] == 'VAE1':
        model = models_tf.VAE1(number_of_input_channels = 1,
                        model_name=config['model_configuration'],
                        num_classes = len(config['unique_labels']),
                        data_augmentation=config['data_augmentation'],
                        class_weights = config['class_weights'],
                        kernel_size=config['kernel_size'],
                        input_size=config['input_size'],
                        norm_layer=normalizer
                        )
    elif config['model_configuration'] == 'VAE2':
        model = models_tf.VAE2(number_of_input_channels = 1,
                        model_name=config['model_configuration'],
                        num_classes = len(config['unique_labels']),
                        data_augmentation=config['data_augmentation'],
                        class_weights = config['class_weights'],
                        kernel_size=config['kernel_size'],
                        input_size=config['input_size'],
                        norm_layer=normalizer
                        )
    elif config['model_configuration'] == 'VAE3':
        model = models_tf.VAE3(number_of_input_channels = 1,
                        model_name=config['model_configuration'],
                        num_classes = len(config['unique_labels']),
                        data_augmentation=config['data_augmentation'],
                        class_weights = config['class_weights'],
                        kernel_size=config['kernel_size'],
                        input_size=config['input_size'],
                        norm_layer=normalizer
                        )
    else:
        model = models_tf.LightOCT(number_of_input_channels = 1,
                        model_name='LightOCT',
                        num_classes = len(config['unique_labels']),
                        data_augmentation=config['data_augmentation'],
                        class_weights = config['class_weights'],
                        kernel_size=config['kernel_size'],
                        input_size=config['input_size'],
                        norm_layer=normalizer
                        )

    # train model
    print(' - Training fold...')
    if debug is True:
        utilities_models_tf.train(model,
                        train_dataset, val_dataset,
                        classification_type =config['classification_type'],
                        unique_labels = config['unique_labels'],
                        loss=[config['loss']],
                        start_learning_rate = config['learning_rate'],
                        scheduler = 'polynomial',
                        power = 0.1,
                        vae_kl_weight=config['vae_kl_weight'],
                        vae_reconst_weight=config['vae_reconst_weight'],
                        max_epochs=20,
                        early_stopping=True,
                        patience=10,
                        save_model_path=os.path.join(config['save_model_path'], 'fold_'+str(cv+1)),
                        save_model_architecture_figure=True if cv==0 else False,
                        verbose=config['verbose']
                        )
    else:
        utilities_models_tf.train(model,
                        train_dataset, val_dataset,
                        classification_type =config['classification_type'],
                        unique_labels = config['unique_labels'],
                        loss=[config['loss']],
                        start_learning_rate = config['learning_rate'],
                        scheduler = 'polynomial',
                        power = 0.1,
                        vae_kl_weight=config['vae_kl_weight'],
                        vae_reconst_weight=config['vae_reconst_weight'],
                        max_epochs=200,
                        early_stopping=True,
                        patience=20,
                        save_model_path=os.path.join(config['save_model_path'], 'fold_'+str(cv+1)),
                        save_model_architecture_figure=True if cv==0 else False,
                        verbose=config['verbose']
                        )

    # test model
    print(' - Testing fold...')
    test_dataset = utilities.TFR_2D_dataset(config['test'],
                    dataset_type = 'test',
                    batch_size=config['batch_size'],
                    buffer_size=1000,
                    crop_size=config['input_size'])
##
    importlib.reload(utilities_models_tf)
    test_gt, test_prediction, test_time = utilities_models_tf.test(model, test_dataset)
    test_fold_summary[cv]={
            'ground_truth':np.argmax(test_gt.numpy(), axis=-1),
            'prediction':test_prediction.numpy(),
            'test_time':float(test_time)
            }

    # delete model to free space
    # del model
    # del train_dataset
    # del val_dataset

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
    Here add also the information needed to re-plot the ROC and PP curves (fpr,
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

test_summary['model_name'] = config['model_save_name']
test_summary['labels'] = [int(i) for i in test_fold_summary[0]['ground_truth']]
test_summary['folds_test_logits_values'] = [test_fold_summary[cv]['prediction'].tolist() for cv in range(config['N_FOLDS'])]
test_summary['test_time'] = utilities.tictoc_from_time(np.sum([test_fold_summary[cv]['test_time'] for cv in range(config['N_FOLDS'])]))
test_summary['test_date'] = time.strftime("%Y%m%d-%H%M%S")

# ############ plot and save confucion matrix
ensemble_pred_argmax = []
ensemble_pred_logits = []
# compute ensemble
# compute the logits mean along the folds
ensemble_pred_logits = np.array(test_summary['folds_test_logits_values']).mean(axis=0)
# compute argmax prediction
ensemble_pred_argmax = np.argmax(ensemble_pred_logits, axis=1)

acc = utilities.plotConfusionMatrix(test_summary['labels'], ensemble_pred_argmax, classes=config['label_description'], savePath=config['save_model_path'], draw=False)

# ############ plot and save ROC curve
fpr, tpr, roc_auc = utilities.plotROC(test_summary['labels'], ensemble_pred_logits, classes=config['label_description'], savePath=config['save_model_path'], draw=False)
# make elements of the dictionary to be lists for saving
for key, value in fpr.items():
    fpr[key]=value.tolist()
for key, value in tpr.items():
    tpr[key]=value.tolist()
for key, value in roc_auc.items():
    roc_auc[key]=value.tolist()

# ############ plot and save ROC curve
precision, recall, average_precision, F1 = utilities.plotPR(test_summary['labels'], ensemble_pred_logits, classes=config['label_description'], savePath=config['save_model_path'], draw=False)
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
with open(os.path.join(config['save_model_path'],'test_summary.txt'), 'w') as fp:
    json.dump(test_summary, fp)

# save score summary

score_test_summary = OrderedDict()

score_test_summary['accuracy'] = acc
score_test_summary['average_precision'] = average_precision
score_test_summary['F1'] = F1

with open(os.path.join(config['save_model_path'],'score_test_summary.txt'), 'w') as fp:
    json.dump(score_test_summary, fp)




