'''
Script that given path to the model, the dataset and the configuration file,
it resumes model training using the information saved in the model_summary_json.txt
file.

STEPS
1 - open configuration file and gets the training configurations
2 - creates data generator based on the configuration file
3 - creates a folder where the new model will be saved (same name as the original,
    but with prefix resumed)
4 - loads the model
5 - resumes training saving the new model in the new folder
'''

import os
import sys
import json
import glob
import types
import time
import pathlib
import random
import shutil
import argparse
import importlib
import numpy as np
import nibabel as nib
from random import shuffle
from datetime import datetime
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras.layers as layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# custom imports
import models_tf
import utilities
import utilities_models_tf

## get parameters

parser = argparse.ArgumentParser(description='Script that resumes the training of the specified model.')
parser.add_argument('-d','--dataset_path' ,required=True, help='Provide the path to dataset containing the files specified in the configuration file and used for the initial training.')
parser.add_argument('-m','--model_path' ,required=True, help='Provide the path to model that one wants to resume training. The expected path is the one where each model fold is located')
parser.add_argument('-f','--fold' ,required=True, help='Which model of the one available in the different folds to use.')
parser.add_argument('-cf','--configuration_file' ,required=False, help='Provide the path to the configuration file generated using the configure_training.py script.', default=None)
parser.add_argument('-r','--overwrite' ,required=True, help='Specify if to overwrite the original model or to save the resumed training model in a separate folder. If False, the resumed model will be saved separately.')
parser.add_argument('-mv','--model_version' ,required=False, help='Specify which model version to resume training of, the best (best) model or the last (last) model. Default is best', default="best"
)
# parser.add_argument('-db','--debug' ,required=False, help='Set to True if one wants to run the training in debug mode (only 5 epochs).', default=False)
parser.add_argument('-e','--epocs' ,required=False, help='Set the maximum number of epochs used to train the model Default 200.', default=200)
parser.add_argument('-p','--patience' ,required=False, help='Set the patiencs for early stopping. Default 25', default=25)
args = parser.parse_args()

configuration_file = args.configuration_file
model_path = args.model_path
dataset_path = args.dataset_path
model_version = args.model_version
fold = args.fold
overwrite = args.overwrite == "True"
debug = args.debug == "True"
max_epochs = int(args.epocs)
patience = int(args.patience)

# # # # # # # # # # # # # # # parse variables DEBUG
# model_path = '/flush/iulta54/Research/P3-OCT_THR/trained_models/test_resume_training'
# model_version = 'best'
# dataset_path = "/flush/iulta54/Research/Data/OCT/Thyroid_2019_DL"
# fold = "1"
# overwrite = False
# debug = True
# max_epochs = 15
# patience = 5
# configuration_file = None


# check if configuration file is provided, if not use the one of in the model_path
if configuration_file == "None":
    configuration_file = os.path.join(model_path, "config.json")

# check if configuration file exists
if not os.path.isfile(configuration_file):
    raise ValueError(f'Configuration file not found. Run the configure_training.py script first. Given {configuration_file}')

# check if model is present
if model_version == "best":
    model_name_configuration = "model.tf"
elif model_version == "last":
    model_name_configuration = "last_model.tf"
else:
    raise ValueError(f'Model version not recognized. Expected best or last, recieved {model_version}')

model_file_path = os.path.join(model_path, f'fold_{fold}', model_name_configuration)

if not os.path.isdir(model_file_path):
    raise ValueError(f'Model file not found. Given {model_file_path}')
else:
    # check that the model_summary_json file is present
    model_summary_file = os.path.join(model_path, f'fold_{fold}', "model_summary_json.txt")
    if not os.path.isfile(model_summary_file):
        raise ValueError(f'Model summary file not found. Given {model_summary_file}')

# check dataset folder
if not os.path.isdir(dataset_path):
    raise ValueError(f'Dataset path not found. Given {dataset_path}')

# load both configuration file and model summary file to print extra information
with open(configuration_file) as json_file:
    config = json.load(json_file)

with open(model_summary_file) as json_file:
    previous_training_summary = json.load(json_file)

# set max_epoch and patienc to default
if max_epochs is None:
    max_epochs = previous_training_summary["MAX_EPOCHS"] if "MAX_EPOCHS" in previous_training_summary.keys() else 200

# take care of model overwrite or not
if not overwrite:
    # create path to for the new model and copy model and configuration file and
    # update the different paths
    save_path = os.path.join(os.path.dirname(model_path),f'resumed_{os.path.basename(model_path)}',f'fold_{fold}')
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    # copy model
    try:
        shutil.copytree(os.path.join(model_path, f'fold_{fold}' ,model_name_configuration), os.path.join(save_path, model_name_configuration))
    except:
        shutil.rmtree(os.path.join(save_path, model_name_configuration))
        shutil.copytree(os.path.join(model_path, f'fold_{fold}' ,model_name_configuration), os.path.join(save_path, model_name_configuration))
    if model_version == "best":
        aus = "model_weights.tf.index"
        shutil.copy(os.path.join(model_path, f'fold_{fold}',"model_weights.tf.index"), save_path)
    elif model_version == "last":
        aus = "last_model_weights.tf.index"
    try:
        shutil.copy(os.path.join(model_path, f'fold_{fold}',aus), save_path)
    except:
        os.remove(os.path.join(save_path, aus))
        shutil.copy(os.path.join(model_path, f'fold_{fold}',aus), save_path)

    # copy configuration file
    try:
        shutil.copy(configuration_file, os.path.dirname(save_path))
    except:
        os.remove(os.path.join(os.path.dirname(save_path), "config.json"))
        shutil.copy(configuration_file, os.path.dirname(save_path))

else:
    save_path = model_path

# print information
strings = {"Model to resume training of" : model_file_path,
           "Configuration file" : configuration_file,
           "Previous model trained for" : f'{len(previous_training_summary["TRAIN_LOSS_HISTORY"])} epochs.',
           "Resuming training for " : max_epochs,
           "Dataset path" : dataset_path,
           "Training on" : f'{len(config["training"][int(fold)-1])} images.',
           "Validating on" : f'{len(config["validation"][int(fold)-1])} images.',
           "Testing on" : f'{len(config["test"])} images.',
           "Overwriting existing model" : overwrite,
           "Resumed training model will be saved at" : save_path
            }

max_len = np.max([len(key) for key in strings.keys()])

for key, value in strings.items():
    print(f'{key:{max_len}s} : {value}')

## open model
print(f'\n   - Loading model...')
tf_model = tf.keras.models.load_model(model_file_path, compile=False)

# wrap model into custom model wrapper

class ModelWrapper(object):
    '''
    This object wraps a tf model to resemble the model object used for normal training
    '''
    def __init__(self, tf_model, configuration_file, summary_previous_training, debug=False):

        # initialize ModelWraper attributes
        self.model = tf_model
        self.number_of_input_channels = tf_model.inputs[0].shape[-1]
        self.num_classes = tf_model.output.shape[-1]
        self.input_size=configuration_file['input_size']
        self.debug = debug
        if configuration_file['class_weights'] is None:
            self.class_weights = np.ones([1, self.num_classes])
        else:
            self.class_weights = configuration_file['class_weights']
        self.model_name = summary_previous_training['Model_name']
        self.kernel_size = summary_previous_training['Kernel_size']


        # save model paramenters
        self.num_filter_start = summary_previous_training['Num_filter_start']
        self.depth = summary_previous_training['Model_depth']
        self.custom_model = summary_previous_training['Custom_model']

        # print model if needed
        if self.debug is True:
            print(self.model.summary())

##
model = ModelWrapper(tf_model, config, previous_training_summary)

## create data generator
print('   - Creating datasets...')
# get the file names for training and validation
X_train = config['training'][int(fold)-1]
X_val = config['validation'][int(fold)-1]
X_test = config['test']

# make sure that the files point to this system dataset
X_train = [os.path.join(dataset_path, pathlib.Path(f).parts[-2], pathlib.Path(f).parts[-1]) for f in X_train]
X_val = [os.path.join(dataset_path, pathlib.Path(f).parts[-2], pathlib.Path(f).parts[-1]) for f in X_val]
X_test = [os.path.join(dataset_path, pathlib.Path(f).parts[-2], pathlib.Path(f).parts[-1]) for f in X_test]

for f in X_train:
    if not os.path.isfile(f):
        raise ValueError(f'{f} not found')

# create datasets
train_dataset = utilities.TFR_2D_dataset(X_train,
                dataset_type = 'train',
                batch_size=config['batch_size'],
                buffer_size=5000,
                crop_size=config['input_size'])

val_dataset = utilities.TFR_2D_dataset(X_val,
                dataset_type = 'test',
                batch_size=config['batch_size'],
                buffer_size=1000,
                crop_size=config['input_size'])

## train model
print('   - Training fold...')

warm_up = False,
warm_up_epochs = 5
warm_up_learning_rate = 0.00001

if 'VAE' in config['model_configuration']:
    utilities_models_tf.train_VAE(model,
                    train_dataset, val_dataset,
                    classification_type =config['classification_type'],
                    unique_labels = config['unique_labels'],
                    loss=[config['loss']],
                    start_learning_rate = config['learning_rate'],
                    scheduler = 'constant',
                    vae_kl_weight=config['vae_kl_weight'],
                    vae_reconst_weight=config['vae_reconst_weight'],
                    power = 0.1,
                    max_epochs=max_epochs,
                    early_stopping=True,
                    patience=patience,
                    warm_up = warm_up,
                    warm_up_epochs = warm_up_epochs,
                    warm_up_learning_rate = warm_up_learning_rate,
                    save_model_path=save_path,
                    save_model_architecture_figure=True,
                    verbose=config['verbose']
                    )
else:
    utilities_models_tf.train(model,
                    train_dataset, val_dataset,
                    classification_type =config['classification_type'],
                    unique_labels = config['unique_labels'],
                    loss=[config['loss']],
                    start_learning_rate = config['learning_rate'],
                    scheduler = 'constant',
                    power = 0.1,
                    max_epochs=max_epochs,
                    early_stopping=True,
                    patience=patience,
                    save_model_path=save_path,
                    save_model_architecture_figure=True,
                    warm_up = warm_up,
                    warm_up_epochs = warm_up_epochs,
                    warm_up_learning_rate = warm_up_learning_rate,
                    verbose=config['verbose']
                    )















