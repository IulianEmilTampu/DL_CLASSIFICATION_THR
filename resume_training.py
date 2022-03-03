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
parser.add_argument('-f','--fold', nargs='+', required=False, help='Which model of the one available in the different folds to use. If None, checking among the different folds to see which model stil needs training. A list of the folds can also be provided', default=None)
parser.add_argument('-cf','--configuration_file' ,required=False, help='Provide the path to the configuration file generated using the configure_training.py script.', default=None)
parser.add_argument('-r','--overwrite' ,required=True, help='Specify if to overwrite the original model or to save the resumed training model in a separate folder. If False, the resumed model will be saved separately.')
parser.add_argument('-mv','--model_version' , nargs='+', required=False, help='Specify which model version to resume training of, the best (best) model or the last (last) model. Default is best. If one is given, the same model configuration is used for all the models. If multiple are given, these should match the number of specified folds.', default="best"
)
parser.add_argument('-db','--debug' ,required=False, help='Set to True if one wants to run the training in debug mode (only 5 epochs).', default=False)
parser.add_argument('-e','--epocs' ,required=False, help='Set the maximum number of epochs used to train the model Default 200.', default=200)
parser.add_argument('-p','--patience' ,required=False, help='Set the patiencs for early stopping. Default 25', default=25)
args = parser.parse_args()

configuration_file = args.configuration_file
model_path = args.model_path
dataset_path = args.dataset_path
model_version = [i for i in args.model_version]
if args.fold[0] == 'None':
    fold = None
else:
    fold = [int(i) for i in args.fold]
overwrite = args.overwrite == "True"
max_epochs = int(args.epocs)
patience = int(args.patience)
debug = args.debug == "True"

# # # # # # # # # # # # # parse variables DEBUG
# model_path = '/flush/iulta54/Research/P3-OCT_THR/trained_models/M6_fold5_c13_BatchNorm_dr0.3_lr0.000001_wcce_weights_batch64'
# model_version = ['last']
# dataset_path = "/flush/iulta54/Research/Data/OCT/Thyroid_2019_DL"
# fold = None
# overwrite = False
# debug = False
# max_epochs = 150
# patience = 150
# configuration_file = "None"
# model_3d = False


# check if configuration file is provided, if not use the one of in the model_path
if configuration_file == "None":
    configuration_file = os.path.join(model_path, "config.json")

# check if configuration file exists
if not os.path.isfile(configuration_file):
    raise ValueError(f'Configuration file not found. Run the configure_training.py script first. Given {configuration_file}')

# check if model versions is present
if not all([any([v=="best", v=="last"]) for v in model_version]):
    raise ValueError(f'Model version not recognized. Expected best or last, recieved {model_version}')

## HEURISTIC FOR CHECKING THE FOLDS AND GET INFORMATION ABOUT WHAT TO TRAIN
fold_dict = []
if fold is None:
    # start heuristic to check which folds need training
    # Models that were interupted do not have the last model saved, only the best
    folds_to_train = [os.path.basename(f) for f in glob.glob(os.path.join(model_path,'fold_*')) if not os.path.isdir(os.path.join(f,'last_model.tf'))]

    print(f'No fold was specified for resuming training.')
    print(f'Found {len(folds_to_train)} with incomplete training.')
    print(f'Restarting model training (using best model) for folds {folds_to_train}')

    # check that the best model is present
    check_best_models = [os.path.isdir(os.path.join(model_path,f,'model.tf')) for f in folds_to_train]
    if not all(check_best_models):
        print(f'Best model not found for folds {[folds_to_train[idx] for idx, i in check_best_models if not i]}')
    else:
        print('All selected folds have the best model saved')

    # build fold dictionary used to run model re-training
    for f, fc in zip(folds_to_train, check_best_models):
        aus = {}
        aus['fold'] = f
        aus['fold_index'] = int(f[f.find('_')+1::])-1
        aus['model_version'] = 'best'
        aus['model_version_name'] = 'model.tf'
        aus['build_model'] = not fc

        if aus['build_model'] == True:
            # here we need to initialize a new model
            aus['model_summary_file'] = None
            aus['trained_epochs'] = 0
            aus['model_path'] = None
        else:
            aus['model_path'] = os.path.join(model_path,f,'model.tf')
            aus['model_summary_file'] = os.path.join(model_path,f,"model_summary_json.txt")
            # load previous model training summary and infere number of trained epochs
            with open(aus['model_summary_file']) as json_file:
                previous_training_summary = json.load(json_file)
            aus['trained_epochs'] = len(previous_training_summary['TRAIN_ACC_HISTORY']) if "TRAIN_ACC_HISTORY" in previous_training_summary.keys() else 0

        # save info
        fold_dict.append(aus)
else:
    folds_to_train = [f'fold_{f}' for f in fold]
    print(f'{len(fold)} folds specified for resuming training ({fold})')
    # check that model version specified for the folds are available
    if len(model_version) != len(folds_to_train):
        model_version = [model_version[0] for i in  range(len(folds_to_train))]


    for f, mv in zip(folds_to_train, model_version):
        mv_name = 'model.tf' if mv == 'best' else 'last_model.tf'
        aus_model_path = os.path.join(model_path,f,mv_name)
        if not os.path.isdir(aus_model_path):
            print(f'Model file not found. Given {aus_model_path}. Reinitializing model')
            # build fold dictionary used to run model re-training
            aus = {}
            aus['fold'] = f
            aus['fold_index'] = int(f[f.find('_')+1::])-1
            aus['model_version'] = mv
            aus['model_version_name'] = mv_name
            aus['build_model'] = True
            aus['model_path'] = aus_model_path
            aus['model_summary_file'] = None
            aus['trained_epochs'] = 0

            # save info
            fold_dict.append(aus)

        else:
            # check that the model_summary_json file is present
            model_summary_file = os.path.join(model_path, f, "model_summary_json.txt")
            if not os.path.isfile(model_summary_file):
                raise ValueError(f'Model summary file not found. Given {model_summary_file}')
            else:
                # build fold dictionary used to run model re-training
                aus = {}
                aus['fold'] = f
                aus['fold_index'] = int(f[f.find('_')+1::])-1
                aus['model_version'] = mv
                aus['model_version_name'] = mv_name
                aus['build_model'] = False
                aus['model_path'] = aus_model_path
                aus['model_summary_file'] = model_summary_file
                # load previous model training summary and infere number of trained epochs
                with open(aus['model_summary_file']) as json_file:
                    previous_training_summary = json.load(json_file)
                aus['trained_epochs'] = len(previous_training_summary['TRAIN_ACC_HISTORY']) if "TRAIN_ACC_HISTORY" in previous_training_summary.keys() else 0

                # save info
                fold_dict.append(aus)

# for f in fold_dict:
#     print(f'Fold {f["fold"]}, version {f["model_version"]}, trained epochs {f["trained_epochs"]}')

## SET UP WHERE TO SAVE THE RETRAINED MODELS
for m in fold_dict:
    # take care of model overwrite or not
    if not overwrite:
        if not m['build_model']:
            # the model exists but training needs to restart
            # create path to for the new model and copy model and configuration file and
            # update the different paths
            save_path = os.path.join(os.path.dirname(model_path),f'resumed_{os.path.basename(model_path)}',m["fold"])
            pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
            # copy model
            try:
                shutil.copytree(m["model_path"], os.path.join(save_path, m["model_version_name"]))
            except:
                shutil.rmtree(os.path.join(save_path, m["model_version_name"]))
                shutil.copytree(m["model_path"], os.path.join(save_path, m["model_version_name"]))
            if m["model_version"] == "best":
                aus = "model_weights.tf.index"
                shutil.copy(os.path.join(model_path, m["fold"],"model_weights.tf.index"), save_path)
            elif  m["model_version"] == "last":
                aus = "last_model_weights.tf.index"
            try:
                shutil.copy(os.path.join(model_path, m["fold"], aus), save_path)
            except:
                os.remove(os.path.join(save_path, aus))
                shutil.copy(os.path.join(model_path, m["fold"], aus), save_path)

            # copy configuration file
            try:
                shutil.copy(configuration_file, os.path.dirname(save_path))
                # update configuration file path
                new_configuration_file = os.path.join(os.path.dirname(save_path),"config.json")
            except:
                os.remove(os.path.join(os.path.dirname(save_path), "config.json"))
                shutil.copy(configuration_file, os.path.dirname(save_path))
                # update configuration file path
                new_configuration_file = os.path.join(os.path.dirname(save_path),"config.json")
        else:
            # model will be initialize so no need to copy, but update save_path and copy the configuration file
            save_path = os.path.join(os.path.dirname(model_path),f'resumed_{os.path.basename(model_path)}',m["fold"])
            pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

            # copy configuration file
            try:
                shutil.copy(configuration_file, os.path.dirname(save_path))
                # update configuration file path
                new_configuration_file = os.path.join(os.path.dirname(save_path),"config.json")
            except:
                os.remove(os.path.join(os.path.dirname(save_path), "config.json"))
                shutil.copy(configuration_file, os.path.dirname(save_path))
                # update configuration file path
                new_configuration_file = os.path.join(os.path.dirname(save_path),"config.json")
    else:
        save_path = model_path


    # open configuration file
    with open(new_configuration_file) as json_file:
        config = json.load(json_file)

    # print information
    strings = {"Model to resume training of" : m['model_path'],
            "Configuration file" : configuration_file,
            "Previous model trained for" : f'{m["trained_epochs"]} epochs.',
            "Resuming training for " : max_epochs,
            "Dataset path" : dataset_path,
            "Training on" : f'{len(config["training"][m["fold_index"]])} images.',
            "Validating on" : f'{len(config["validation"][m["fold_index"]])} images.',
            "Testing on" : f'{len(config["test"])} images.',
            "Overwriting existing model" : overwrite,
            "Resumed training model will be saved at" : save_path
                }

    max_len = np.max([len(key) for key in strings.keys()])

    if debug is True:
        string = "Running resume training routine in debug mode (using lower number of epochs (4) and 10% of the dataset)"
        l = len(string)
        print(f'\n{"-"*l}')
        print(f'{string:^{l}}')
        print(f'{"-"*l}\n')

        # reducing the number of training epochs
        max_epochs = 4
        patience = 4
    else:
        string = "Running resume training routine"
        l = len(string)
        print(f'{"-"*l}')
        print(f'{string:^20}')
        print(f'{"-"*l}\n')

    for key, value in strings.items():
        print(f'{key:{max_len}s} : {value}')

    ## open model

    if m["build_model"]:
        print(f'\n- Building model...')
        # construct model from scratch
        if config['model_configuration'] == 'LightOCT':
            model = models_tf.LightOCT(number_of_input_channels = config['num_channels'] if "num_channels" in config.keys() else 1,
                            model_name=config['model_configuration'],
                            num_classes = len(config['unique_labels']),
                            data_augmentation=config['data_augmentation'],
                            class_weights = config['class_weights'],
                            kernel_size=config['kernel_size'],
                            input_size=config['input_size'],
                            )
        elif config['model_configuration'] == 'M2':
            model = models_tf.M2(number_of_input_channels = 1,
                            model_name='M2',
                            num_classes = len(config['unique_labels']),
                            input_size=config['input_size'],
                            data_augmentation=config['data_augmentation'],
                            class_weights = config['class_weights'],
                            kernel_size=config['kernel_size'],
                            )
        elif config['model_configuration'] == 'M3':
            model = models_tf.M3(number_of_input_channels = 1,
                            model_name=config['model_configuration'],
                            num_classes = len(config['unique_labels']),
                            data_augmentation=config['data_augmentation'],
                            class_weights = config['class_weights'],
                            kernel_size=config['kernel_size'],
                            )
        elif config['model_configuration'] == 'M4':
            model = models_tf.M4(number_of_input_channels = config['num_channels'] if "num_channels" in config.keys() else 1,
                            model_name=config['model_configuration'],
                            normalization = config['model_normalization'],
                            dropout_rate = config['dropout_rate'],
                            num_classes = len(config['unique_labels']),
                            data_augmentation=config['data_augmentation'],
                            class_weights = config['class_weights'],
                            kernel_size=config['kernel_size'],
                            )
        elif config['model_configuration'] == 'M5':
            model = models_tf.M5(number_of_input_channels = config['num_channels'] if "num_channels" in config.keys() else 1,
                            model_name=config['model_configuration'],
                            normalization = config['model_normalization'],
                            dropout_rate = config['dropout_rate'],
                            num_classes = len(config['unique_labels']),
                            data_augmentation=config['data_augmentation'],
                            class_weights = config['class_weights'],
                            kernel_size=config['kernel_size'],
                            )
        elif config['model_configuration'] == 'M6':
            model = models_tf.M6(number_of_input_channels = 1,
                            input_size=config['input_size'],
                            model_name=config['model_configuration'],
                            num_classes = len(config['unique_labels']),
                            data_augmentation=config['data_augmentation'],
                            class_weights = config['class_weights'],
                            kernel_size=config['kernel_size']
                            )
        elif config['model_configuration'] == 'ResNet50':
            model = models_tf.ResNet50(number_of_input_channels = 1,
                            model_name=config['model_configuration'],
                            num_classes = len(config['unique_labels']),
                            data_augmentation=config['data_augmentation'],
                            class_weights = config['class_weights'],
                            )
        elif config['model_configuration'] == 'EfficientNet_B7':
            model = models_tf.EfficientNet_B7(number_of_input_channels = 1,
                            model_name=config['model_configuration'],
                            num_classes = len(config['unique_labels']),
                            data_augmentation=config['data_augmentation'],
                            class_weights = config['class_weights'],
                            )
        elif config['model_configuration'] == 'InceptionV3':
            model = models_tf.InceptionV3(number_of_input_channels = 1,
                            model_name=config['model_configuration'],
                            num_classes = len(config['unique_labels']),
                            data_augmentation=config['data_augmentation'],
                            class_weights = config['class_weights'],
                            input_size=config['input_size'],
                            )
        elif config['model_configuration'] == 'VAE_DEBUG':
            model = models_tf.VAE_DEBUG(number_of_input_channels = 1,
                            model_name=config['model_configuration'],
                            num_classes = len(config['unique_labels']),
                            data_augmentation=config['data_augmentation'],
                            class_weights = config['class_weights'],
                            kernel_size=config['kernel_size'],
                            input_size=config['input_size']
                            )
        elif config['model_configuration'] == 'ViT':
            model = models_tf.ViT(number_of_input_channels = 1,
                            model_name=config['model_configuration'],
                            num_classes = len(config['unique_labels']),
                            data_augmentation=config['data_augmentation'],
                            class_weights = config['class_weights'],
                            input_size=config['input_size'],
                            patch_size=config["vit_patch_size"],
                            projection_dim=config["vit_projection_dim"],
                            num_heads=config["vit_num_heads"],
                            mlp_head_units=config["vit_mlp_head_units"],
                            transformer_layers=config["vit_transformer_layers"],
                            transformer_units=config["vit_transformer_units"])
        else:
            raise ValueError('Specified model configuration not available. Provide one that is implemented in models_tf.py')
            sys.exit()

    else:
        print(f'\n- Loading model...')
        # load model
        tf_model = tf.keras.models.load_model(m["model_path"], compile=False)

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

        # load model summary
        with open(m['model_summary_file']) as json_file:
            previous_training_summary = json.load(json_file)

        model = ModelWrapper(tf_model, config, previous_training_summary)

    # check if model is 3D
    model_3D = True if len(config["input_size"])>2 else False

## create data generator
    print('- Creating datasets...')
    # get the file names for training and validation
    X_train = config['training'][m["fold_index"]]
    X_val = config['validation'][m["fold_index"]]
    X_test = config['test']

    # make sure that the files point to this system dataset
    X_train = [os.path.join(dataset_path, pathlib.Path(f).parts[-2], pathlib.Path(f).parts[-1]) for f in X_train]
    X_val = [os.path.join(dataset_path, pathlib.Path(f).parts[-2], pathlib.Path(f).parts[-1]) for f in X_val]
    X_test = [os.path.join(dataset_path, pathlib.Path(f).parts[-2], pathlib.Path(f).parts[-1]) for f in X_test]

    if debug:
        X_train = random.sample(X_train, int(len(X_train)*0.1))

    for f in X_train:
        if not os.path.isfile(f):
            raise ValueError(f'{f} not found')

    # create datasets
    if model_3D:
        train_dataset = utilities.TFR_3D_sparse_dataset(X_train,
                        dataset_type = 'train',
                        batch_size=1,
                        buffer_size=500,
                        crop_size=config['input_size'])

        val_dataset = utilities.TFR_3D_sparse_dataset(X_val,
                        dataset_type = 'test',
                        batch_size=1,
                        buffer_size=500,
                        crop_size=config['input_size'])
    else:
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
    print('- Training fold...')

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
                        max_epochs=max_epochs - m["trained_epochs"],
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
                        classification_type = config['classification_type'],
                        unique_labels = config['unique_labels'],
                        loss=[config['loss']],
                        start_learning_rate = config['learning_rate'],
                        scheduler = 'constant',
                        power = 0.1,
                        max_epochs=max_epochs - m["trained_epochs"],
                        early_stopping=True,
                        patience=patience,
                        save_model_path=save_path,
                        save_model_architecture_figure=True,
                        warm_up = warm_up,
                        warm_up_epochs = warm_up_epochs,
                        warm_up_learning_rate = warm_up_learning_rate,
                        verbose=config['verbose']
                        )















