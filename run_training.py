"""
Script that runs the training of a deep leaning model for classification
of 2D OCT thyroid images using the configurations saved by the configure_training.py file.

Steps
- loads the configuration file
- create trainig/validation/test dataloader with on-the-fly augmentation
- load the CNN model and define loss function and training hyperparamters
- train the model
- save trained model along with training curves.
"""

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

import tensorflow as tf
import tensorflow.keras.layers as layers

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# custom imports
import models_tf
import utilities
import utilities_models_tf

## parse the configuration file

parser = argparse.ArgumentParser(
    description="Script that runs a cross-validation training for OCT 2D image classification. It uses the configuration file created using the configure_training.py file. Run the configuration first!"
)

parser.add_argument(
    "-cf",
    "--configuration_file",
    required=True,
    help="Provide the path to the configuration file generated using the configure_training.py script.",
)
parser.add_argument(
    "-db",
    "--debug",
    required=False,
    help="Set to True if one wants to run the training in debug mode (only 5 epochs).",
    default=False,
)
parser.add_argument(
    "-e",
    "--epocs",
    required=False,
    help="Set the maximum number of epochs used to train the model Default 200.",
    default=200,
)
parser.add_argument(
    "-p",
    "--patience",
    required=False,
    help="Set the patiencs for early stopping. Default 25",
    default=25,
)
args = parser.parse_args()
parser.add_argument(
    "-f",
    "--folds",
    nargs="+",
    required=False,
    help="Specify which folds to train.",
    default="None",
)
args = parser.parse_args()

configuration_file = args.configuration_file
debug = args.debug == "True"
max_epochs = int(args.epocs)
patience = int(args.patience)
folds = [int(i) for i in args.folds] if args.folds != "None" else None

# # # # # # # # # # # # # # # DEBUG
# configuration_file = '/flush/iulta54/Research/P3-OCT_THR/trained_models/test_M4_2D_prj/config.json'
# debug = False
# max_epochs = 300
# patience = 300
# folds = None

if not os.path.isfile(configuration_file):
    raise ValueError(
        f"Configuration file not found. Run the configure_training.py script first. Given {configuration_file}"
    )

if debug is True:
    string = "Running training routine in debug mode (using lower number of epochs and 20% of the dataset)"
    l = len(string)
    print(f'\n{"-"*l}')
    print(f"{string:^{l}}")
    print(f'{"-"*l}\n')

    # reducing the number of training epochs
    max_epochs = 4
    patience = 4

else:
    print(f'{"-"*24}')
    print(f'{"Running training routine":^20}')
    print(f'{"-"*24}\n')

with open(configuration_file) as json_file:
    config = json.load(json_file)

## create folders where to save the data and models for each fold
if folds is None:
    folds = range(config["N_FOLDS"])

for cv in folds:
    if not os.path.isdir(
        os.path.join(config["save_model_path"], "fold_" + str(cv + 1))
    ):
        os.mkdir(os.path.join(config["save_model_path"], "fold_" + str(cv + 1)))

## initialise variables where to save test summary

test_fold_summary = {}

## loop through the folds
importlib.reload(utilities_models_tf)
importlib.reload(utilities)

# ############################ TRAINING
for cv in folds:
    print(
        "Working on fold {}/{}. Start time {}".format(
            cv + 1, config["N_FOLDS"], datetime.now().strftime("%H:%M:%S")
        )
    )

    print(" - Creating datasets...")
    # get the file names for training and validation
    X_train = config["training"][cv]
    X_val = config["validation"][cv]

    if debug is True:
        # train on 20% of the dataset
        X_train = X_train[0 : int(np.ceil(0.2 * len(X_train)))]
        X_val = X_val[0 : int(np.ceil(0.2 * len(X_val)))]

    for f in X_train:
        if not os.path.isfile(f):
            raise ValueError(f"{f} not found")

    # create datasets
    train_dataset = utilities.TFR_2D_dataset(
        X_train,
        dataset_type="train",
        batch_size=config["batch_size"],
        buffer_size=5000,
        crop_size=config["input_size"],
    )

    # # set normalization layer on the training dataset
    # tr_feature_ds = train_dataset.map(lambda x, y: x)
    # normalizer = layers.experimental.preprocessing.Normalization(axis=-1)
    # normalizer.adapt(tr_feature_ds)
    normalizer = None

    val_dataset = utilities.TFR_2D_dataset(
        X_val,
        dataset_type="test",
        batch_size=config["batch_size"],
        buffer_size=1000,
        crop_size=config["input_size"],
    )

    # create model based on specification
    if config["model_configuration"] == "LightOCT":
        model = models_tf.LightOCT(
            number_of_input_channels=config["num_channels"]
            if "num_channels" in config.keys()
            else 1,
            model_name=config["model_configuration"],
            num_classes=len(config["unique_labels"]),
            data_augmentation=config["data_augmentation"],
            class_weights=config["class_weights"],
            kernel_size=config["kernel_size"],
            input_size=config["input_size"],
        )
    elif config["model_configuration"] == "M2":  # (REMAINING FROM TESTING)
        model = models_tf.M2(
            number_of_input_channels=1,
            model_name="M2",
            num_classes=len(config["unique_labels"]),
            input_size=config["input_size"],
            data_augmentation=config["data_augmentation"],
            class_weights=config["class_weights"],
            kernel_size=config["kernel_size"],
        )
    elif config["model_configuration"] == "M4":
        model = models_tf.M4(
            number_of_input_channels=config["num_channels"]
            if "num_channels" in config.keys()
            else 1,
            model_name=config["model_configuration"],
            normalization=config["model_normalization"],
            dropout_rate=config["dropout_rate"],
            num_classes=len(config["unique_labels"]),
            data_augmentation=config["data_augmentation"],
            class_weights=config["class_weights"],
            kernel_size=config["kernel_size"],
        )
    elif config["model_configuration"] == "M6":
        model = models_tf.M6(
            number_of_input_channels=1,
            input_size=config["input_size"],
            model_name=config["model_configuration"],
            num_classes=len(config["unique_labels"]),
            data_augmentation=config["data_augmentation"],
            class_weights=config["class_weights"],
            kernel_size=config["kernel_size"],
        )
    elif config["model_configuration"] == "ResNet50":
        model = models_tf.ResNet50(
            number_of_input_channels=1,
            model_name=config["model_configuration"],
            num_classes=len(config["unique_labels"]),
            data_augmentation=config["data_augmentation"],
            class_weights=config["class_weights"],
        )
    elif config["model_configuration"] == "EfficientNet_B7":  # (REMAINING FROM TESTING)
        model = models_tf.EfficientNet_B7(
            number_of_input_channels=1,
            model_name=config["model_configuration"],
            num_classes=len(config["unique_labels"]),
            data_augmentation=config["data_augmentation"],
            class_weights=config["class_weights"],
        )
    elif config["model_configuration"] == "InceptionV3":  # (REMAINING FROM TESTING)
        model = models_tf.InceptionV3(
            number_of_input_channels=1,
            model_name=config["model_configuration"],
            num_classes=len(config["unique_labels"]),
            data_augmentation=config["data_augmentation"],
            class_weights=config["class_weights"],
            input_size=config["input_size"],
        )
    elif config["model_configuration"] == "VAE_DEBUG":  # (REMAINING FROM TESTING)
        model = models_tf.VAE_DEBUG(
            number_of_input_channels=1,
            model_name=config["model_configuration"],
            num_classes=len(config["unique_labels"]),
            data_augmentation=config["data_augmentation"],
            class_weights=config["class_weights"],
            kernel_size=config["kernel_size"],
            input_size=config["input_size"],
        )
    elif config["model_configuration"] == "ViT":
        model = models_tf.ViT(
            number_of_input_channels=1,
            model_name=config["model_configuration"],
            num_classes=len(config["unique_labels"]),
            data_augmentation=config["data_augmentation"],
            class_weights=config["class_weights"],
            input_size=config["input_size"],
            patch_size=config["vit_patch_size"],
            projection_dim=config["vit_projection_dim"],
            num_heads=config["vit_num_heads"],
            mlp_head_units=config["vit_mlp_head_units"],
            transformer_layers=config["vit_transformer_layers"],
            transformer_units=config["vit_transformer_units"],
        )
    else:
        raise ValueError(
            "Specified model configuration not available. Provide one that is implemented in models_tf.py"
        )
        sys.exit()

    # train model
    print(" - Training fold...")
    warm_up = (False,)
    warm_up_epochs = 5
    warm_up_learning_rate = 0.00001

    if "VAE" in config["model_configuration"]:  # (REMAINING FROM TESTING)
        utilities_models_tf.train_VAE(
            model,
            train_dataset,
            val_dataset,
            classification_type=config["classification_type"],
            unique_labels=config["unique_labels"],
            loss=[config["loss"]],
            start_learning_rate=config["learning_rate"],
            scheduler="constant",
            vae_kl_weight=config["vae_kl_weight"],
            vae_reconst_weight=config["vae_reconst_weight"],
            power=0.1,
            max_epochs=max_epochs,
            early_stopping=True,
            patience=patience,
            warm_up=warm_up,
            warm_up_epochs=warm_up_epochs,
            warm_up_learning_rate=warm_up_learning_rate,
            save_model_path=os.path.join(
                config["save_model_path"], "fold_" + str(cv + 1)
            ),
            save_model_architecture_figure=True if cv == 0 else False,
            verbose=config["verbose"],
        )
    else:
        utilities_models_tf.train(
            model,
            train_dataset,
            val_dataset,
            classification_type=config["classification_type"],
            unique_labels=config["unique_labels"],
            loss=[config["loss"]],
            start_learning_rate=config["learning_rate"],
            scheduler="constant",
            power=0.1,
            max_epochs=max_epochs,
            early_stopping=True,
            patience=patience,
            save_model_path=os.path.join(
                config["save_model_path"], "fold_" + str(cv + 1)
            ),
            save_model_architecture_figure=True if cv == 0 else False,
            warm_up=warm_up,
            warm_up_epochs=warm_up_epochs,
            warm_up_learning_rate=warm_up_learning_rate,
            verbose=config["verbose"],
        )
