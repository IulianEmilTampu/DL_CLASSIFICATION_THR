"""
Script that runs the training of a deep leaning model for classification
of 3D sparse volume of OCT thyroid data using the configurations saved by the configure_training.py file.

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
import models_3D_tf
import utilities
import utilities_models_tf

## parse the configuration file

parser = argparse.ArgumentParser(
    description="Script that runs a cross-validation training for OCT 3D sparse volume classification. It uses the configuration file created using the configure_training.py file. Run the configuration first!"
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
# configuration_file = '/flush/iulta54/Research/P3-OCT_THR/trained_models/ViT_3D_c13_lr0.000001_pts16_prjd32_batch4/config.json'
# debug = True
# max_epochs = 5
# patience = 5
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
importlib.reload(models_3D_tf)

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
    # just to make sure, shuffle again
    random.shuffle(X_train)

    if debug is True:
        # train on 20% of the dataset
        X_train = X_train[0 : int(np.ceil(0.5 * len(X_train)))]
        X_val = X_val[0 : int(np.ceil(0.5 * len(X_val)))]

    for f in X_train:
        if not os.path.isfile(f):
            raise ValueError(f"{f} not found")

    # create datasets
    train_dataset = utilities.TFR_3D_sparse_dataset(
        X_train,
        dataset_type="train",
        batch_size=config["batch_size"],
        buffer_size=1000,
        crop_size=config["input_size"],
    )

    val_dataset = utilities.TFR_3D_sparse_dataset(
        X_val,
        dataset_type="test",
        batch_size=config["batch_size"],
        buffer_size=1000,
        crop_size=config["input_size"],
    )

    # create model based on specification
    if config["model_configuration"] == "LightOCT_3D":
        model = models_3D_tf.LightOCT_3D(
            num_classes=len(config["unique_labels"]),
            num_channels=1,
            input_size=config["input_size"],
            data_augmentation=False,
            class_weights=config["class_weights"],
            kernel_size=config["kernel_size"],
            model_name=config["model_configuration"],
            debug=False,
        )

    elif config["model_configuration"] == "ViT_3D":
        model = models_3D_tf.ViT_3D(
            num_image_in_sequence=15,
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
        save_model_path=os.path.join(config["save_model_path"], "fold_" + str(cv + 1)),
        save_model_architecture_figure=True if cv == 0 else False,
        warm_up=warm_up,
        warm_up_epochs=warm_up_epochs,
        warm_up_learning_rate=warm_up_learning_rate,
        verbose=config["verbose"],
    )
