"""
Main script that configures the training of a deep leaning model for classification
of 2D OCT thyroid images (normal vs diseased or disease type).

Steps
- create trainig/validation/test dataloader with on-the-fly augmentation
- load the CNN model and define loss function and training hyperparamters
- train the model
- save trained model along with training curves.
- run testing and save model performance
"""
#%% IMPORTS
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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# CUSTOM IMPORTS
import models_tf
import utilities
import utilities_models_tf

#%% DEFINE INLINE PARAMETERS

parser = argparse.ArgumentParser(
    description="Script that runs a cross-validation training for OCT 2D image classification."
)
parser.add_argument(
    "-wd",
    "--working_directory",
    required=False,
    help="Provide the Working Directory where the models_tf.py, utilities.py and utilities_models_tf.py files are.This folder will also be the one where the trained models will be saved. If not provided, the current working directory is used",
    default=os.getcwd(),
)
parser.add_argument(
    "-df",
    "--dataset_folder",
    required=True,
    help="Provide the Dataset Folder where the Train and Test folders are present along with the dataset information file.",
)
parser.add_argument(
    "-tts",
    "--train_test_split",
    required=False,
    help="Provide the path to the train_test_split.json file specifying the test and training dataset. This is to be used for the default classifications such as c1, c2, c3. This file can be created using the set_test_set.py file.",
    default=None,
)
parser.add_argument(
    "-mc",
    "--model_configuration",
    required=False,
    help="Provide the Model Configuration (LightOCT, M2, M3, ResNet50, VAE or others if implemented in the models_tf.py file).",
    default="LightOCT",
)
parser.add_argument(
    "-norm",
    "--model_normalization",
    required=False,
    help="Provide what type of normalization to use inside the model (BatchNorm or GroupNorm).",
    default="BatchNorm",
)
parser.add_argument(
    "-dr",
    "--dropout_rate",
    required=False,
    help="Provide the dropout rate.",
    default=0.2,
)
parser.add_argument(
    "-mn",
    "--model_name",
    required=False,
    help="Provide the Model Name. This will be used to create the folder where to save the model. If not provided, the current datetime will be used",
    default=datetime.now().strftime("%H:%M:%S"),
)
parser.add_argument(
    "-ct",
    "--classification_type",
    required=False,
    help="Provide the Classification Type. Chose between 1 (normal-vs-disease), 2 (normal-vs-enlarged-vs-shrinked) and 3 (normal-vs-all_diseases_available). If not provided, normal-vs-disease will be used.",
    default="c1",
)
parser.add_argument(
    "-cct",
    "--custom_classification_type",
    required=False,
    help="If the classification type is custom (not one of the dfefault one). If true, training test split will be generated here instead of using the already available one in the dataset folder. Note that all the custom classification arte based on the per-disease class split.",
    default=False,
)
parser.add_argument(
    "-f", "--folds", required=False, help="Number of folds. Default is 3", default="3"
)
parser.add_argument(
    "-l",
    "--loss",
    required=False,
    help="Loss to use to train the model (cce, wcce or sfce). Default is cce",
    default="cce",
)
parser.add_argument(
    "-lr", "--learning_rate", required=False, help="Learning rate.", default=0.001
)
parser.add_argument(
    "-bs", "--batch_size", required=False, help="Batch size.", default=50
)
parser.add_argument(
    "-is",
    "--input_size",
    nargs="+",
    required=False,
    help="Model input size.",
    default=(200, 200),
)
parser.add_argument(
    "-nCh",
    "--num_channels",
    required=False,
    help="Number of input channels.",
    default=1,
)
parser.add_argument(
    "-3d",
    "--sparse_3d_dataset",
    required=False,
    help="If the training dataset is 2D images (False) or 3D sparse volume (True)",
    default=False,
)
parser.add_argument(
    "-ks",
    "--kernel_size",
    nargs="+",
    required=False,
    help="Encoder conv kernel size.",
    default=(5, 5),
)
parser.add_argument(
    "-augment",
    "--augmentation",
    required=False,
    help="Specify if data augmentation is to be performed (True) or not (False)",
    default=True,
)
parser.add_argument(
    "-v",
    "--verbose",
    required=False,
    help="How much to information to print while training: 0 = none, 1 = at the end of an epoch, 2 = detailed progression withing the epoch.",
    default=0.1,
)
parser.add_argument(
    "-ids",
    "--imbalance_data_strategy",
    required=False,
    help="Strategy to use to tackle imbalance data",
    default="weights",
)
parser.add_argument(
    "-db",
    "--debug",
    required=False,
    help="True if want to use a smaller portion of the dataset for debugging",
    default=False,
)
parser.add_argument(
    "-ctd",
    "--check_training",
    required=False,
    help="If True, checks that none of the test images is in the training/validation set",
    default=True,
)
parser.add_argument(
    "-nivc",
    "--num_img_per_class_validation",
    required=False,
    help="Number of images for each class in the validation set.",
    default=1000,
)
parser.add_argument(
    "-nmv",
    "--num_min_volumes_in_validation",
    required=False,
    help="Minumim number of volumes from which the validation images are taken from.",
    default=2,
)

# VAE arguments (REMAININGS FOR TESTING)
parser.add_argument(
    "-vld",
    "--vae_latent_dim",
    required=False,
    help="Dimension of the VAE latent space",
    default=128,
)
parser.add_argument(
    "-vkl",
    "--vae_kl_weight",
    required=False,
    help="KL weight in for the VAE loss",
    default=0.1,
)
parser.add_argument(
    "-vrl",
    "--vae_reconst_weight",
    required=False,
    help="Reconstruction weight in for the VAE loss",
    default=0.1,
)


# ViT arguments
parser.add_argument(
    "-vit_ps",
    "--vit_patch_size",
    required=False,
    help="Patch size setting for the ViT model",
    default=16,
)
parser.add_argument(
    "-vit_pd",
    "--vit_projection_dim",
    required=False,
    help="Projection dimension for the ViT model",
    default=64,
)
parser.add_argument(
    "-vit_nh",
    "--vit_num_heads",
    required=False,
    help="Number of attention heads for the ViT model",
    default=4,
)
parser.add_argument(
    "-vit_mhu",
    "--vit_mlp_head_units",
    nargs="+",
    required=False,
    help="Size of the dense layers of the final classifier in the ViT model",
    default=[2048, 1024],
)
parser.add_argument(
    "-vit_tl",
    "--vit_transformer_layers",
    required=False,
    help="Number of transformer layer in teh ViT model.",
    default=8,
)
parser.add_argument(
    "-vit_tu",
    "--vit_transformer_units",
    nargs="+",
    required=False,
    help="# Size of the transformer layers in the ViT model",
    default=None,
)

args = parser.parse_args()

# parse variables
working_folder = args.working_directory
dataset_folder = args.dataset_folder
train_test_split = args.train_test_split
model_configuration = args.model_configuration
model_normalization = args.model_normalization
dropout_rate = float(args.dropout_rate)
model_save_name = args.model_name
classification_type = args.classification_type
custom_classification = args.custom_classification_type == "True"
loss = args.loss
learning_rate = float(args.learning_rate)
batch_size = int(args.batch_size)
input_size = [int(i) for i in args.input_size]
num_channels = int(args.num_channels)
sparse_3d_dataset = args.sparse_3d_dataset == "True"
data_augmentation = args.augmentation
N_FOLDS = int(args.folds)
verbose = int(args.verbose)
imbalance_data_strategy = args.imbalance_data_strategy
kernel_size = [int(i) for i in args.kernel_size]
debug = args.debug == "True"
check_training = args.check_training == "True"
n_images_per_class = int(args.num_img_per_class_validation)
min_n_volumes = int(args.num_min_volumes_in_validation)

# VAE variables (REMAININGS FROM TESTING)
vae_latent_dim = int(args.vae_latent_dim)
vae_kl_weight = float(args.vae_kl_weight)
vae_reconst_weight = float(args.vae_reconst_weight)

# ViT variables
vit_patch_size = int(args.vit_patch_size)
vit_projection_dim = int(args.vit_projection_dim)
vit_num_heads = int(args.vit_num_heads)
vit_mlp_head_units = [int(i) for i in args.vit_mlp_head_units]
vit_transformer_layers = int(args.vit_transformer_layers)
vit_transformer_units = args.vit_transformer_units
if vit_transformer_units == None:
    # compute default
    vit_transformer_units = [vit_projection_dim * 2, vit_projection_dim]

# # # # # # # # DEBUG
# working_folder = '/flush/iulta54/Research/P3-OCT_THR/'
# dataset_folder = '/flush/iulta54/Research/Data/OCT/Thyroid_2019_DL_2D_prj/2D_isotropic_TFR'
# train_test_split = '/flush/iulta54/Research/Data/OCT/Thyroid_2019_DL_2D_prj/train_test_split.json'
# model_configuration = 'M4'
# model_save_name = f'test_{model_configuration}_2D_prj'
# classification_type = 'c13'
# custom_classification = True
# loss = 'wcce'
# learning_rate = 0.0001
# dropout_rate = 0.3
# model_normalization = "BatchNorm"
# batch_size = 4
# input_size = [200, 200]
# num_channels = 2
# data_augmentation = True
# N_FOLDS = 1
# verbose = 2
# imbalance_data_strategy = 'weights'
# kernel_size = [5,5]
# check_training = False
# debug = False
# n_images_per_class = 20
# min_n_volumes = 2
# sparse_3d_dataset = False
#
# if "VAE" in model_configuration:
#     vae_latent_dim = 128
#     vae_kl_weight = 0.1
#     vae_reconst_weight = 0.1
#
# if "ViT" in model_configuration:
#     vit_patch_size = 16
#     vit_projection_dim = 64
#     vit_num_heads = 8
#     vit_mlp_head_units = [2048,  1024]
#     vit_transformer_layers = 8
#     vit_transformer_units = None
#     if vit_transformer_units == None:
#         compute default
#         vit_transformer_units = [vit_projection_dim * 2, vit_projection_dim]

#%%  CHECK INPUT VARIABLES
if os.path.isdir(working_folder):
    # check if the trained_model folder exists, if not create it
    if not os.path.isdir(os.path.join(working_folder, "trained_models")):
        print(
            "trained_model folders does not exist in the working path, creating it..."
        )
        save_path = os.path.join(working_folder, "trained_models")
        os.mkdir(save_path)
else:
    print(
        f"The provided working folder does not exist. Input a valid one. Given {working_folder}"
    )
    sys.exit()

if not os.path.isdir(dataset_folder):
    print(
        f"The dataset folder provided does not exist. Input a valid one. Given {dataset_folder}"
    )
    sys.exit()

if not custom_classification:
    # check if the train_test_split file is provided.
    if not os.path.isfile(train_test_split):
        raise ValueError(
            f"Custom classification is set to false, but give train_test_split file is not specified. Provide a valid one. Given {train_test_split}"
        )

#%% PRINT CONFIGURATION VARIABLES

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
print(f'{"Model save name":<26s}: {model_save_name}')
print(f'{"Sparse 3D dataset":<26s}: {sparse_3d_dataset}')
print(f'{"Classification type":<26s}: {classification_type}')
print(f'{"Custom classification":<26s}: {custom_classification}')
print(f'{"Loss function":<26s}: {loss}')
print(f'{"Learning rate":<26s}: {learning_rate}')
print(f'{"Batch size":<26s}: {batch_size}')
print(f'{"Input size":<26s}: {input_size}')
print(f'{"Data augmentation":<26s}: {data_augmentation} ')

if model_configuration == "VAE":  # (REMAININGS FROM TESTING)
    print(f'{"VAE latent space dimension":<26s}: {vae_latent_dim}')
    print(f'{"VAE KL loss weight":<26s}: {vae_kl_weight}')

if "ViT" in model_configuration:
    print(f'{"ViT patch size":<26s}: {vit_patch_size}')
    print(f'{"ViT projection dim.":<26s}: {vit_projection_dim}')


## get all file names, and configure based on classification type and unique labels
"""
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
"""
classification_type_dict = {}

classification_type_dict["c1"] = {}
classification_type_dict["c1"]["unique_labels"] = [0, 1]
classification_type_dict["c1"]["class_labels"] = ["normal", "abnormal"]
classification_type_dict["c1"]["filter_by"] = ["c1"]


classification_type_dict["c2"] = {}
classification_type_dict["c2"]["unique_labels"] = [0, 1, 2, 3]
classification_type_dict["c2"]["class_labels"] = [
    "normal",
    "enlarged",
    "shrunk",
    "depleted",
]
classification_type_dict["c2"]["filter_by"] = ["c2"]

classification_type_dict["c3"] = {}
classification_type_dict["c3"]["unique_labels"] = [0, 1, 2, 3, 4, 5]
classification_type_dict["c3"]["class_labels"] = [
    "normal",
    "Goiter",
    "Adenoma",
    "Hashimoto",
    "Graves",
    "Cancer",
]
classification_type_dict["c3"]["filter_by"] = ["c3"]

# normal vs shrunk-depleted
classification_type_dict["c4"] = {}
classification_type_dict["c4"]["unique_labels"] = [0, [2, 3, 4, 5]]
classification_type_dict["c4"]["class_labels"] = ["normal", "shrunk-depleted"]
classification_type_dict["c4"]["filter_by"] = ["c2", "c3"]

# normal vs enlarged
classification_type_dict["c5"] = {}
classification_type_dict["c5"]["unique_labels"] = [0, 1]
classification_type_dict["c5"]["class_labels"] = ["normal", "enlarged"]
classification_type_dict["c5"]["filter_by"] = ["c2", "c3"]

# enlarged vs shrunk-depleted
classification_type_dict["c6"] = {}
classification_type_dict["c6"]["unique_labels"] = [1, [2, 3, 4, 5]]
classification_type_dict["c6"]["class_labels"] = ["enlarged", "shrunk-depleted"]
classification_type_dict["c6"]["filter_by"] = ["c2", "c3"]

# structure-based classification
classification_type_dict["c13"] = {}
classification_type_dict["c13"]["unique_labels"] = [0, 1, [2, 3, 4, 5]]
classification_type_dict["c13"]["class_labels"] = [
    "normal",
    "enlarged",
    "shrunk-depleted",
]
classification_type_dict["c13"]["filter_by"] = ["c2", "c3"]

# density-based classification
classification_type_dict["c14"] = {}
classification_type_dict["c14"]["unique_labels"] = [[0, 1], [2, 3, 4, 5]]
classification_type_dict["c14"]["class_labels"] = ["high-density", "low-density"]
classification_type_dict["c14"]["filter_by"] = ["c2", "c3"]

# OTHER CUSTOM CLASSIFICATIONS - see class labels for more information
classification_type_dict["c7"] = {}
classification_type_dict["c7"]["unique_labels"] = [0, 2, 3, 4, 5]
classification_type_dict["c7"]["class_labels"] = [
    "normal",
    "Adenoma",
    "Hashimoto",
    "Graves",
    "Cancer",
]
classification_type_dict["c7"]["filter_by"] = ["c2", "c3"]

classification_type_dict["c8"] = {}
classification_type_dict["c8"]["unique_labels"] = [0, 2]
classification_type_dict["c8"]["class_labels"] = ["normal", "Adenoma"]
classification_type_dict["c8"]["filter_by"] = ["c2", "c3"]

classification_type_dict["c9"] = {}
classification_type_dict["c9"]["unique_labels"] = [2, 3, 4, 5]
classification_type_dict["c9"]["class_labels"] = [
    "Adenoma",
    "Hashimoto",
    "Graves",
    "Cancer",
]
classification_type_dict["c9"]["filter_by"] = ["c2", "c3"]

classification_type_dict["c10"] = {}
classification_type_dict["c10"]["unique_labels"] = [0, 3]
classification_type_dict["c10"]["class_labels"] = ["normal", "Hashimoto"]
classification_type_dict["c10"]["filter_by"] = ["c2", "c3"]

classification_type_dict["c11"] = {}
classification_type_dict["c11"]["unique_labels"] = [0, 4]
classification_type_dict["c11"]["class_labels"] = ["normal", "Graves"]
classification_type_dict["c11"]["filter_by"] = ["c2", "c3"]

classification_type_dict["c12"] = {}
classification_type_dict["c12"]["unique_labels"] = [0, 5]
classification_type_dict["c12"]["class_labels"] = ["normal", "Cancer"]
classification_type_dict["c12"]["filter_by"] = ["c2", "c3"]

# check if we are using a default classification type. If yes, use the train_test_split.json file

if custom_classification:
    print(
        f"\nClassification type is not a default one. Splitting the data accordingly."
    )
    print(
        f'{"Unique labels":<26s}: {classification_type_dict[classification_type]["unique_labels"]} '
    )
    print(
        f'{"Label description":<26s}: {classification_type_dict[classification_type]["class_labels"]} '
    )
    # infere file extention from the dataset files
    _, extension = os.path.splitext(glob.glob(os.path.join(dataset_folder, "*"))[10])
    file_names = glob.glob(os.path.join(dataset_folder, "*" + extension))

else:
    if (
        classification_type == "c1"
        or classification_type == "c2"
        or classification_type == "c3"
    ):
        if os.path.isfile(train_test_split):
            print(
                f"\nUsing default training test split (available in the train_test_split file)"
            )
            print(
                f'{"Unique labels":<26s}: {classification_type_dict[classification_type]["unique_labels"]} '
            )
            with open(train_test_split) as file:
                split = json.load(file)
                train_val_filenames = split["training"]
                print(f"Initial training files: {len(train_val_filenames)}")

                if classification_type == "c1":
                    test_filenames = split["c1_test"]
                if classification_type == "c2":
                    test_filenames = split["c2_test"]
                if classification_type == "c3":
                    test_filenames = split["c3_test"]

                # append basefolder and extention to the files
                # infere file extention from the dataset files
                _, extension = os.path.splitext(
                    glob.glob(os.path.join(dataset_folder, "*"))[10]
                )
                train_val_filenames = [
                    os.path.join(dataset_folder, f + extension)
                    for f in train_val_filenames
                ]
                test_filenames = [
                    os.path.join(dataset_folder, f + extension) for f in test_filenames
                ]

        else:
            raise ValueError(
                f"Using default classification type, but not train_test_split.json file found. Run the set_test_set.py first"
            )
    else:
        raise ValueError(
            f"Custom classification type was set to False, but the given classification type is not a default one. Given {classification_type} expecting c1, c2 or c3."
        )


#%% DATASET SPLITTING - VERY IMPORTANT PART

if custom_classification:
    print("Working on the test dataset...")
    importlib.reload(utilities)
    """
    Use n_images_per_class of at least 2 volumes for each class as test sample
    1 - find unique volumes for normal and all the different disease.
    2 - randomly select volumes for test (to reach n_images_per_class images)
    3 - randomly select n_images_per_class from the selected volumes for each class
    4 - cluster disease/normal based on the classification_type specification.
        Each class specified by the classification_type is composed by an equal
        number of images from diseases/normal that it is made of.
    5 - get all the remaining files for train+validation (make sure not to
        include disease/normal that are not specified in the custom classification)
    """

    """
    Get all files organized based on the more detailed classification (per disease).
    Set also a filter for files exclusion based on the default classifications.
    By default excluding using the more detailed classification (per disease),
    but one can be more restrictive and filter out by c1 (normal-vs-diseased)
    and c2 (normal-enlarged-shrunk-depleted).
    If only filtering on c3, there might be cases where samples are
    set as, for example, shrunk because belonging to graves but they have
    large sparse follicles (c2 = 9)
    """
    file_names, labels, per_disease_file_names = utilities.get_organized_files(
        file_names,
        classification_type=classification_type,
        custom=True,
        custom_labels=[0, 1, 2, 3, 4, 5],
        filter_by=classification_type_dict[classification_type]["filter_by"],
    )

    # 1 get unique volumes from each class
    for c in per_disease_file_names.values():
        # reduce the name to contain only the sample code and scan_code
        aus = [os.path.basename(i[0 : i.find("c1") - 1]) for i in c["file_names"]]
        c["unique_volumes"] = list(dict.fromkeys(aus))

    # 2 and 3
    random.seed(29122009)
    for c in per_disease_file_names.values():
        # skip if there are no volumes for this class or the number of total images
        # are less than the one needed and flag it
        if len(c["unique_volumes"]) == 0 or len(c["file_names"]) <= n_images_per_class:
            c["usable_class"] = False
        else:
            c["usable_class"] = True
            # for this class, shuffle the volumes and get all the images untill we reach the limit
            random.shuffle(c["unique_volumes"])
            count = 0
            idx = 0
            c["random_selected_files"] = []
            c["random_selected_files_index"] = []
            while count <= n_images_per_class or idx < min_n_volumes:
                # get all the files from that volume
                indexes = [
                    i
                    for i, f in enumerate(c["file_names"])
                    if c["unique_volumes"][idx] in f
                ]
                c["random_selected_files"].extend([c["file_names"][i] for i in indexes])
                c["random_selected_files_index"].extend(indexes)
                count += len(indexes)
                idx += 1

    # untill now we have files from every disease and for the normal samples all separated.
    # Now cluster based on the classification_type setting. Take an equal number
    # of images from every class in case these are grouped together.

    # 4
    test_filenames = []

    for l in classification_type_dict[classification_type]["unique_labels"]:
        if type(l) is list:
            # compute how many images to be take from each class in the grouping
            # during the aggregation. Use the information about the usability
            # of the class
            n_usable_classes = np.sum(
                [
                    1 if per_disease_file_names[str(ll)]["usable_class"] else 0
                    for ll in l
                ]
            )
            n_img = int(np.floor(n_images_per_class / n_usable_classes))
            # setting the number of images to all the classes in the group handling
            # the one that do not have images and the fact that the sum of all
            # images should be n_images_per_class
            for ll in l:
                if per_disease_file_names[str(ll)]["usable_class"]:
                    per_disease_file_names[str(ll)]["samples_to_take"] = n_img
                else:
                    per_disease_file_names[str(ll)]["samples_to_take"] = 0

            # add the extra images to the first class (should always be only one image)
            if n_images_per_class % n_usable_classes != 0:
                for ll in l:
                    if per_disease_file_names[str(ll)]["usable_class"]:
                        per_disease_file_names[str(ll)]["samples_to_take"] += 1
                        break

            # now actuallly get the images
            for ll in l:
                if per_disease_file_names[str(ll)]["usable_class"]:
                    test_filenames.extend(
                        random.sample(
                            per_disease_file_names[str(ll)]["random_selected_files"],
                            per_disease_file_names[str(ll)]["samples_to_take"],
                        )
                    )
                    print(
                        f'Unique label {ll}. Took {per_disease_file_names[str(ll)]["samples_to_take"]} random files from the specified classes.'
                    )
                else:
                    print(
                        f"Unique label {ll}. Took 0 files from the specified classe since no images are available."
                    )
        else:
            test_filenames.extend(
                random.sample(
                    per_disease_file_names[str(l)]["random_selected_files"],
                    n_images_per_class,
                )
            )
            print(f"Unique label {l}. Took {n_images_per_class} random files.")

    # 5 get the remaining training validation files
    train_val_filenames = []

    for l in classification_type_dict[classification_type]["unique_labels"]:
        if type(l) is list:
            for ll in l:
                train_val_filenames.extend(
                    [
                        f
                        for i, f in enumerate(
                            per_disease_file_names[str(ll)]["file_names"]
                        )
                        if i
                        not in per_disease_file_names[str(ll)][
                            "random_selected_files_index"
                        ]
                    ]
                )
        else:
            train_val_filenames.extend(
                [
                    f
                    for i, f in enumerate(per_disease_file_names[str(l)]["file_names"])
                    if i
                    not in per_disease_file_names[str(l)]["random_selected_files_index"]
                ]
            )

## compute class weights on the training dataset and apply imbalance data strategy
if debug:
    print("Running in debug mode - using less training/validation data (20000) \n")
    # random.seed(29)
    random.shuffle(train_val_filenames)
    train_val_filenames = train_val_filenames[0:20000]

(
    train_val_filenames,
    train_val_labels,
    per_disease_file_names,
) = utilities.get_organized_files(
    train_val_filenames,
    classification_type=classification_type,
    custom=not (
        classification_type == "c1"
        or classification_type == "c2"
        or classification_type == "c3"
    ),
    custom_labels=classification_type_dict[classification_type]["unique_labels"],
    filter_by=classification_type_dict[classification_type]["filter_by"],
)

class_weights = np.array(
    [len(c["file_names"]) for c in per_disease_file_names.values()]
)
if imbalance_data_strategy == "oversampling":
    # get the class with highest number of elements
    better_represented_class = np.argmax(class_weights)
    num_sample_to_eversample = [
        class_weights[better_represented_class] - len(i) for i in per_disease_file_names
    ]

    # check if oversampling is reasonable (not replicate an entire dataset more
    # than 3 times).
    rep = 0

    for idx, i in enumerate(num_sample_to_eversample):
        # only oversample where is needed
        if i != 0:
            if int(i // len(per_disease_file_names["file_names"][idx])) > rep:
                rep = int(i // len(per_disease_file_names["file_names"][idx]))

    if rep < 50:
        # sample where needed and add to the training file names
        for idx, i in enumerate(num_sample_to_eversample):
            # only oversample where is needed
            if i != 0:
                n_class_samples = len(per_disease_file_names["file_names"][idx])
                train_val_filenames.extend(
                    per_disease_file_names["file_names"][idx]
                    * int(i // n_class_samples)
                )
                train_val_filenames.extend(
                    random.sample(
                        per_disease_file_names["file_names"][idx],
                        int(i % n_class_samples),
                    )
                )

        class_weights = np.ones(len(per_disease_file_names))
        print(f"\nUsing {imbalance_data_strategy} strategy to handle imbalance data.")
        print(f"Setting loss function to cce given the oversampling strategy")
        loss = "cce"
    else:
        print(
            f"Avoiding oversampling strategy since this will imply repeating one of the classes more that 3 times"
        )
        print(
            f"Using class weights instead. Setting loss function to weighted categorical cross entropy (wcce)"
        )
        imbalance_data_strategy = "weights"
        # class_weights = class_weights.sum() / class_weights**1
        class_weights = (1 / class_weights) * (class_weights.sum())
        loss = "wcce"

elif imbalance_data_strategy == "weights":
    print(f"\nUsing {imbalance_data_strategy} strategy to handle imbalance data.")
    print(f"Setting loss function to wcce given the {imbalance_data_strategy} strategy")
    # class_weights = class_weights.sum() / class_weights**1
    class_weights = (1 / class_weights) * (class_weights.sum())
    loss = "wcce"

elif imbalance_data_strategy == "none":
    print(f"\nUsing {imbalance_data_strategy} strategy to handle imbalance data.")
    print(f"Setting loss function to {loss}.")
    class_weights = np.ones(len(per_disease_file_names))

n_train = len(train_val_filenames)
n_test = len(test_filenames)

#%% CHECK THAT NO TESTING FILES ARE IN THE TRAINING or VALIDATION POOL (THIS MIGHT TAKE TIME)
if check_training:
    print(
        "Checking if any test samples are in the training - validation pool (this may take time...)"
    )
    duplicated = []
    for idx, ts in enumerate(test_filenames):
        print(f"Checked {idx+1}/{len(test_filenames)} \r", end="")
        for tr in train_val_filenames:
            if os.path.basename(ts) == os.path.basename(tr):
                duplicated.append(ts)
                raise ValueError(
                    f"Some of the testing files are in the trianing - validation pool ({len(duplicated)} out of {len(test_filenames)}). CHECK IMPLEMENTATION!!!"
                )
    print("No testing files found in the training - validation pool. All good!!!")
else:
    print(
        f'\n {"¤"*10} \n ATTENTION! Not checking if test images are in the training/validation pool. \n Use with care!!! \n {"¤"*10}'
    )

print(
    f"\nWill train and validate on {n_train} images (some might have been removed since not classifiable in this task)"
)
print(f"Will test on {n_test} images ({n_test//len(class_weights)} for each class)")
print(f'{"Class weights":<10s}: {class_weights}')

## prepare for cross validation
"""
Make sure that images from the same volumes are not in the both the training and
validation sets. So, as before, we take out the volume names, select train and
validation for every fold and then save the images belonging to that volumes.
1 - get unique volumes for each class
2 - split each class independently for cross validation
3 - save file names for each fold
"""
print(f"\nSetting cross-validation files...")
# 1 get unique volumes from each class
for c in per_disease_file_names.values():
    # reduce the name to contain only the sample code and scan_code
    aus = [os.path.basename(i[0 : i.find("c1") - 1]) for i in c["file_names"]]
    c["unique_volumes"] = list(dict.fromkeys(aus))
    random.shuffle(c["unique_volumes"])
# 2
if N_FOLDS >= 2:
    kf = KFold(n_splits=N_FOLDS)
    per_fold_train_files = [[] for i in range(N_FOLDS)]
    per_fold_val_files = [[] for i in range(N_FOLDS)]

    for idx1, c in enumerate(per_disease_file_names.values()):
        # for all classes
        for idx, (train_volume_index, val_volume_index) in enumerate(
            kf.split(c["unique_volumes"])
        ):
            # use the indexes of the unique volumes for split the data
            # training
            for v in train_volume_index:
                tr_vol = c["unique_volumes"][v]
                per_fold_train_files[idx].extend(
                    [f for f in train_val_filenames if tr_vol in f]
                )
            # validation
            for v in val_volume_index:
                val_vol = c["unique_volumes"][v]
                # print(f'Fold {idx} - Validation: {val_vol}')
                per_fold_val_files[idx].extend(
                    [f for f in train_val_filenames if val_vol in f]
                )

    # shuffle training files (since that there can be many files, the buffer size
    # for the generator should be very large. By shuffling now we can reduce the
    # buffer size).

    for c in range(N_FOLDS):
        random.shuffle(per_fold_train_files[c])
        random.shuffle(per_fold_val_files[c])

else:
    # set 1000 images from each class as validation (like the testing)
    per_fold_train_files = [[] for i in range(N_FOLDS)]
    per_fold_val_files = [[] for i in range(N_FOLDS)]

    random.seed(29)
    per_class_random_files = []
    index_of_selected_files = []

    # randomly select as many volumes per class as needed to reach n_images_per_class
    for c in per_disease_file_names.values():
        # for this class, shuffle the volumes and get all the images untill we reach the limit
        random.shuffle(c["unique_volumes"])
        count = 0
        idx = 0
        per_class_random_files.append([])
        while count <= n_images_per_class or idx < min_n_volumes:
            # get all the files from that volume
            indexes = [
                i
                for i, f in enumerate(train_val_filenames)
                if c["unique_volumes"][idx] in f
            ]
            per_class_random_files[-1].extend([train_val_filenames[i] for i in indexes])
            index_of_selected_files.extend(indexes)
            count += len(indexes)
            idx += 1

    # 3 get exactly n_images_per_class from each class and set it to the test set
    for f in per_class_random_files:
        per_fold_val_files[0].extend(random.sample(f, n_images_per_class))

    # 4 get the remaining training validation files
    per_fold_train_files[0] = [
        f for i, f in enumerate(train_val_filenames) if i not in index_of_selected_files
    ]

# shuffle again files
for c in range(N_FOLDS):
    random.shuffle(per_fold_train_files[c])
    random.shuffle(per_fold_val_files[c])

# make sure that no training file is in the validation set
for f in per_fold_train_files[0]:
    if f in per_fold_val_files[0]:
        raise ValueError(
            "Train training split did not go as planned. Check implementation"
        )

# check that the split is valid
for c in range(N_FOLDS):
    for f in per_fold_val_files[c]:
        if f in per_fold_train_files[c]:
            print(f"File {os.path.basename(f)} in both set for fold {c}")
            raise ValueError(
                "Train validation split did not go as planned \n Some validation file are in the training set. Check implementation!"
            )

print(f"Cross-validation set. Running a {N_FOLDS}-fold cross validation")
print(f"Images from the validation set are taken from volumes not in the training sets")

for f in range(N_FOLDS):
    print(
        f"Fold {f+1}: training on {len(per_fold_train_files[f]):5d} and validation on {len(per_fold_val_files[f]):5d}"
    )

#%% SAVE INFORMATION IN A CONFIGURATION FILE USED BY THE run_training.py
"""
The configuration file will be used by the training routine to access the
the train-val-test files as well as the different set-up for the model. Having a
separate configuration file helps keeping the training routine more clean.
"""
print(f"\nSavingconfiguration file...")

json_dict = OrderedDict()
json_dict["working_folder"] = working_folder
json_dict["dataset_folder"] = dataset_folder
json_dict["sparse_3d_dataset"] = sparse_3d_dataset

json_dict["classification_type"] = classification_type
json_dict["unique_labels"] = classification_type_dict[classification_type][
    "unique_labels"
]
json_dict["label_description"] = classification_type_dict[classification_type][
    "class_labels"
]

json_dict["model_configuration"] = model_configuration
json_dict["dropout_rate"] = dropout_rate
json_dict["model_normalization"] = model_normalization
json_dict["model_save_name"] = model_save_name
json_dict["loss"] = loss
json_dict["learning_rate"] = learning_rate
json_dict["batch_size"] = batch_size
json_dict["input_size"] = input_size
json_dict["num_channels"] = num_channels
json_dict["kernel_size"] = kernel_size
json_dict["data_augmentation"] = data_augmentation

json_dict["N_FOLDS"] = N_FOLDS
json_dict["verbose"] = verbose
json_dict["imbalance_data_strategy"] = imbalance_data_strategy

json_dict["training"] = per_fold_train_files
json_dict["validation"] = per_fold_val_files
json_dict["test"] = test_filenames
json_dict["class_weights"] = list(class_weights)

# save VAE configuration if VAE model
if "VAE" in model_configuration:
    json_dict["vae_latent_dim"] = vae_latent_dim
    json_dict["vae_kl_weight"] = vae_kl_weight
    json_dict["vae_reconst_weight"] = vae_reconst_weight


# save ViT configuration if ViT model
if "ViT" in model_configuration:
    json_dict["vit_patch_size"] = vit_patch_size
    json_dict["vit_projection_dim"] = vit_projection_dim
    json_dict["vit_num_heads"] = vit_num_heads
    json_dict["vit_mlp_head_units"] = vit_mlp_head_units
    json_dict["vit_transformer_layers"] = vit_transformer_layers
    json_dict["vit_transformer_units"] = vit_transformer_units

# save file
save_model_path = os.path.join(working_folder, "trained_models", model_save_name)

if not os.path.isdir(save_model_path):
    os.mkdir(save_model_path)

json_dict["save_model_path"] = save_model_path


with open(os.path.join(save_model_path, "config.json"), "w") as fp:
    json.dump(json_dict, fp)

print(f"Configuration file created. Avvailable at {save_model_path}")
