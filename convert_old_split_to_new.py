'''
Script that given an old train_val_test_filenames_json.txt file, creates the
train_test_split.json file used in the new training routine.
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

## help function to get old files from the new dataset
def fixFileNames(file_names, new_dataset_folder, debug=False):
    '''
    Given a list of files using the old name convention, converts them in the
    new convention name using the files in the new dataset folder.
    '''

    '''
    Get unique sample and scan codes from the old file names. There files used
    the following convention: Sample-code_Scan-code_slice-number_label_x.extention
    '''
    # get only the file name without extention
    file_names = [os.path.splitext(os.path.basename(f))[0] for f in file_names]

    # get sample-code_scan-code
    old_sample_scan_id = [f[0:f.find('_')+5] for f in file_names]
    old_sample_scan_id = list(set(old_sample_scan_id))

    # get the labels for the different sample-code_scan-code
    old_sample_scan_class = []
    for old_id in old_sample_scan_id:
        for f in file_names:
            if old_id in f:
                old_sample_scan_class.append(f[-11])
                break
        # print(f'ID {old_id} -> label {old_sample_scan_class[-1]}')

    # get all the files in the new dataset folder. Here also take only the
    # file name without extension
    new_dataset_files = glob.glob(os.path.join(new_dataset_folder, '*'))
    # pop out any possible train_test_split split file
    new_dataset_files = [f for f in new_dataset_files if not 'train_test_split' in f]
    new_dataset_files = [os.path.splitext(os.path.basename(f))[0] for f in new_dataset_files]

    # get unique sample, scan and volume identifiers
    new_sample_scan_id = [f[0:f.find('_')+8] for f in new_dataset_files]
    new_sample_scan_id = list(set(new_sample_scan_id))


    new_files = []
    new_sample_scan_class = []
    not_found = []
    # check that the old sample and scan id exist in the new dataset
    for idx, old_id in enumerate(old_sample_scan_id):
        aus = []
        for new_id in new_sample_scan_id:
            # check if old ID can be found in the new IDs
            if old_id in new_id:
                # id match. Check if multiple or only one id. If only one id,
                # then take the files in the new dataset folder belonging to
                # the new id and save.
                aus.append(new_id)
        if len(aus) == 0:
            # old ID was not found among the new ones
            not_found.append(old_id)
        else:
            # old ID was found, get files
            for i in aus:
                new_files.extend([f for f in new_dataset_files if i in f])
                # save new label information (just for debug)
                new_sample_scan_class.append(new_files[-1][-18:-4])
                # print(f'ID {old_id:10s} -> old label {old_sample_scan_class[idx]} || new id {aus} -> new label {new_sample_scan_class}')

    if debug:
        print(f'# original files: {len(file_names):5d} - # new files: {len(new_files):5d}')
        if not_found:
            print(f'This old sample IDs were not found in the new dataset, thus skipped: {not_found}')
        else:
            print('All old IDs were found')

    return new_files, not_found

## set paths

# where the 4 different types of datasets are saved (2D_anisotropic and 2D_isotropic - nifit and tfrecords)
dataset_folder = '/flush/iulta54/Research/Data/OCT/Thyroid_2019_refined_DeepLearning/'

''''
Need to load files for c1, c2 and c3 classification since the testing test is
different for the different classification sets.
'''
old_files = {'c1': '/flush/iulta54/Research/P3-THR_DL/trained_models_old/VAE_original_c1_isotropic_wa_lsd128/train_val_test_filenames_json.txt',
             'c2': '/flush/iulta54/Research/P3-THR_DL/trained_models_old/VAE_original_c2_isotropic_wa_lsd128/train_val_test_filenames_json.txt',
             'c3': '/flush/iulta54/Research/P3-THR_DL/trained_models_old/VAE_original_c3_isotropic_wa_lsd128/train_val_test_filenames_json.txt'}

old_files = {'c1': '/flush/iulta54/Research/P3-THR_DL/trained_models_old/LightOCT_c1_isotropic_wa_ks3/train_val_test_filenames_json.txt',
             'c2': '/flush/iulta54/Research/P3-THR_DL/trained_models_old/LightOCT_c2_isotropic_wa_ks3/train_val_test_filenames_json.txt',
             'c3': '/flush/iulta54/Research/P3-THR_DL/trained_models_old/LightOCT_c3_isotropic_wa_ks3/train_val_test_filenames_json.txt'}

## open files and convert file names
per_classification_test_files = {}
remaining_test_files = {}
per_classification_skipped_volumes = {}
per_classification_num_files = {}


for classification, file_path in old_files.items():
    # open file
    with open(file_path) as old_file:
        old_split = json.load(old_file)

    # convert testing and training/validation files
    new_files, skipped_vol = fixFileNames(old_split['Testing'], os.path.join(dataset_folder,'2D_isotropic_TFR'))
    per_classification_test_files[classification] = new_files
    per_classification_skipped_volumes[classification] = skipped_vol
    per_classification_num_files[classification] = {'old_testing': len(old_split['Testing']),
                                                   'new_testing': len(new_files),
                                                   'old_training': 0,
                                                   'new_training': 0
                                                }

    # go through the cross validations and add together the validation and training files
    # the general configure_training.py script will take care of setting the
    # cross validation
    remaining_test_files[classification] = []
    aus_file_list = []

    for cv in range(len(old_split['Training'])):
        aus_file_list.extend(old_split['Training'][cv])
        aus_file_list.extend(old_split['Validation'][cv])

    new_files, skipped_vol = fixFileNames(aus_file_list, os.path.join(dataset_folder,'2D_isotropic_TFR'))
    remaining_test_files[classification].extend(new_files)
    per_classification_skipped_volumes[classification].extend(skipped_vol)

    per_classification_num_files[classification]['old_training'] = len(aus_file_list)
    per_classification_num_files[classification]['new_training'] = len(new_files)


    # now make sure that none of the sets files are in the training set
    for ts in per_classification_skipped_volumes[classification]:
        for tr in remaining_test_files[classification]:
            if ts in tr:
                raise ValueError('Some files in the test dataset are also found in the training split. Check implementation')

## print summary information
print(f'Note that old trainings were using oversampling, thus the number of files is infated compared to the real ones')
print(f'{"Classification type":^21s}|{"# old test":^12s}|{"# new test":^12s}|{"# old train":^12s}|{"# new train":^12s}')

for c in  old_files.keys():
    print(f'{c:^21s}|{str(per_classification_num_files[c]["old_testing"]):^12s}|{str(per_classification_num_files[c]["new_testing"]):^12s}|{str(per_classification_num_files[c]["old_training"]):^12s}|{str(per_classification_num_files[c]["new_training"]):^12s}')

## save the file in a .json file

json_dict = OrderedDict()
json_dict['name'] = "2D_OCT_Thyroid_Classification_training_test_split"
json_dict['description'] = "Training and testing split for the default classifications (roll back - using images from the new dataset that match the ones from the old one). Note that the roll back is not one-to-one since some of the volumes in the old dataset were excluded in the new dataset. Infromation about which volumes were excluded is saved."
json_dict['imageSize'] = "2D"
json_dict['licence'] = ""
json_dict['release'] = "1.0"
json_dict['modality'] = "Spectral_Domain_OCT"

json_dict['c1_test'] = per_classification_test_files['c1']
json_dict['c2_test'] = per_classification_test_files['c2']
json_dict['c3_test'] = per_classification_test_files['c3']
# any of the training is good
json_dict['training'] = remaining_test_files['c1']

json_dict['c1_skipped'] = list(set(per_classification_skipped_volumes['c1']))
json_dict['c2_skipped'] = list(set(per_classification_skipped_volumes['c2']))
json_dict['c3_skipped'] = list(set(per_classification_skipped_volumes['c3']))

dataset = ['2D_anisotropic', '2D_anisotropic_TFR', '2D_isotropic', '2D_isotropic_TFR']
for i in dataset:
    with open(os.path.join(dataset_folder,i,'train_test_split_rollback.json'), 'w') as fp:
        json.dump(json_dict, fp)

