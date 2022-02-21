'''
Script designed for the thyroid OCT dataset with file names specified by the
create_dataset.py script. This script takes out from the entire file list the
test set. This is done in a way that for each default classification type (c1,
c2, c3) each class is equally represented by 1000 images originating from at
least two volumes.
This is not trivial since not always the images have coherent classes among the
classification tasks.
Given that each volume can be attributed to a class based on three numeric
values, we can use this information to take out images from each class and each
classification type.
In particular, we will have:
- c1 (normal-vs-abnormal): total of 2000 test images (1000 for each class) with
            the images in the abnormal class taken from all the abnormal classes
            e.g. 200 from goiter, 200 from adenoma, etc.
- c2 (normal-vs-enlarged-vs-shrunk-vs-depleted): a total of 4000 images (1000
            from each class). Also here, for the classes that include more
            diseased (e.g. Cancer can be both depleted and shrunk) an equal
            number of images from each disease will be picked.
- c3 (norma-vs-all the diseases): 6000 test images with 1000 from each class
'''

import os
import sys
import json
import glob
import types
import csv
import time
import random
import argparse
import importlib
import numpy as np
from collections import OrderedDict
## auxiliary function
def count_class_files(file_name, class_counter, b_scans):
    '''
    Simple utility that adds the number of b-scans to the right class in
    the different classification tasks

    Parameters
    ----------
    file_name : str
        String contaiining the file name which encodes the class type for each
        classification type
    class_counter : dict
        Dictionary containing the overall count for each classification type
    b_scans : int
        Number of b-scans to add

    Output
    -----
    class_counter : dict
        Updeted dictionary
    '''

    c1 = int(file_name[file_name.find('c1')+3])
    c2 = int(file_name[file_name.find('c2')+3])
    c3 = int(file_name[file_name.find('c3')+3])

    # count file for each class (long series of if - can be made more elegant)
    # c1
    if c1 == 0:
        class_counter['c1'][0] += b_scans
    elif c1 == 1:
        class_counter['c1'][1] += b_scans
    else:
        class_counter['c1'][2] += b_scans

    # c2
    if c2 == 0:
        class_counter['c2'][0] += b_scans
    elif c2 == 1:
        class_counter['c2'][1] += b_scans
    elif c2 == 2:
        class_counter['c2'][2] += b_scans
    elif c2 == 3:
        class_counter['c2'][3] += b_scans
    elif c2 == 9:
        class_counter['c2'][4] += b_scans

    # c3
    if c3 == 0:
        class_counter['c3'][0] += b_scans
    elif c3 == 1:
        class_counter['c3'][1] += b_scans
    elif c3 == 2:
        class_counter['c3'][2] += b_scans
    elif c3 == 3:
        class_counter['c3'][3] += b_scans
    elif c3 == 4:
        class_counter['c3'][4] += b_scans
    elif c3 == 5:
        class_counter['c3'][5] += b_scans
    elif c3 == 9:
        class_counter['c3'][6] += b_scans

    return class_counter

def count_files(file_list, search_key):
    '''
    Utility that given a list of files, counts how mani files have the string
    search_key in it.
    '''
    aus = [search_key in f for f in file_list]
    return np.sum(np.array(aus))

## load the annotation_information.csv file

dataset_specs = '/flush/iulta54/Research/Data/OCT/Thyroid_2019_DL/annotation_information.csv'
dataset_folder ='/flush/iulta54/Research/Data/OCT/Thyroid_2019_DL/2D_isotropic_TFR'

volumes = []
# get only file names without extension
files = [os.path.basename(x).rsplit('.',1)[0] for x in glob.glob(os.path.join(dataset_folder, '*'))]
# remove the train_test_split.json file if present
if 'train_test_split' in files:
    files.remove('train_test_split')

annotation_code = []
class_volume_counter = {'c1':[0,0,0],
                        'c2':[0,0,0,0,0],
                        'c3':[0,0,0,0,0,0,0]
                        }

# indexes = [i for i, x in enumerate(annotation_code) if x==0]

with open(dataset_specs) as csvfile:
    file = csv.reader(csvfile, delimiter=',', skipinitialspace=False)
    # skip header
    next(file)
    # get file information
    for row in file:
        sample = row[0]
        scan_code = row[1]
        classification_name = row[2]
        # bscans = int(row[11])
        bscans = count_files(files, scan_code)
        # infere label for the different classes based on the name
        c1 = int(classification_name[classification_name.find('c1')+3])
        c2 = int(classification_name[classification_name.find('c2')+3])
        c3 = int(classification_name[classification_name.find('c3')+3])
        volumes.append({'sample':sample, 'scan_code':scan_code, 'file_name':classification_name, 'c1':c1, 'c2':c2, 'c3':c3, 'n_bscans':bscans})
        annotation_code.append(int(str(c1)+str(c2)+str(c3)))

        # count file for each class (long series of if - can be made more elegant)
        class_volume_counter = count_class_files(classification_name, class_volume_counter,1)


# set the numer of images per class
n_images_per_class = 1000
min_n_volumes = 2

'''
This means that the n_images_per_class images for each class need to be coming
from at least min_n_volumes different volumes.
'''

## For every type of class, get a shuffled list of the volumes belonging to the class
normal_index = np.where(np.array(annotation_code) == 0)[0]
goiter_index = np.where(np.array(annotation_code) == 111)[0]
adenoma_index = np.where(np.array(annotation_code) == 122)[0]
graves_index = np.where(np.array(annotation_code) == 124)[0]
cancer_shrunk_index = np.where(np.array(annotation_code) == 125)[0]
cancer_depleted_index = np.where(np.array(annotation_code) == 135)[0]
hashimoto_index = np.where(np.array(annotation_code) == 123)[0]

random.seed (19)

random.shuffle(normal_index)
random.shuffle(goiter_index)
random.shuffle(adenoma_index)
random.shuffle(graves_index)
random.shuffle(cancer_shrunk_index)
random.shuffle(cancer_depleted_index)
random.shuffle(hashimoto_index)

# save information for easy access
classes = ['normal', 'goiter', 'adenoma', 'graves', 'cancer_shrunk', 'cancer_depleted', 'hashimoto']
per_class_random_indexes = {'normal':normal_index,
                            'goiter':goiter_index,
                            'adenoma':adenoma_index,
                            'graves':graves_index,
                            'cancer_shrunk':cancer_shrunk_index,
                            'cancer_depleted':cancer_depleted_index,
                            'hashimoto':hashimoto_index}

for c in classes:
    print(f'Class: {c:15s} -> {len(per_class_random_indexes[c]):2d} total volumes')

## get volumes that have at least n_images_per_class images from each class

per_class_random_volumes = {}

for c in classes:
    count = 0
    idx = 0
    per_class_random_volumes[c] = []
    while not all([count >= n_images_per_class, idx >= min_n_volumes]):
        per_class_random_volumes[c].append(volumes[per_class_random_indexes[c][idx]])
        count += volumes[per_class_random_indexes[c][idx]]['n_bscans']
        idx += 1

for c in classes:
    print(f'Class: {c:15} -> {len(per_class_random_volumes[c]):2d} selected volumes')


## get the files for the randomly selected volumes

per_class_random_files = {}
index_of_selected_files = []

for c in classes:
    per_class_random_files[c] = []
    for v in per_class_random_volumes[c]:
        per_class_random_files[c].extend([f for f in files if v['file_name'] in f])
        index_of_selected_files.extend([i for i, f in enumerate(files) if v['file_name'] in f])

for c in classes:
    print(f'Class: {c:15} -> {len(per_class_random_files[c]):4d} files')


## organize n_images_per_class in the test sets

per_classification_test_files = {}

# c1 - get n_images_per_class from normal and remaining equally distributed from
# the other classes
per_classification_test_files['c1'] = random.sample(per_class_random_files['normal'], n_images_per_class)
n_other_classes = int(np.ceil(n_images_per_class/5))

per_classification_test_files['c1'].extend(random.sample(per_class_random_files['goiter'], n_other_classes))
per_classification_test_files['c1'].extend(random.sample(per_class_random_files['adenoma'], n_other_classes))
per_classification_test_files['c1'].extend(random.sample(per_class_random_files['graves'], n_other_classes))
per_classification_test_files['c1'].extend(random.sample(per_class_random_files['hashimoto'], n_other_classes))
per_classification_test_files['c1'].extend(random.sample(per_class_random_files['cancer_shrunk'], n_other_classes//2))
per_classification_test_files['c1'].extend(random.sample(per_class_random_files['cancer_depleted'], n_other_classes//2))


print(f'Number of test images for c1: {len(per_classification_test_files["c1"])}')

# c2 - get n_images_per_class from normal, enlarged, shrunk (equally distributed
# from the one that are shrunk) and depleted (equally distributed from the one
# that are depleted)
per_classification_test_files['c2'] = random.sample(per_class_random_files['normal'], n_images_per_class)
per_classification_test_files['c2'].extend(random.sample(per_class_random_files['goiter'], n_images_per_class))
per_classification_test_files['c2'].extend(random.sample(per_class_random_files['adenoma'], n_images_per_class//4))
per_classification_test_files['c2'].extend(random.sample(per_class_random_files['graves'], n_images_per_class//4))
per_classification_test_files['c2'].extend(random.sample(per_class_random_files['hashimoto'], n_images_per_class//4))
per_classification_test_files['c2'].extend(random.sample(per_class_random_files['cancer_shrunk'], n_images_per_class//4))
per_classification_test_files['c2'].extend(random.sample(per_class_random_files['cancer_depleted'], n_images_per_class))

print(f'Number of test images for c2: {len(per_classification_test_files["c2"])}')


# c3 - get n_images_per_class from all the classes (from cancer get equal number
# from shrunk and enlarged)
per_classification_test_files['c3'] = random.sample(per_class_random_files['normal'], n_images_per_class)
per_classification_test_files['c3'].extend(random.sample(per_class_random_files['goiter'], n_images_per_class))
per_classification_test_files['c3'].extend(random.sample(per_class_random_files['adenoma'], n_images_per_class))
per_classification_test_files['c3'].extend(random.sample(per_class_random_files['graves'], n_images_per_class))
per_classification_test_files['c3'].extend(random.sample(per_class_random_files['hashimoto'], n_images_per_class))
per_classification_test_files['c3'].extend(random.sample(per_class_random_files['cancer_shrunk'], n_images_per_class//2))
per_classification_test_files['c3'].extend(random.sample(per_class_random_files['cancer_depleted'], n_images_per_class//2))

print(f'Number of test images for c3: {len(per_classification_test_files["c3"])}')

## get all the remaining files

remaining_test_files = [f for i, f in enumerate(files) if i not in index_of_selected_files]

print(f'Number of remaining training images: {len(remaining_test_files)}')

## save the file in a .json file

json_dict = OrderedDict()
json_dict['name'] = "Sparse_3D_OCT_Thyroid_Classification_training_test_split"
json_dict['description'] = "Training and testing split for the default classifications"
json_dict['imageSize'] = "3D"
json_dict['licence'] = ""
json_dict['release'] = "1.0"
json_dict['modality'] = "Spectral_Domain_OCT"

json_dict['c1_test'] = per_classification_test_files['c1']
json_dict['c2_test'] = per_classification_test_files['c2']
json_dict['c3_test'] = per_classification_test_files['c3']
json_dict['training'] = remaining_test_files

with open(os.path.join(os.path.dirname(dataset_folder),'train_test_split.json'), 'w') as fp:
    json.dump(json_dict, fp)

