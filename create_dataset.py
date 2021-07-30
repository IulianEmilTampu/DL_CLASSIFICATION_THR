'''
Script that given a series of nifti volumes of OCT data, trims the volumes to not
include to top most part of the volume that is above the glass. Note that the
volume is trimmed as much as possible to still have 1.4mm in total depth (if there
is lots of space between the glass top and the tissue, by removing all the glass
part, the remaining tissue is small and infuences the resampling later on
introduction of interpolation that we do not want).

After the glass top part is removed, the volume is reshaped to match the spatial
size specification (anisotropic volume).

The reshaped volumes are then resampled using FSL FLIRT functionality using the
specified final volume resolution (isotropic volumes).

When seting the spatial size and resolution, keep in mind the fact that if the
final interpolated volume is smaller it will be interpolated -> these interpolations
will biase the network.

The raw data is assumed to be organized as follows:
.../raw_oct_volumes/
├── TH01
│   ├── TH01_0001.nii.gz
│   ├── TH01_0002.nii.gz
│   ├── ...
├── TH02
│   ├── TH02_0001.nii.gz
│   ├── TH02_0002.nii.gz
│   ├── ...

The anisotropic and isotropic volumes are then converted into TFRecords and saved
in the following way:
.../dataset_folder/
├── Train
│   ├── class_1
│   │   ├── TH01_0001_0001_label_1.extension
│   │   ├── TH01_0002_0001_label_1.extension
│   │   ├── TH01_0003_0001_label_1.extension
│   ├── class_2
│   │   ├── TH01_0001_0001_label_2.extension
│   │   ├── TH01_0002_0001_label_2.extension
│   │   ├── TH01_0003_0001_label_2.extension
│   ├── ...
├── Test
│   ├── class_1
│   │   ├── TH02_0001_0001_label_1.extension
│   │   ├── TH02_0002_0001_label_1.extension
│   │   ├── TH02_0003_0001_label_1.extension
│   ├── class_2
│   │   ├── TH03_0001_0001_label_2.extension
│   │   ├── TH03_0002_0001_label_2.extension
│   │   ├── TH03_0003_0001_label_2.extension
│   ├── ...
'''

import os
import csv
import glob
import time
import json
import random
import pathlib
import argparse
import subprocess
import numpy as np
import nibabel as nib
import tensorflow as tf
from datetime import datetime
from collections import OrderedDict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

## ausiliary functions
def tictoc(tic=0, toc=1):
    '''
    # Returns a string that contains the number of days, hours, minutes and
    seconds elapsed between tic and toc
    '''
    elapsed = toc-tic
    days, rem = np.divmod(elapsed, 86400)
    hours, rem = np.divmod(rem, 3600)
    minutes, rem = np.divmod(rem, 60)
    seconds, rem = np.divmod(rem, 1)
    milliseconds = rem*1000

    return "%2dd:%02dh:%02dm:%02ds:%02dms" % (days, hours, minutes, seconds, milliseconds)

def get_first_glass_enface(volume):
    '''
    Returns the index of the first enface slice that shows the glass reflection
    in an OCT volume data.
    INPUT
    volume: OCT volumetric data with axes ordered [z,x,y]
    '''
    for s in range(volume.shape[0]):
        # for all the enface slices, check if it is the first glass slide
        threshold = 85
        # arbitrary threshold that identifies the white pixels due to the glass reflection
        aus = np.sum(np.where( volume[s,:,:] > threshold ))
        if aus > 500:
            return s

# General functions to convert values to a type compatible to a tf.exampe
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a floast_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array


## parse inline parameters and check them
'''
What we need to parce is
- dataset folder where the nifti files are organized in folders based on the sample code
- destination folder where to save the dataset
- spatial size [x, y, z] dimensions in mm that the final anisotropic ans isotropic dataset should have
'''

parser = argparse.ArgumentParser(description='Script that prepares the OCT data for deep learning training.')
parser.add_argument('-dt','--data', required=True, help='Path to the folder containing the OCT nifti files organized per sample code.')
parser.add_argument('-ds', '--destination', required=True, help='Path to where to save the created datasets.')
parser.add_argument('-s', '--dataset_specs', required=True, help='Path to the csv file containing the information on which volume to use and their class.')
parser.add_argument('-ss','--spatial_size', required=True, nargs='+', help='Spatial size of the isotropic volumes in mm [z, x, y].')
parser.add_argument('-r','--resolution', required=True, nargs='+', help='Resolution of the isotropic volumes in mm [z, x, y].')
args = parser.parse_args()

# parse variables
data_folder = args.data
destination_folder = args.destination
spatial_size = [float(i) for i in args.spatial_size]
isotropic_res = [float(i) for i in args.resolution]
dataset_specs = args.dataset_specs

print('\n\n OCT dataset preparation script (this may take a while depending on the number of files and their size...).\n')

# ### check if provided folders and variables are ok
if not os.path.isdir(data_folder):
    raise TypeError('Invalid data folder. Given {} but the folder does not exist'.format(data_folder))

if not os.path.isdir(destination_folder):
    raise TypeError('Invalid destination folder. Given {} but the folder does not exist'.format(destination_folder))

if not os.path.isfile(dataset_specs):
    raise TypeError('Invalid dataset specification file. Given {} but the folder does not exist'.format(destination_folder))

if len(spatial_size) != 3:
    raise TypeError('The spatial size provided is not correct. Expected 3 values, but only {} were given.'.format(len(spatial_size)))

if len(isotropic_res) != 3:
    raise TypeError('The resolution provided is not correct. Expected 3 values, but only {} were given.'.format(len(isotropic_res)))

# #### check if all the files specified in the dataset_specs exist in the data_folder
# load the names of the volumes divided by their labels
with open(dataset_specs) as csvfile:
    file = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
    next(file)
    # get unique classes
    classes = np.unique(np.array([int(row[3]) for row in file]))
    files = {i:[] for i in classes}

with open(dataset_specs) as csvfile:
    file = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
    next(file)
    for row in file:
        sample = row[0]
        scan_code = row[1]
        label_name = row[2]
        label_class = int(row[3])
        files[label_class].append({'sample':sample, 'scan_code':scan_code, 'label_name':label_name, 'label_class':label_class})

# loop through the files and check if they exist
missing=[]
count_file = 0
for key, value in files.items():
    for f in value:
        file_name = os.path.join(data_folder, f['sample'], f['scan_code']+ '_Mode3D.nii' )
        count_file += 1
        if not os.path.isfile(file_name):
            missing.append(file_name)

# print if some are missing
if missing:
    print('Some files specified in the dataset_specification csv file are missing:')
    for f in missing:
        print(f)
else:
    print('All the files in the dataset_specification csv file were found.')

# initiate log file where dataset preparation information is saved for debug
logfile_path = os.path.join(destination_folder, 'logfile.txt')
if os.path.isfile(logfile_path):
    os.remove(logfile_path)

logfile = open(logfile_path, 'a')
logfile.write('Log file for OCT data preparation. \n')
logfile.write('Starting at {}.\n'.format(datetime.now().strftime("%H:%M:%S")))
logfile.write('Data and destination folders checked. \n')
logfile.write('All files specified in the dataset_specification csv file were found. \n')
logfile.close()

start = time.time()

## remove glass top and reshape volumes to the specified spatial size


counter = 0
for key, value in files.items():
    for f in value:
        print('{:33s} - File {:3d}/{:3d} -> {:25s} \r'.format('Step 1/3 (cropping and resampling)',counter+1, count_file, 'Opening file'), end='')

        # initiate log dictionary
        log_dict={}

        # open nifti file
        file_name = os.path.join(data_folder, f['sample'], f['scan_code']+ '_Mode3D.nii' )
        volume_template = nib.load(file_name)
        header = volume_template.header
        volume_data = volume_template.get_fdata().astype('float32')

        # ## get volume resolutions and find the en-face axes (transpose volume and
        # resolutions to have z dimension as first)
        anisotropic_res = header['pixdim'][1:4]
        log_dict['Initial_anisotropic_volume_shape'] = volume_data.shape
        log_dict['Anisotropic_resolution'] = anisotropic_res
        if np.argmin(anisotropic_res) == 1:
            volume_data.transpose([1,0,2])
            anisotropic_res.transpose([1,0,2])
        elif np.argmin(anisotropic_res) == 2:
            volume_data.transpose([2,0,1])
            anisotropic_res.transpose([2,0,1])

        # ## get the first glass en-face slice position
        print('{:33s} - File {:3d}/{:3d} -> {:25s} \r'.format('Step 1/3 (cropping and resampling)',counter+1, count_file, 'Removing top glass'), end='')
        s = get_first_glass_enface(volume_data)

        # check if the remaining part of the volume has a depth of at least the
        # one specified for the spatial size in z
        remaining_slides = volume_data.shape[0]- s
        remaining_depth = remaining_slides*anisotropic_res[0]
        if remaining_depth < spatial_size[0]:
            # print('Volume {}: leaving some free air'.format(os.path.basename(file_name)))
            s = volume_data.shape[0] - int(np.ceil(spatial_size[0]/anisotropic_res[0]))

        log_dict['First_glass_enface_slide'] = s

        # trim the volume to remove glass top
        volume_data = volume_data[s::, :, :]

        # crop volume to the spatial size specifications
        volume_data = volume_data[0:int(np.ceil(spatial_size[0]/anisotropic_res[0])),
                                  0:int(np.ceil(spatial_size[1]/anisotropic_res[1])),
                                  0:int(np.ceil(spatial_size[2]/anisotropic_res[2]))
                                  ]
        log_dict['Final_anisotropic_volume_shape'] = volume_data.shape

        # ## save trimed and croped anisotropic volume
        print('{:33s} - File {:3d}/{:3d} -> {:25s} \r'.format('Step 1/3 (cropping and resampling)',counter+1, count_file, 'Saving anisotropic volume'), end='')
        # make folders
        anisotropic_sample_folder = os.path.join(destination_folder,'anisotropic', f['sample'])
        if not os.path.isdir(anisotropic_sample_folder):
            os.makedirs(anisotropic_sample_folder)

        volume_data = nib.Nifti1Image(volume_data, volume_template.affine, volume_template.header)
        nib.save(volume_data, os.path.join(anisotropic_sample_folder, f['scan_code']+ '.nii.gz'))

        # ## resample file using FSL flirt functions
        # make folder for isotropic files
        iso_folder = os.path.join(destination_folder,'isotropic')
        if  not os.path.isdir(iso_folder):
            os.mkdir(iso_folder)
        iso_sample_folder = os.path.join(iso_folder,f['sample'])
        if  not os.path.isdir(iso_sample_folder):
            os.mkdir(iso_sample_folder)

        # create template
        print('{:33s} - File {:3d}/{:3d} -> {:25s} \r'.format('Step 1/3 (cropping and resampling)', counter+1, count_file, 'Create FSL empty template'), end='')
        bash_comand = '/usr/local/fsl/bin/fslcreatehd {} {} {} 1 {} {} {} 1 0 0 0 16 {}'.format(
                            int(np.floor(spatial_size[0]/isotropic_res[0])),
                            int(np.floor(spatial_size[1]/isotropic_res[1])),
                            int(np.floor(spatial_size[2]/isotropic_res[2])),
                            isotropic_res[0],
                            isotropic_res[1],
                            isotropic_res[2],
                            os.path.join(iso_folder, 'temp.nii.gz'))

        os.system(bash_comand)
        log_dict['Template_creation_comand'] = bash_comand

        # isotropic resampling
        bash_comand = '/usr/local/fsl/bin/flirt -in {} -applyxfm -init /usr/local/fsl/etc/flirtsch/ident.mat -out {} -paddingsize 0.0 -interp sinc -sincwidth 7 -sincwindow hanning -datatype float -ref {}'.format(
                            os.path.join(anisotropic_sample_folder, f['scan_code'] + '.nii.gz'),
                            os.path.join(iso_sample_folder, f['scan_code']+ '.nii.gz'),
                            os.path.join(iso_folder, 'temp.nii.gz'))

        print('{:33s} - File {:3d}/{:3d} -> {:25s} \r'.format('Step 1/3 (cropping and resampling)',counter+1, count_file, 'Isotropic resampling'), end='')
        os.system(bash_comand)
        log_dict['Isotropic_resampling_comand'] = bash_comand

        #remove temp.nii.gz file
        os.remove(os.path.join(iso_folder, 'temp.nii.gz'))

        # update counter
        counter += 1

        # log info
        logfile = open(logfile_path, 'a')
        logfile.write('File {}: \n'.format(file_name))
        for key, values in log_dict.items():
            logfile.write(' - {}: {}\n'.format(key, values))
        logfile.write('\n')
        logfile.close()

## convert anisotropic and isotropic volumes in 2D TFRecond datasets (make also a version where images are .nii.gz files)
# ## randomly pick n volumes from each class and flag it as a test volume
random.seed(29)
n_test_volumes_per_class = 2

for key, value in files.items():
    # randomly pick volumes for testing
    test_idx = random.sample(range(0, len(value)), n_test_volumes_per_class)
    # set train or test flag for the volumes
    for idx2, f in enumerate(value):
        if idx2 in test_idx:
            f['dataset']='test'
        else:
            f['dataset']='train'


logfile = open(logfile_path, 'a')
logfile.write('Train and test split done using {} test volumes from each class.\n'.format(n_test_volumes_per_class))
logfile.close()

for dataset_type in ['anisotropic', 'isotropic']:
    logfile = open(logfile_path, 'a')
    logfile.write('Starting saving {} dataset.\n'.format(dataset_type))
    logfile.close()
    counter = 0
    # ## create TFR dataset folders
    TFR_dataset_folder = os.path.join(destination_folder,'2D_classification_dataset_'+ dataset_type + '_TFR')
    if not os.path.isdir(TFR_dataset_folder):
        os.mkdir(TFR_dataset_folder)
    # make train and test folders along with the folders for each class
    for i in ['Train', 'Test']:
        if not os.path.isdir(os.path.join(TFR_dataset_folder, i)):
            os.mkdir(os.path.join(TFR_dataset_folder, i))
        for j in classes:
            if not os.path.isdir(os.path.join(TFR_dataset_folder, i, 'class_'+str(j))):
                os.mkdir(os.path.join(TFR_dataset_folder, i, 'class_'+str(j)))

    logfile = open(logfile_path, 'a')
    logfile.write('TFR tree folder created.\n')
    logfile.close()

    # ## create normal .nii.gz dataset folders
    dataset_folder = os.path.join(destination_folder,'2D_classification_dataset_'+ dataset_type)
    if not os.path.isdir(dataset_folder):
        os.mkdir(dataset_folder)
    # make train and test folders along with the folders for each class
    for i in ['Train', 'Test']:
        if not os.path.isdir(os.path.join(dataset_folder, i)):
            os.mkdir(os.path.join(dataset_folder, i))
        for j in classes:
            if not os.path.isdir(os.path.join(dataset_folder, i, 'class_'+str(j))):
                os.mkdir(os.path.join(dataset_folder, i, 'class_'+str(j)))

    logfile = open(logfile_path, 'a')
    logfile.write('nii.gz tree folder created.\n')
    logfile.close()

    # ## loop through the classes
    tr_count = 0
    ts_count = 0
    tr_file_names = []
    ts_file_names = []
    tr_TFR_file_names = []
    ts_TFR_file_names = []

    for key, value in files.items():
        # loop through all the volumes in this class
        if dataset_type == 'anisotropic':
            print('{:25s} - volume {:3d}/{:3d} \r'.format('Step 2/3 - Saving {} dataset'.format(dataset_type), counter+1, count_file), end='')
        else:
           print('{:25s} - volume {:3d}/{:3d} \r'.format('Step 3/3 - Saving {} dataset'.format(dataset_type), counter+1, count_file), end='')
        for f in value:
            # load the numpy data volume
            file_name = os.path.join(destination_folder, dataset_type, f['sample'], f['scan_code']+ '.nii.gz' )
            volume_template = nib.load(file_name)
            volume_data = volume_template.get_fdata().astype('float32')
            # loop through all the (z,x) images in this volume and save
            for scan in range(volume_data.shape[-1]):
                img = np.squeeze(volume_data[:,:,scan]).astype('float32')
                file_name = f['scan_code'] + '_' + '%03d' %(scan) + '_label_' + str(f['label_class'])

                if f['dataset'] == 'train':
                    save_path = os.path.join(dataset_folder, 'Train', 'class_'+str(f['label_class']))
                    TFR_save_path = os.path.join(TFR_dataset_folder, 'Train', 'class_'+str(f['label_class']))
                    tr_file_names.append(file_name + '.nii.gz')
                    tr_TFR_file_names.append(file_name + '.tfrecords')
                    tr_count += 1
                elif f['dataset'] == 'test':
                    save_path = os.path.join(dataset_folder, 'Test', 'class_'+str(f['label_class']))
                    TFR_save_path = os.path.join(TFR_dataset_folder, 'Test', 'class_'+str(f['label_class']))
                    ts_file_names.append(file_name + '.nii.gz')
                    ts_TFR_file_names.append(file_name + '.tfrecords')
                    ts_count += 1

                # save nii.gz file
                nib.save(nib.Nifti1Image(img,
                                        affine=volume_template.affine,
                                        header=volume_template.header),
                        os.path.join(save_path, file_name + '.nii.gz'))

                # save tfrecord
                writer =  tf.io.TFRecordWriter(os.path.join(TFR_save_path,file_name + '.tfrecords'))
                # Creates a tf.Example message ready to be written to a file for all the images
                feature = {
                        'xdim' : _int64_feature(img.shape[0]),
                        'zdim' : _int64_feature(img.shape[1]),
                        'nCh' : _int64_feature(1),
                        'image' : _bytes_feature(serialize_array(img)),
                        'label' : _int64_feature(int(f['label_class']))
                        }
                # wrap feature with the Example class
                tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
                # write to file
                writer.write(tf_example.SerializeToString())

                # close file
                writer.close()
                del feature
                del tf_example
                del writer

            # save log information
            logfile = open(logfile_path, 'a')
            logfile.write('{} saved to TFR and nii.gz in {} \n'.format(f['scan_code'], dataset_folder))
            logfile.close()


            counter += 1
    # save dataset information
    json_dict = OrderedDict()
    json_dict['name'] = "2D_OCT_Thyroid_Classification"
    json_dict['description'] = "2D {} dataset of b-scan OCT images of normal and diseased thyroid tissue".format(dataset_type)
    json_dict['imageSize'] = "2D"
    json_dict['reference'] = "Tampu et all., Biomedical Optics Express. 2020 Aug 1;11(8):4130-49."
    json_dict['licence'] = ""
    json_dict['release'] = "0.0"
    json_dict['modality'] = "Spectral_Domain_OCT"
    json_dict['numTraining'] = tr_count
    json_dict['numTest'] = ts_count

    # nii.gz
    json_dict['TrainSamples'] = tr_file_names
    json_dict['Testsamples'] = ts_file_names
    json_dict['randSeed'] = None

    with open(os.path.join(dataset_folder,'dataset_info.json'), 'w') as fp:
        json.dump(json_dict, fp)

    # tfrecords
    json_dict['TrainSamples'] = tr_TFR_file_names
    json_dict['Testsamples'] = ts_TFR_file_names
    json_dict['randSeed'] = None

    with open(os.path.join(TFR_dataset_folder,'dataset_info.json'), 'w') as fp:
        json.dump(json_dict, fp)

end = time.time()
total_time = tictoc(start, end)

# save log information
logfile = open(logfile_path, 'a')
logfile.write('\n Finished! Dataset creation took {}.\n Dataset can be found at {}.'.format(total_time, destination_folder))
logfile.close()

print('\n')
print('Dataset preparation took {}. It is now available at {}'.format(total_time, destination_folder))












