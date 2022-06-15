"""
Script that given a series of nifti volumes of OCT data, trims the volumes to not
include to top most part of the volume that is above the glass. Note that the
volume is trimmed as much as possible to still have 1.4mm in total depth (if there
is lots of space between the glass top and the tissue, by removing all the glass
part, the remaining tissue is small and infuences the resampling later on
introducing artefacts of interpolation that we do not want).

After the glass top part is removed, each 2D image in the 3D volumes is cropped
to match the final spatial size. Images are isotropically resampled to the specified
pixel resolution. Based on the resolution of the original volume 15 interspread 
images are packet together to create a sparse 3D volume covering a spatial
size of 1.4 x 1.4 x 0.3 mm in x,y, and z.

The raw data is assumed to be organized as follows:
.../raw_oct_volumes/
├── TH01
│   ├── TH01_0001_v1_c1_1_c2_2_c3_5.nii.gz
│   ├── TH01_0002_v1_c1_0_c2_3_c3_2.nii.gz
│   ├── ...
├── TH02
│   ├── TH02_0001_v1_c1_1_c2_2_c3_5.nii.gz
│   ├── TH02_0002_v1_c1_1_c2_3_c3_4.nii.gz
│   ├── ...

The name of the files is important since defines to which class the sample
belongs in the different classification types (3 for now - bynary, 5 classes and
7 classes). Samples can not be clastered based on the disease since the same
disease can be both, for example, shrunk and depleted.

Here is a decription of the file name (TH02_0002_v1_c1_0_c2_3_c3_4):
- TH02: Sample code
- 0002: scan code
- v1: volume 1 from this scan code (there can be many volumes: v2, v3, etc.)
- c1_0: identifies the binary classification (c1) and the label of the sample
        for that classification (0). For c1 there are 3 labels:
        0 = normal
        1 = abnormal
        9 = None (not classifiable)
- c2_3: identifies the 4-class classification (c2) and the label of the sample
        for that classification (3). For c2 there are 5 labels:
        0 = normal
        1 = enlarged
        2 = shrunk
        3 = depleted
        9 = None (not classifiable)
- c2_4: identifies the 5-class classificationn (c3) and the label of the sample
        for that classification (4). For c3 there are 6 labels:
        0 = normal
        1 = Goiter
        2 = adenoma
        3 = hashimoto
        4 = Graves
        5 = Cancer
        9 = None (not classifiable)

The anisotropic and isotropic images are then converted into TFRecords and saved
in the following way:
.../dataset_folder/
├── sparse_3D_isotropic
│   ├── TH06_0001_v1_c1_1_c2_2_c3_5_0001.nii
│   ├── TH06_0001_v1_c1_1_c2_2_c3_5_0002.nii
├── sparce_3D_isotropic_TFR
│   ├── TH06_0001_v1_c1_1_c2_2_c3_5_0001.tfr
│   ├── TH06_0001_v1_c1_1_c2_2_c3_5_0002.tfr


This nomenclature adds from the previous the reference to the number of the 
sparse volume originated from the original OCT data. For example 
(TH02_0001_v1_c1_1_c2_2_c3_5_0003.extension), where the last 4 digits
(0003) identify the number of the sperse volume

Note that partition between test and train+validation can not be done at this
point since there are volumes that can be classified for some classification
types but not in others (e.g. abnormal volume (c1 = 0) but unknown disease
(c3 = 9)). The spartition will be done when running the training configuration (configure_training.py).

IMPORTANT
Make sure to use a random number seed to generate the same spartition
for every run, so that the test data remains the same for all the models trained
on for a particular classification task.
"""


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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}

## ausiliary functions
def tictoc(tic=0, toc=1):
    """
    # Returns a string that contains the number of days, hours, minutes and
    seconds elapsed between tic and toc
    """
    elapsed = toc - tic
    days, rem = np.divmod(elapsed, 86400)
    hours, rem = np.divmod(rem, 3600)
    minutes, rem = np.divmod(rem, 60)
    seconds, rem = np.divmod(rem, 1)
    milliseconds = rem * 1000

    return "%2dd:%02dh:%02dm:%02ds:%02dms" % (
        days,
        hours,
        minutes,
        seconds,
        milliseconds,
    )


def data_normalization(dataset, quantile=0.995):
    """
    Normalizes the data between [-1,1] using the specified quantile value.
    Note that the data that falls outside the [-1, 1] interval is not clipped,
    thus there will be values outside the normalization interval

    Using the formula:

        x_norm = 2*(x - x_min)/(x_max - x_min) - 1

    where
    x_norm:     normalized data in the [-1, 1] interval
    x:          data to normalize
    x_min:      min value based on the specified quantile
    x_max:      max value based on the specified quantile
    """

    x_min = np.quantile(dataset, 1 - quantile)
    x_max = np.quantile(dataset, quantile)

    return 2.0 * (dataset.astype("float32") - x_min) / (x_max - x_min) - 1.0


def resample_image(img, img_res, iso_res):
    from skimage.transform import resize

    """
    Returns the isotropic resampled version of an input image.

    Parameters
    ----------
    img : numpy array
        The image (2D) to be resampled
    img_res : numpy array or list
        Specifies the raw resoultion of the image
    iso_res : float
        Specifies the resolution to which resample the image

    Output
    ------
    resampled_img : numpy array
        The resampled image

    Steps
    -----
    1 - From the original size, the original resolution and the isotropic
    resolution infere the final size of the volume
    2 - resample the volume using the skimage.transform.resize function
    """

    final_shape = np.floor((img.shape * img_res) / iso_res)
    return resize(img, final_shape, order=3, mode="reflect", anti_aliasing=True)


def get_first_glass_enface(volume):
    """
    Returns the index of the first enface slice that shows the glass reflection
    in an OCT volume data.
    INPUT
    volume: OCT volumetric data with axes ordered [z,x,y]
    """
    s = 0
    for i in range(volume.shape[0]):
        # for all the enface slices, check if it is the first glass slide
        intensity_threshold = 85
        n_of_white_pixels = 500
        # arbitrary threshold that identifies the white pixels due to the glass reflection
        aus = np.sum(np.where(volume[i, :, :] > intensity_threshold))
        if aus > n_of_white_pixels:
            s = i
            return s
    return s


def count_class_files(file_name, class_counter, b_scans):
    """
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
    """

    c1 = int(file_name[file_name.find("c1") + 3])
    c2 = int(file_name[file_name.find("c2") + 3])
    c3 = int(file_name[file_name.find("c3") + 3])

    # count file for each class (long series of if - can be made more elegant)
    # c1
    if c1 == 0:
        class_counter["c1"][0] += b_scans
    elif c1 == 1:
        class_counter["c1"][1] += b_scans
    else:
        class_counter["c1"][2] += b_scans

    # c2
    if c2 == 0:
        class_counter["c2"][0] += b_scans
    elif c2 == 1:
        class_counter["c2"][1] += b_scans
    elif c2 == 2:
        class_counter["c2"][2] += b_scans
    elif c2 == 3:
        class_counter["c2"][3] += b_scans
    elif c2 == 9:
        class_counter["c2"][4] += b_scans

    # c3
    if c3 == 0:
        class_counter["c3"][0] += b_scans
    elif c3 == 1:
        class_counter["c3"][1] += b_scans
    elif c3 == 2:
        class_counter["c3"][2] += b_scans
    elif c3 == 3:
        class_counter["c3"][3] += b_scans
    elif c3 == 4:
        class_counter["c3"][4] += b_scans
    elif c3 == 5:
        class_counter["c3"][5] += b_scans
    elif c3 == 9:
        class_counter["c3"][6] += b_scans

    return class_counter


def get_indexes_sparse_volume(
    resolution, num_total_slices, depth=0.300, num_slices_per_volume=15
):
    """
    Utility that computes which slide index each sparced annotation volume should
    have.

    Parameters
    ----------
    resolution : float
        Spatial resolution between slices in millimiters.
    num_total_slices : int
        Number of total slices that the volume is made of
    depth : float
        Depth that each sparced volume should have (larger in case resolution
         * number of slices is larger than the the given depth). Depth given in
         millimiters.
    num_slices_per_volume : int
        Number of slices that each sparse volume MUST have.

    Output
    -----
    index_sparse_slices : list
        List of list with each (inner) list containing the indexes of the slices
        to take from the original volume to create a sparse volume of 15 slides
        with depth of at least .300 millimiters.
    """

    # TO BE IMPLEMENTED: add check of the diffferent variables

    # check that the the volume can afford at least one sparse volume
    if num_total_slices * resolution < depth:
        return []
    else:
        # we can work on the volume
        num_slides_to_cover_depth = int(np.ceil(depth / resolution))
        space_between_slices = int(
            np.round(num_slides_to_cover_depth / num_slices_per_volume)
        )
        master_sequence = list(
            range(0, num_slides_to_cover_depth, space_between_slices)
        )

        shift = 0
        index_sparse_slices = []
        while (np.array(master_sequence) + shift)[-1] < num_total_slices:
            index_sparse_slices.append(list(np.array(master_sequence) + shift))
            shift += 1

        return index_sparse_slices


# General functions to convert values to a type compatible to a tf.exampe
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):  # if value ist tensor
        value = value.numpy()  # get value of tensor
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
"""
What we need to parce is
- dataset folder where the nifti files are organized in folders based on the sample code
- destination folder where to save the dataset
- spatial size [x, y, z] dimensions in mm that the final anisotropic ans isotropic dataset should have
"""

parser = argparse.ArgumentParser(
    description="Script that prepares the OCT data for deep learning training."
)
parser.add_argument(
    "-dt",
    "--data",
    required=True,
    help="Path to the folder containing the OCT nifti files organized per sample code.",
)
parser.add_argument(
    "-ds",
    "--destination",
    required=True,
    help="Path to where to save the created datasets.",
)
parser.add_argument(
    "-s",
    "--dataset_specs",
    required=True,
    help="Path to the csv file containing the information on which volume to use and their class.",
)
parser.add_argument(
    "-ss",
    "--spatial_size",
    required=True,
    nargs="+",
    help="Final spatial size of the images in mm [depth, lateral].",
)
parser.add_argument(
    "-r",
    "--resolution",
    required=True,
    help="Resolution of the isotropic images in mm.",
)
parser.add_argument(
    "-sd",
    "--sparse_depth",
    required=True,
    help="Depth of the sparse volume in the y direction (stack of b-scans) in mm.",
)
parser.add_argument(
    "-sns",
    "--sparse_num_slices",
    required=True,
    help="Number of b-scans in each sparse 3D volume covering sparse_depth.",
)
args = parser.parse_args()

# parse variables
data_folder = args.data
destination_folder = args.destination
spatial_size = [float(i) for i in args.spatial_size]
isotropic_res = float(args.resolution)
dataset_specs = args.dataset_specs
num_slices_per_volume = int(args.sparse_num_slices)
depth = float(args.sparse_depth)

# # DEBUG
# data_folder = (
#     "/flush/iulta54/Research/Data/OCT/Thyroid_2019_refined_DeepLearning/raw_data"
# )
# destination_folder = "/flush/iulta54/Research/Data/OCT/Thyroid_2019_DL_3D"
# spatial_size = [1.4, 2.0]
# isotropic_res = 0.007
# dataset_specs = (
#     "/flush/iulta54/Research/Data/OCT/Thyroid_2019_DL/annotation_information.csv"
# )
# num_slices_per_volume = 15
# depth = 0.3


print(
    "\n\n OCT sparse 3D dataset preparation script (this may take a while depending on the number of files and their size...).\n"
)

# ### check if provided folders and variables are ok
if not os.path.isdir(data_folder):
    raise TypeError(
        "Invalid data folder. Given {} but the folder does not exist".format(
            data_folder
        )
    )

if not os.path.isdir(destination_folder):
    raise TypeError(
        "Invalid destination folder. Given {} but the folder does not exist".format(
            destination_folder
        )
    )

if not os.path.isfile(dataset_specs):
    raise TypeError(
        "Invalid dataset specification file. Given {} but the file does not exist".format(
            dataset_specs
        )
    )

if len(spatial_size) != 2:
    raise TypeError(
        "The spatial size provided is not correct. Expected 2 values, but only {} were given.".format(
            len(spatial_size)
        )
    )

# #### check if all the files specified in the dataset_specs exist in the data_folder
# load the names of the volumes and save label information
files = []

class_volume_counter = {
    "c1": [0, 0, 0],
    "c2": [0, 0, 0, 0, 0],
    "c3": [0, 0, 0, 0, 0, 0, 0],
}
with open(dataset_specs) as csvfile:
    file = csv.reader(csvfile, delimiter=",", skipinitialspace=False)
    # skip header
    next(file)
    # get file information
    for row in file:
        sample = row[0]
        scan_code = row[1]
        classification_name = row[2]
        convention = int(row[3])
        # infere label for the different classes based on the name
        c1 = int(classification_name[classification_name.find("c1") + 3])
        c2 = int(classification_name[classification_name.find("c2") + 3])
        c3 = int(classification_name[classification_name.find("c3") + 3])
        files.append(
            {
                "sample": sample,
                "scan_code": scan_code,
                "file_name": classification_name,
                "c1": c1,
                "c2": c2,
                "c3": c3,
                "convention": convention,
            }
        )

        # count file for each class (long series of if - can be made more elegant)
        class_volume_counter = count_class_files(
            classification_name, class_volume_counter, 1
        )

# loop through the files and check if they exist
missing = []
count_file = 0
for f in files:
    file_name = os.path.join(data_folder, f["sample"], f["file_name"] + ".nii")
    if not os.path.isfile(file_name):
        missing.append(file_name)
    else:
        count_file += 1

# print if some are missing
if missing:
    print("Some files specified in the dataset_specification csv file are missing:")
    for f in missing:
        print(f)
else:
    print("All the files in the dataset_specification csv file were found.")

# initiate log file where dataset preparation information is saved for debug
logfile_path = os.path.join(destination_folder, "logfile.txt")
if os.path.isfile(logfile_path):
    os.remove(logfile_path)

logfile = open(logfile_path, "a")
logfile.write("Log file for OCT data preparation. \n")
logfile.write(f'Starting at {datetime.now().strftime("%H:%M:%S")}. \n')
logfile.write("Data and destination folders checked. \n")
logfile.write(
    "All files specified in the dataset_specification csv file were found. \n"
)
logfile.write(
    f'Files per class and classification type: \n c1: {class_volume_counter["c1"]} \n c2: {class_volume_counter["c2"]} \n c3: {class_volume_counter["c3"]} \n'
)
logfile.close()

start = time.time()

## remove glass top and reshape volumes to the specified spatial size
class_bscan_counter = {
    "c1": [0, 0, 0],
    "c2": [0, 0, 0, 0, 0],
    "c3": [0, 0, 0, 0, 0, 0, 0],
}

for counter, f in enumerate(files):
    print(
        f'Volume {counter+1:3d}/{count_file:3d} - {"step 1/3 (cropping and resampling)":40s} \r',
        end="",
    )

    # initiate log dictionary
    log_dict = {}

    # open nifti file
    file_name = os.path.join(data_folder, f["sample"], f["file_name"] + ".nii")
    volume_template = nib.load(file_name)
    header = volume_template.header
    volume_data = volume_template.get_fdata().astype("float32")

    # ## get volume resolutions and find the en-face axes
    anisotropic_res = np.array(header["pixdim"][1:4])
    if np.argmin(anisotropic_res) == 0:
        # the volume has the convention that FSL uses x (depth), y (A-scan), z (interpolation)
        # transpose the volume to have the specified b-scan as the first two dimensions
        if int(f["convention"]) == 1:
            # here using the interpolated b-scan (x, z)
            volume_data = volume_data.transpose((0, 2, 1))
            anisotropic_res = np.array(
                [anisotropic_res[0], anisotropic_res[2], anisotropic_res[1]]
            )
        elif int(f["convention"]) != 2:
            raise ValueError(
                f'File convention not recognized. Given {f["convention"]}, expected 1 or 2'
            )
    else:
        raise ValueError(
            f"Unrecognized dimension order. Expecting depth as first dimension. Given {anisotropic_res}"
        )

    log_dict["Initial_anisotropic_volume_shape"] = volume_data.shape
    log_dict["Anisotropic_resolution"] = anisotropic_res
    # ## add the number of b-scans to the right classes
    class_bscan_counter = count_class_files(
        f["file_name"], class_bscan_counter, volume_data.shape[-1]
    )

    # ## get the first glass en-face slice position
    s = get_first_glass_enface(volume_data)

    # check if the remaining part of the volume has a depth of at least the
    # one specified for the spatial size in z
    remaining_slides = volume_data.shape[0] - s
    remaining_depth = remaining_slides * anisotropic_res[0]
    if remaining_depth < spatial_size[0]:
        # print('Volume {}: leaving some free air'.format(os.path.basename(file_name)))
        s = volume_data.shape[0] - int(np.ceil(spatial_size[0] / anisotropic_res[0]))
        if s < 0:
            s = 0

    log_dict["First_glass_enface_slide"] = s

    # trim the volume to remove glass top
    volume_data = volume_data[s::, :, :]

    # crop volume to the spatial size specifications
    volume_data = volume_data[
        0 : int(np.ceil(spatial_size[0] / anisotropic_res[0])),
        0 : int(np.ceil(spatial_size[1] / anisotropic_res[1])),
        :,
    ]
    # normalise volume in [-1,1] using 98% percentile
    volume_data = data_normalization(volume_data, quantile=0.98)

    log_dict["Final_anisotropic_image_shape"] = volume_data.shape[0:2]

    # ## save every 2D anisotropic b-scan as .nii and TFR
    print(
        f'Volume {counter+1:3d}/{count_file:3d} - {"step 2/3 (saving anisotropic b-scans)":40s}\r',
        end="",
    )

    # make folders
    nii_save_folder = os.path.join(destination_folder, "sparse_3D_anisotropic")
    tfr_save_folder = os.path.join(destination_folder, "sparse_3D_anisotropic_TFR")
    if not os.path.isdir(nii_save_folder):
        os.makedirs(nii_save_folder)
    if not os.path.isdir(tfr_save_folder):
        os.makedirs(tfr_save_folder)

    # get the list of which slides to use for every sparce-volume
    slices = get_indexes_sparse_volume(
        resolution=anisotropic_res[-1],
        num_total_slices=volume_data.shape[-1],
        depth=depth,
        num_slices_per_volume=num_slices_per_volume,
    )

    # go through the volume, take out the different slice and save
    if slices:
        log_dict["Num_sparce_volumes"] = len(slices)
        for idx, sparse_slices in enumerate(slices):
            sparse_volume = volume_data[:, :, sparse_slices]
            file_name = "_".join([f["file_name"], "%03d" % (idx)])
            # save .nii
            nib.save(
                nib.Nifti1Image(
                    sparse_volume, volume_template.affine, volume_template.header
                ),
                os.path.join(nii_save_folder, file_name + ".nii"),
            )
            # save TFR
            writer = tf.io.TFRecordWriter(
                os.path.join(tfr_save_folder, file_name + ".tfrecords")
            )
            # Creates a tf.Example message ready to be written to a file for all the images
            feature = {
                "xdim": _int64_feature(sparse_volume.shape[0]),
                "zdim": _int64_feature(sparse_volume.shape[1]),
                "ydim": _int64_feature(sparse_volume.shape[-1]),
                "nCh": _int64_feature(1),
                "sparse_volume": _bytes_feature(serialize_array(sparse_volume)),
                "label_c1": _int64_feature(int(f["c1"])),
                "label_c2": _int64_feature(int(f["c2"])),
                "label_c3": _int64_feature(int(f["c3"])),
                "file_name": _bytes_feature(serialize_array(file_name)),
                "sparse_volume_depth": _float_feature(depth),
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
    else:
        log_dict["Num_sparce_volumes"] = None

    # ## save every 2D isotropic b-scan as .nii and TFR
    print(
        f'Volume {counter+1:3d}/{count_file:3d} - {"step 3/3 (saving isotropic b-scans)":40s}\r',
        end="",
    )

    # make folders
    nii_save_folder = os.path.join(destination_folder, "sparse_3D_isotropic")
    tfr_save_folder = os.path.join(destination_folder, "sparse_3D_isotropic_TFR")
    if not os.path.isdir(nii_save_folder):
        os.makedirs(nii_save_folder)
    if not os.path.isdir(tfr_save_folder):
        os.makedirs(tfr_save_folder)

    # isotropic resampling of each b-scan in the volume
    isotropic_volume_data = []
    for scan in range(volume_data.shape[-1]):
        # get the b-scan
        b_scan = resample_image(
            volume_data[:, :, scan], anisotropic_res[0:2], isotropic_res
        )
        isotropic_volume_data.append(b_scan)

    isotropic_volume_data = np.stack(isotropic_volume_data).transpose((1, 2, 0))

    # go through the volume, take out the different slice and save
    if slices:
        for idx, sparse_slices in enumerate(slices):
            sparse_volume = isotropic_volume_data[:, :, sparse_slices]
            file_name = "_".join([f["file_name"], "%03d" % (idx)])
            # save .nii
            nib.save(
                nib.Nifti1Image(
                    sparse_volume, volume_template.affine, volume_template.header
                ),
                os.path.join(nii_save_folder, file_name + ".nii"),
            )
            # save TFR
            writer = tf.io.TFRecordWriter(
                os.path.join(tfr_save_folder, file_name + ".tfrecords")
            )
            # Creates a tf.Example message ready to be written to a file for all the images
            feature = {
                "xdim": _int64_feature(sparse_volume.shape[0]),
                "zdim": _int64_feature(sparse_volume.shape[1]),
                "ydim": _int64_feature(sparse_volume.shape[-1]),
                "nCh": _int64_feature(1),
                "sparse_volume": _bytes_feature(serialize_array(sparse_volume)),
                "label_c1": _int64_feature(int(f["c1"])),
                "label_c2": _int64_feature(int(f["c2"])),
                "label_c3": _int64_feature(int(f["c3"])),
                "file_name": _bytes_feature(serialize_array(file_name)),
                "sparse_volume_depth": _float_feature(depth),
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

    log_dict["Isotropic_resolution"] = isotropic_res
    log_dict["Final_isotropic_image_shape"] = b_scan.shape

    # log info
    logfile = open(logfile_path, "a")
    logfile.write(f'Volume {counter+1:3d}/{count_file:3d} - {f["file_name"]} \n')
    for key, values in log_dict.items():
        logfile.write(" - {}: {}\n".format(key, values))
    logfile.write("\n")
    logfile.close()

## Save dataset information

json_dict = OrderedDict()
json_dict["name"] = "Sparse_3D_OCT_Thyroid_Classification"
json_dict[
    "description"
] = "Sparse 3D dataset of OCT b-scan images (anisotropic and isotropic) of normal and diseased thyroid tissue"
json_dict["imageSize"] = "Sparse 3D"
json_dict[
    "reference"
] = "Tampu et all., Biomedical Optics Express. 2020 Aug 1;11(8):4130-49."
json_dict["licence"] = ""
json_dict["release"] = "2.0"
json_dict["modality"] = "Spectral_Domain_OCT"

json_dict["C1_volumes"] = class_volume_counter["c1"]
json_dict["C2_volumes"] = class_volume_counter["c2"]
json_dict["C3_volumes"] = class_volume_counter["c3"]

json_dict["C1_bscans"] = class_bscan_counter["c1"]
json_dict["C2_bscans"] = class_bscan_counter["c2"]
json_dict["C3_bscans"] = class_bscan_counter["c3"]

json_dict["Sparse_depth"] = depth


with open(os.path.join(destination_folder, "dataset_info.json"), "w") as fp:
    json.dump(json_dict, fp)

end = time.time()
total_time = tictoc(start, end)

# save log information
logfile = open(logfile_path, "a")
logfile.write(
    f"\n Finished! Dataset creation took {total_time}.\n Dataset can be found at {destination_folder}."
)
logfile.close()

print("\n")
print(
    f"Dataset preparation took {total_time}. It is now available at {destination_folder}"
)
