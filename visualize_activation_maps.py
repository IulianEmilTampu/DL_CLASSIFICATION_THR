'''
Script that uses the implementation of Grad-CAM to visualize activation maps
for a selected layer and a selected class

Steps
1 - get model infromation and additional files
2 - load the image we want to inspect
3 - load model
4 - apply gradCam
'''

import os
import sys
import cv2
import glob
import json
import pickle
import random
import pathlib
# import imutils
import argparse
import importlib
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

# local imports
import utilities
import utilities_models_tf

'''
for in line implementation

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to the input image")
ap.add_argument("-m", "--model", type=str, default="vgg",
    choices=("vgg", "resnet"),
    help="model to be used")
args = vars(ap.parse_args())
'''



## 1 load folders and information about the model
# get dataset info from the configuration file
from_configuration_file = True
model_version = "best"
set_to_use = "test"

# get the right model based on the model_version_specification
if model_version=="best":
    model_name_version = "model.tf"
elif model_version=="last":
    model_name_version = "last_model.tf"

if from_configuration_file:
    model_name = 'ViT_3D_c3_lr0.00001_pts16_prjd32_batch4'
    trained_models_path = '/flush/iulta54/Research/P3-OCT_THR/trained_models/3D_models/c3'
    model_path = os.path.join(trained_models_path, model_name)
    if "3D" in model_name:
        dataset_path = '/flush/iulta54/Research/Data/OCT/Thyroid_2019_DL_3D'
    else:
        dataset_path = '/flush/iulta54/Research/Data/OCT/Thyroid_2019_DL'

    # load configuration file
    with open(os.path.join(model_path,'config.json')) as json_file:
        config = json.load(json_file)

    # take one testing. training and validation images (tr and val for fold specific fold)
    # make sure that the files point to this system dataset
    fold = 0
    test_img = [os.path.join(dataset_path, pathlib.Path(f).parts[-2], pathlib.Path(f).parts[-1]) for f in config['test']]
    tr_img = [os.path.join(dataset_path, pathlib.Path(f).parts[-2], pathlib.Path(f).parts[-1]) for f in config['training'][fold]]
    val_img = [os.path.join(dataset_path, pathlib.Path(f).parts[-2], pathlib.Path(f).parts[-1]) for f in config['validation'][fold]]

    # some other settings
    crop_size = config['input_size'] # (h, w)
    class_description = ', '.join([str(i)+'='+c for i, c in enumerate(config['label_description'])])

    # 3d model?
    model_3D = True if len(config["input_size"])==3 else False
else:
    # specify manually the files to show
    test_img = []
    tr_img = []
    val_img = []
    crop_size = []

    # dataset_path = '/home/iulta54/Desktop/Testing/TH_DL_dummy_dataset/Created/LigthOCT_TEST_isotropic_20000s'
    # file_names = glob.glob(os.path.join(dataset_path, '*'))
    # c_type='c1'
    # file_names, labels, organized_files = utilities.get_organized_files(file_names, c_type, categorical=False)

# make folder where to save imaes
save_path = os.path.join(trained_models_path, model_name,f'GradCAM_{set_to_use}')
pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

## 2 create dataset we want to use for getting the images to inspect
importlib.reload(utilities)
importlib.reload(utilities_models_tf)

seed = 29122009
batch_size = 50
images_to_show = 50

# check data coming out of the generators
debug = False
show = True
model_to_check = 0

if debug is True:
    if model_3D:
        test_dataset_debug = utilities.TFR_3D_sparse_dataset(test_img,
                    dataset_type = 'test',
                    batch_size=50,
                    buffer_size=100,
                    crop_size=crop_size)
        x, y = next(iter(test_dataset_debug))
        aus_y = y
        if show == True:
            y = utilities_models_tf.fix_labels_v2(y.numpy(),
                        classification_type='c3',
                        unique_labels=config['unique_labels'],
                        categorical=False)
            sample = (x.numpy(), y.numpy())
            utilities.plot_sparse_volume(sample, which_sample=10)

            utilities.plot_histogram_sparse_volume(sample,which_sample=10)

    else:
        test_dataset_debug = utilities.TFR_2D_dataset(test_img,
                    dataset_type = 'train',
                    batch_size=batch_size,
                    buffer_size=5000,
                    crop_size=crop_size)
        x, y = next(iter(test_dataset_debug))
        aus_y = y
        if show == True:
            y = utilities_models_tf.fix_labels_v2(y.numpy(),
                        classification_type='c3',
                        unique_labels=config['unique_labels'],
                        categorical=False)
            sample = (x.numpy(), y.numpy())
            utilities.show_batch_2D(sample)

# # # # create generator based on model specifications

if set_to_use == "test":
    img = test_img
elif set_to_use == "train":
    img = tr_img
elif set_to_use == "validation":
    img = val_img

if model_3D:
    test_dataset = utilities.TFR_3D_sparse_dataset(img,
                dataset_type = 'train',
                batch_size=1,
                buffer_size=5000,
                crop_size=crop_size)
else:
    test_dataset = utilities.TFR_2D_dataset(img,
                dataset_type = 'train',
                batch_size=batch_size,
                buffer_size=10000,
                crop_size=crop_size)

## 3 load model
# load model

if os.path.exists(os.path.join(model_path, 'fold_' + str(fold+1),model_name_version)):
    model = tf.keras.models.load_model(os.path.join(model_path, 'fold_' + str(fold+1), model_name_version), compile=False)
else:
    raise Exception('Model not found')

debug = True
if debug is True:
    for layer in reversed(model.layers):
        print(layer.name, layer.output_shape)

# deactivate augmentation layer if present
# find index of Augmentation layer
idx = [i for i,l in enumerate(model.layers) if l.name == "Augmentation"]
if idx:
    idx = idx[0]
    model.layers[idx].layers[0].horizontal = False
    model.layers[idx].layers[0].vertical = False
    model.layers[idx].layers[1].height_factor = 0
    model.layers[idx].layers[1].height_lower = 0
    model.layers[idx].layers[1].height_upper = 0
    model.layers[idx].layers[2].factor = 0
    model.layers[idx].layers[2].lower = 0
    model.layers[idx].layers[2].upper = 0

##  compute model predicion on images until reached the number to show
images = []
pred_logits = tf.zeros([0, len(config['unique_labels'])])
labels = tf.zeros([0, len(config['unique_labels'])])

for x, y in test_dataset:
    # get prediction
    y_logits = model(x, training=False)
    # save images for later
    images.extend(x.numpy())
    # save predictions taking into consideration VAE models
    if type(y_logits) is list:
        # the model is a VEA, taking only the prediction
        y_logits = y_logits[0]
    pred_logits = tf.concat([pred_logits, y_logits], axis=0)
    # save gt
    labels = tf.concat([labels, utilities_models_tf.fix_labels_v2(y.numpy(),
                config['classification_type'],
                config['unique_labels'])],
                axis=0)

    # stop is reached the number of images to show
    if pred_logits.numpy().shape[0] >= images_to_show:
        break

images = np.array(images)

# # # # # # augment image in case
# images = utilities_models_tf.augmentor(images)

labels = np.argmax(labels.numpy(), axis=-1)
# pred_logits = np.argmax(pred_logits.numpy(), axis=-1)

## 4.1 - Plot activation for the predicted class for consecutive layers (from shallow to deep)
importlib.reload(utilities)
debug = False

#save image or not
save = True
heatmap_raw = []
heatmap_rgb = []

# set-up number of images to plot and number of classes to show
n_samples_per_image = 5
n_images = images.shape[0] // n_samples_per_image
if images.shape[0] % n_samples_per_image != 0:
    n_images += 1

if debug is True:
    n_images = 1

if "ViT" in config["model_configuration"]:
    name_layers = []
    # get the last normalization layer
    for layer in model.layers:
        if "layer_normalization" in layer.name:
            name_layers.append(layer.name)
else:
    name_layers = []
    if model_3D:
        print('Looking for 2D conv layers...')
        for layer in model.layers:
            if 'conv' in layer.name:
                # in VAE there are transpose convolutions, we don't need those
                if 'transpose' not in layer.name:
                    # since ResNet layers are all named convX_blockY_type(conv, bn, relu)
                    # here we check that we are only taking the actuall conv layers
                    if 'block' in layer.name:
                        # here we are looking at a block of layers, only take the conv one
                        if layer.name[-4::]=='conv':
                            name_layers.append(layer.name)
                    elif 'conv3d' in layer.name:
                        # here no conv blocks
                        name_layers.append(layer.name)
    else:
        print('Looking for 2D conv layers...')
        for layer in model.layers:
            if 'conv' in layer.name:
                # in VAE there are transpose convolutions, we don't need those
                if 'transpose' not in layer.name:
                    # since ResNet layers are all named convX_blockY_type(conv, bn, relu)
                    # here we check that we are only taking the actuall conv layers
                    if 'block' in layer.name:
                        # here we are looking at a block of layers, only take the conv one
                        if layer.name[-4::]=='conv':
                            name_layers.append(layer.name)
                    elif 'conv2d' in layer.name:
                        # here no conv blocks
                        name_layers.append(layer.name)

print('Found {} layers -> {}'.format(len(name_layers), name_layers))

## Whick layer to show. if None, all will be displayed. This is useful when
# looking at large models with many layers
importlib.reload(utilities)
conv_to_show = None
# conv_to_show = [-4,-3,-2,-1]

if conv_to_show is not None:
    name_layers = [name_layers[i] for i in conv_to_show]
else:
    conv_to_show = [(i+1) for i, _ in enumerate(name_layers)]


# compute activation maps for each image and each network layer
for i in range(images.shape[0]):
    print('Computing activation maps for each layer for the predicted class: {}/{} \r'.format(i, images.shape[0]), end='')
    image = np.expand_dims(images[i], axis=0)
    c = np.argmax(pred_logits[i])
    # for all the images, compute heatmap for all the layers
    heatmap_raw.append([])
    heatmap_rgb.append([])
    for nl in name_layers:
        if "ViT" in config["model_configuration"]:
            cam = utilities.gradCAM(model, c, layerName=nl, ViT=True, is_3D=model_3D,  debug=False)
        else:
            cam = utilities.gradCAM(model, c, layerName=nl, is_3D=model_3D, debug=False)
        aus_raw, aus_rgb = cam.compute_heatmap(image)
        heatmap_raw[i].append(aus_raw)
        heatmap_rgb[i].append(aus_rgb)
print('Done.')

# start plotting
for i in range(n_images):
    print('Creating figure {:3}/{:3} \r'.format(i+1, n_images), end='')

    # greate figure
    # set different axis aspect ratios. The last axes is for the heat map -> smaller axes
    aus = [1 for i in range(len(name_layers) + 2)]
    aus[-1] = 0.1
    gridspec = {'width_ratios': aus}
    fig, axes = plt.subplots(nrows=n_samples_per_image, ncols=len(name_layers) + 2, figsize=(15,n_samples_per_image*2), gridspec_kw=gridspec)
    fig.suptitle('Consecutive activation maps for the predicted class \n {}'.format(class_description), fontsize=16)

    # fill in all axes
    for j in range(n_samples_per_image):
        idx = i*n_samples_per_image + j

        if idx >= images.shape[0]:
            break

        # original image
        if model_3D:
            # plot the middle images
            original_image = np.squeeze(images[idx])[:,:,images.shape[3]//2]
        else:
            original_image = np.squeeze(images[idx,:,:,0])

        axes[j, 0].imshow(original_image, cmap='gray', interpolation=None)
        if labels[idx] == np.argmax(pred_logits[idx]):
            axes[j, 0].set_title('gt {} - pred {}'.format(labels[idx],np.argmax(pred_logits[idx])), color='g')
        else:
            axes[j, 0].set_title('gt {} - pred {}'.format(labels[idx],np.argmax(pred_logits[idx])), color='r')

        axes[j, 0].set_xticks([])
        axes[j, 0].set_yticks([])

        # layer heatmaps
        for idx1, nl in enumerate(name_layers):
            im = axes[j, idx1+1].imshow(heatmap_raw[idx][idx1]/255, cmap='jet', vmin=0, vmax=1, interpolation=None)
            axes[j, idx1+1].set_title('layer {}'.format(conv_to_show[idx1]))
            axes[j, idx1+1].set_xticks([])
            axes[j, idx1+1].set_yticks([])

        # add colorbar
        cax = axes[j, -1]
        plt.colorbar(im, cax=cax)
        plt.tight_layout()


    if save is True:
        # fig.savefig(os.path.join(save_path, 'fold_'+str(fold)+'_activationMap_forConsecutiveLayers_%03d.pdf' % i), bbox_inches='tight', dpi = 100)
        fig.savefig(os.path.join(save_path, 'fold_'+str(fold)+'_activationMap_forConsecutiveLayers_%03d.png' % i), bbox_inches='tight', dpi = 100)
        plt.close(fig)
    else:
        # draw the image
        plt.draw()

if save is not True:
    plt.show()

## 4.2 across class plots
'''
Here for every image we plot the activation map for every class.
'''
save = True

debug = False

# get all the needed heatmaps
layer_name = name_layers[-1]
heatmap_raw = []
heatmap_rgb = []


for i in range(images.shape[0]):
    print(f'Computing activation maps for each class {i:3d}/{images.shape[0]:3d} \r', end='')
    image = np.expand_dims(images[i], axis=0)
    # for all the images, compute heatmap for all the classes
    heatmap_raw.append([])
    heatmap_rgb.append([])
    for j in range(len(config["unique_labels"])):
        if "ViT" in config["model_configuration"]:
            cam = utilities.gradCAM(model, j, layerName=nl, ViT=True, is_3D=model_3D, debug=False)
        else:
            cam = utilities.gradCAM(model, j, layerName=nl, is_3D=model_3D, debug=False)
        aus_raw, aus_rgb = cam.compute_heatmap(image)
        heatmap_raw[i].append(aus_raw)
        heatmap_rgb[i].append(aus_rgb)
print('Done.')

# set-up number of images to plot and number of classes to show
n_samples_per_image = 5
classes_to_show = [i for i in range(len(config["unique_labels"]))]
if not classes_to_show:
    classes_to_show = [x for x in range(len(unique_labels))]

n_images = images.shape[0] // n_samples_per_image
if images.shape[0] % n_samples_per_image != 0:
    n_images += 1

if debug is True:
    n_images = 1

print('Plotting activation maps for classes {}'.format(classes_to_show))

# start plotting
for i in range(n_images):
    print('Creating figure {:3}/{:3} \r'.format(i+1, n_images), end='')

    # set different axis aspect ratios. The last axes is for the heat map -> smaller axes
    aus = [1 for i in range(len(classes_to_show) + 2)]
    aus[-1] = 0.1
    gridspec = {'width_ratios': aus}
    fig, axes = plt.subplots(nrows=n_samples_per_image, ncols=len(classes_to_show) + 2, figsize=(15,n_samples_per_image*2),gridspec_kw=gridspec)
    fig.suptitle('Activation maps for layer {} \n {}'.format(layer_name, class_description), fontsize=15)

    # fill in all axes
    for j in range(n_samples_per_image):
        idx = i*n_samples_per_image + j

        if idx >= images.shape[0]:
            break

        # original image
        if model_3D:
            # plot the middle images
            original_image = np.squeeze(images[idx])[:,:,images.shape[3]//2]
        else:
            original_image = np.squeeze(images[idx,:,:,0])

        axes[j, 0].imshow(original_image, cmap='gray', interpolation=None)

        if labels[idx] == np.argmax(pred_logits[idx]):
            axes[j, 0].set_title('gt {} - pred {}'.format(labels[idx],np.argmax(pred_logits[idx])), color='g')
        else:
            axes[j, 0].set_title('gt {} - pred {}'.format(labels[idx],np.argmax(pred_logits[idx])), color='r')

        axes[j, 0].set_xticks([])
        axes[j, 0].set_yticks([])

        # class heatmaps
        for idx1, c in enumerate(classes_to_show):
            im = axes[j, idx1+1].imshow(heatmap_raw[idx][c]/255, cmap='jet', vmin=0, vmax=1, interpolation=None)
            if c == np.argmax(pred_logits[idx]):
                axes[j, idx1+1].set_title('class_{} - prob. {:05.3f}'.format(c, pred_logits[idx, c]), fontweight='bold')
            else:
                axes[j, idx1+1].set_title('class_{} - prob. {:05.3f}'.format(c, pred_logits[idx, c]))
            axes[j, idx1+1].set_xticks([])
            axes[j, idx1+1].set_yticks([])

        # add colorbar
        cax = axes[j, -1]
        plt.colorbar(im, cax=cax)
        plt.tight_layout()

    if save is True:
        # fig.savefig(os.path.join(save_path, 'fold_'+str(fold)+'_activationMap_'+layer_name+'_%03d.pdf' % i), bbox_inches='tight', dpi = 100)
        fig.savefig(os.path.join(save_path, 'fold_'+str(fold)+'_activationMap_'+layer_name+'_%03d.png' % i), bbox_inches='tight', dpi = 100)
        plt.close(fig)
    else:
        # draw the image
        plt.draw()

if save is not True:
    plt.show()

## 4.3 single image gradCAM
# importlib.reload(GradCAM)
#
# # select one image and one label
# idx = 3
# image = np.expand_dims(images[idx], axis=0)
# label = labels[idx].numpy()
# pred = np.argmax(pred_logits[idx])
#
#
# # initialize our gradient class activation map and build the heatmap
# cam = GradCAM.gradCAM(model, pred, layerName='conv2d_11', debug=True)
# heatmap_raw, heatmap_rgb = cam.compute_heatmap(image)
#
# # Plotting
# fig  = plt.figure(figsize=(10,20))
# ax1 = plt.subplot(2,2,1)
# ax1.imshow(np.squeeze(image), cmap='gray', interpolation=None)
# ax1.set_title('Original Image - gt {} - pred {}'.format(label,pred))
#
# ax2 = plt.subplot(2,2,2)
# # im = ax2.imshow(heatmap_rgb/255, cmap='jet', vmin=0, vmax=1, interpolation=None)
# im = ax2.imshow(heatmap_raw/255, cmap='jet', vmin=0, vmax=1, interpolation=None)
# ax2.set_title('Activation map from last model layer')
# plt.colorbar(im, ax=ax2)
#
#
# ax3 = plt.subplot(2,1,2)
# im = ax3.imshow(np.squeeze(image), cmap='gray', interpolation=None)
# im = ax3.imshow(heatmap_rgb/255, cmap='jet', vmin=0, vmax=1, interpolation=None, alpha=0.2)
# plt.colorbar(im, ax=ax3)
#
# plt.show()

# ## 4.4 - PLOT VAE ENCODED DISCTIBUTION
# from mpl_toolkits.mplot3d import Axes3D
# def plot_label_clusters(vae, data, labels):
#     # display a 2D plot of the digit classes in the latent space
#     z_mean, _, _, _, _, _ = model(data)
#     fig = plt.figure(figsize=(12, 10))
#     ax = Axes3D(fig)
#     ax.scatter(z_mean[:, 15], z_mean[:, 65],z_mean[:, 120], c=labels)
#     ax.set_xlabel("z[0]")
#     ax.set_ylabel("z[1]")
#     plt.show()
#
# if 'VAE' in model_name:
#     plot_label_clusters(model, images.numpy(), labels.numpy().tolist())



