'''
Script that tests a trained models on the training dataset. It returns:
¤ overall accuracy of each model
¤ accuracy for each independend class
¤ confusion matrix for each model
¤ ROC curve that compares the models
¤ (ROC curve that compars the folds of each model)

Steps
1 - get paths and models to test
2 - load testing dataset
3 - run throught the cross-validation folds and compute predictions for all images
4 - plot and save confusion matrix
5 - plot and save ROC curve
'''

import os
import sys
import cv2
import glob
import json
import time
import pickle
import random
import pathlib
import argparse
import importlib
import numpy as np
from datetime import datetime
from collections import OrderedDict
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from imgaug import augmenters as iaa
import imgaug as ia

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

# local imports
import utilities

## 1 - get models information and additional files
base_folder = '/flush/iulta54/Research/P3-THR_DL/trained_models'
# dataset_path = '/flush/iulta54/Research/Data/OCT/Thyroid_2019/2D_classification_dataset_per_class_organization/Test'
# dataset_path = '/flush/iulta54/Research/Data/OCT/Thyroid_2019/2D_classification_dataset_LeaveOneOut_per_class_organization/Test'

dataset_path = {
    'pytorch_gen':'/flush/iulta54/Research/Data/OCT/Thyroid/2D_classification_dataset_LeaveOneOut_anisotropic_per_class_organization/Test',
    'tf_gen': '/flush/iulta54/Research/Data/OCT/Thyroid/2D_classification_dataset_LeaveOneOut_anisotropic_per_class_organization_TFR/Test'}

# here specify which model or models to test

# ¤¤¤¤¤¤¤¤¤¤¤¤¤¤ NORMAL vs DISEASE
# models = ['M1_1_LeaveOneOut', 'M2_1_LeaveOneOut','M3_1_LeaveOneOut']
# model_descriptions = ['LightOCT', '3_layer_model','2_layer_model_convconv']

# ¤¤¤¤¤¤¤¤¤¤¤¤¤¤ NORMAL vs ENLARGED vs DEPLETED
# models = ['M1_2_LeaveOneOut', 'M2_2_LeaveOneOut','M3_2_LeaveOneOut']
# model_descriptions = ['LightOCT', '3_layer_model','2_layer_model_convconv']

# ¤¤¤¤¤¤¤¤¤¤¤¤¤¤ NORMAL vs ALL DISEASES
# models = ['M1_3_LeaveOneOut', 'M2_3_LeaveOneOut','M3_3_LeaveOneOut']
# model_descriptions = ['LightOCT', '3_layer_model','2_layer_model_convconv']

# ¤¤¤¤¤¤¤¤¤¤¤¤¤¤ ResNet
# models = ['ResNet50_1_LeaveOneOut_anisotropic']
# model_description = ['ResNet50']

# ¤¤¤¤¤¤¤¤¤¤¤¤¤ weighted loss
# model = ['M1_2_LeaveOneOut_anisotropic_wcce', 'M2_2_LeaveOneOut_anisotropic_wcce', 'M3_2_LeaveOneOut_anisotropic_wcce']
# model_description = ['M1', 'M2', 'M3']

# ¤¤¤¤¤¤¤¤¤¤¤¤¤ Variational autoencoders
# model = ['VAE_3_TFR']
# model_description = ['VAE']

# ¤¤¤¤¤¤¤¤¤¤¤¤¤ Variational autoencoders
# model = ['VAE_3_TFR']
# model_description = ['VAE']

# ¤¤¤¤¤¤¤¤¤¤¤¤¤
model = ['M1_3_LeaveOneOut_anisotropic_wcce_TFR', 'M2_3_LeaveOneOut_anisotropic_wcce_TFR', 'M3_3_LeaveOneOut_anisotropic_wcce_TFR', 'VAE_3_TFR']
model_description = ['M1', 'M2', 'M3', 'M4']



## 2 - load testing dataset
importlib.reload(utilities)

# given that now models have been trained using both pytorch and tf data generators,
# every model will have to have its own list of test files

models = []

for m in range(len(model)):
    # open dataset information
    with open(os.path.join(base_folder, model[m],'train_val_test_filenames_json.txt')) as json_file:
        data = json.load(json_file)
        image_files = data['Testing']
        num_folds = len(data['Training'])
        unique_labels = data['unique_labels']
        try:
            class_labels = data['class_labels']
        except:
            class_labels = range(len(unique_labels))

    # infere generator type by looking at the file extention
    if pathlib.Path(image_files[0]).suffix == '.gz':
        # this model has been trained using a pythorch generator
        gen_type = 'pytorch_gen'
    elif pathlib.Path(image_files[0]).suffix == '.tfrecords':
        # this model has been trained using a tf generator
        gen_type = 'tf_gen'

    # fix name of image_files
    if len(image_files[0]) > 50:
        # filenames contain the full path. Change to load the data from the dataset_path
        image_files = [os.path.join(dataset_path[gen_type], os.path.basename(os.path.dirname(x)),os.path.basename(x)) for x in image_files]
    else:
        if pathlib.Path(image_files[0]).suffix == '.gz':
            image_files = [os.path.join(dataset_path[gen_type], 'class_'+ x[-8], x) for x in image_files]
        elif pathlib.Path(image_files[0]).suffix == '.tfrecords':
            image_files = [os.path.join(dataset_path[gen_type], 'class_'+ x[-11], x) for x in image_files]

    # save infomation in dictionary for this model
    aus = {}
    aus['model'] = model[m]
    aus['description'] = model_description[m]
    aus['test_files'] = image_files
    aus['num_folds'] = num_folds
    aus['unique_labels'] = unique_labels
    aus['gen_type'] = gen_type

    models.append(aus)

# general generator imformation
seed = 29
crop_size = (250,250) # (h, w)
batch_size = 64

seq = iaa.Sequential([
    iaa.Resize({'height': crop_size[0], 'width': crop_size[1]})
], random_order=False) # apply augmenters in random order

transformer = transforms.Compose([
    utilities.ChannelFix(channel_order='last')
    ])

## check data coming out of the generators
importlib.reload(utilities)
debug = True
show = True
model_to_check = 0

if debug is True:
    # check what type of generator the model needs and build it
    if models[model_to_check]['gen_type'] == 'pytorch_gen':
        # build pytorch generator
        transformer_d = transforms.Compose([
            utilities.ChannelFix(channel_order='first')
            ])
        test_dataset_debug = utilities.OCTDataset2D_classification(models[model_to_check]['test_files'],
                    models[model_to_check]['unique_labels'],
                    transform=transformer_d,
                    augmentor=seq)
        test_dataset_debug = DataLoader(test_dataset_debug, batch_size=18,
                                shuffle=True, num_workers=0, pin_memory=True)
        sample = next(iter(test_dataset_debug))

        # show images if needed
        if show == True:
            utilities.show_batch_2D(sample)
    if models[model_to_check]['gen_type'] == 'tf_gen':
        # build tf generator
        test_dataset_debug = utilities.TFR_2D_dataset(models[model_to_check]['test_files'],
                        dataset_type = 'train',
                        batch_size=18,
                        buffer_size=1000,
                        crop_size=(crop_size[0], crop_size[1]))
        x, y = next(iter(test_dataset_debug))
        if show == True:
            sample = (x.numpy(), y.numpy())
            utilities.show_batch_2D(sample)

## 3 - run throught the cross-validation folds and compute predictions for all images
importlib.reload(utilities)

# check if test_summary_file is present in the model folder. If not, perform testing
# else open the file and store it in the prediction_database variable

'''
VERY IMPORTANT!
During testing, is important that all the folds test on the same images and images
should be alligned - by using shuffle, the iterator will produce a new sequence
of images for every fold. This does not allow for the folds to be ensambled
since, even if each fold is tested on the same images, these are not in the same order.
'''

# where to save all the predictions: model, fold, predictions
prediction_database = []

for m in range(len(models)):
    print('Working on model {}'.format(models[m]['model']))
    prediction_database.append([])
    tic = time.time()

    # check if test_summary.txt file exists in the model folder
    if os.path.isfile(os.path.join(base_folder, models[m]['model'], 'test_summary.txt')):
        print(' - test_summary file found. Using data from previous testing.')
        # open the test_summary file and store information in the predicted_database variable
        with open(os.path.join(base_folder, models[m]['model'], 'test_summary.txt')) as json_file:
            data = json.load(json_file)
            try:
                folds_test_values = data['folds_test_values']
            except:
                folds_test_values = data['folds_test_logits_values']
            gt = data['labels']
            test_time = data['test_time']
            # save in predicted dataset variable
            # prediction_database[m].append(folds_test_values)
            prediction_database[m] = folds_test_values

        print(' - classification of {} images took {} for this model'.format(len(folds_test_values[0]), test_time))
    else:
        print('test_summary file not found. Running test now...')
        aus_prediction_database = []

        # create generator based on model specifications
        if models[m]['gen_type'] == 'pytorch_gen':
            test_dataset = utilities.OCTDataset2D_classification(models[m]['test_files'],
                    models[m]['unique_labels'],
                    transform=transformer,
                    augmentor=seq)
            test_dataset = DataLoader(test_dataset, batch_size=batch_size,
                                    shuffle=False, num_workers=0, pin_memory=True)
        elif models[m]['gen_type'] == 'tf_gen':
            test_dataset = utilities.TFR_2D_dataset(models[m]['test_files'],
                            dataset_type = 'test',
                            batch_size=batch_size,
                            buffer_size=1000,
                            crop_size=(crop_size[0], crop_size[1]))

        for cv in range(models[m]['num_folds']):
            print(' - runing predictions on fold {}/{}'.format(cv+1, num_folds))
            # load model
            model_path = os.path.join(base_folder, models[m]['model'], 'fold_'+ str(cv+1), 'model')
            if os.path.exists(model_path + '.h5'):
                model_path = model_path + '.h5'
            elif os.path.exists(model_path + '.tf'):
                model_path = model_path + '.tf'
            else:
                raise Exception('Model not found')

            model = load_model(model_path, compile=False)
            # run through all the images in the test dataset
            aus_pred = []
            aus_gt = []
            for im, label in test_dataset:
                im = im.numpy()
                pred = model(im)
                if type(pred) is list:
                    # the model is a VEA, taking only the prediction
                    pred = pred[4].numpy().tolist()
                else:
                    pred = pred.numpy().tolist()

                if models[m]['gen_type'] == 'tf_gen':
                    # need to fix the labels here based on the unique label information
                    label = utilities.fix_labels(label.numpy(),
                                            models[m]['unique_labels'],
                                            categorical=False)

                aus_pred.extend(pred)
                aus_gt.extend(label)
            # save predictions and gts
            prediction_database[m].append(np.array(aus_pred))
            aus_prediction_database.append(aus_pred)

            if m == 0:
                gt = np.array(aus_gt)
        # print classification time for the model
        toc = time.time()
        print(' - classification of {} images took {} for this model'.format(len(image_files), utilities.tictoc(tic,toc)))

        # save test summary
        print(' - Saving test summary...')

        json_dict = OrderedDict()
        json_dict['model_name'] = models[m]['model']
        json_dict['labels'] = [int(i) for i in aus_gt]
        json_dict['folds_test_values'] = aus_prediction_database
        json_dict['test_time'] = utilities.tictoc(tic,toc)
        json_dict['test_date'] = time.strftime("%Y%m%d-%H%M%S")
        with open(os.path.join(base_folder, models[m]['model'],'test_summary.txt'), 'w') as fp:
            json.dump(json_dict, fp)

        # delete some variables
        del test_dataset
        del model

gt_backup = gt

## 4 - plot and save confusion matrix
importlib.reload(utilities)
ensemble_pred_argmax = []
ensemble_pred_logits = []

debug = False
dummy = False

for idx, m in enumerate(models):
    # compute ensemble
    # get out values for the model
    aus_ensemble_pred = np.array(prediction_database[idx])
    # compute the logits mean along the folds
    aus_ensemble_pred = aus_ensemble_pred.mean(axis=0)
    # compute argmax prediction
    ensemble_pred_logits.append(aus_ensemble_pred)
    aus_ensemble_pred = np.argmax(aus_ensemble_pred, axis=1)
    ensemble_pred_argmax.append(aus_ensemble_pred)

    # path where to save model
    save_path = os.path.join(base_folder, m['model'])

    if debug is True:
        if dummy is True:
            print('Using dummy arrays')
            gt = a
            aus_ensemble_pred = b

        for c in range(len(unique_labels)):
            # compute tp and fp for each class
            # tp = np.sum(np.multiply((gt == c), (ensemble_pred_argmax[idx] == c)))
            # tn = np.sum(np.multiply((gt != c), (ensemble_pred_argmax[idx] != c)))
            #
            # fp = np.sum(np.multiply((gt != c), (ensemble_pred_argmax[idx] == c)))
            # fn = np.sum(np.multiply((gt == c), (ensemble_pred_argmax[idx] != c)))
            tp = np.sum(np.multiply((gt == c), (aus_ensemble_pred == c)))
            tn = np.sum(np.multiply((gt != c), (aus_ensemble_pred!= c)))

            fp = np.sum(np.multiply((gt != c), (aus_ensemble_pred == c)))
            fn = np.sum(np.multiply((gt == c), (aus_ensemble_pred != c)))
            print('class {} #sample -> {}: tp {:03}, fp {:03}, tn {:03}, fn {:03}'.format(c, np.sum(gt==c), int(tp), int(fp), int(tn), int(fn)))


    # save confusion matrix
    acc = utilities.plotConfusionMatrix(gt, aus_ensemble_pred, classes=class_labels, savePath=save_path, draw=True)
    # print accuracy for reference
    print('Model {} -> accuracy {:02.4}'.format(m['model'], acc))
plt.show()

ensemble_pred_argmax = np.array(ensemble_pred_argmax)
ensemble_pred_logits = np.array(ensemble_pred_logits)


## 5 - plot and save ROC curves
'''
Check this link for better understanding of micro and macro-averages
https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin

Here computing both the macro-average ROC and the micro-average ROC.
Using code from https://scikit-learn.org/dev/auto_examples/model_selection/plot_roc.html with modification

Saving independent ROCs for each model and the comparison between them (micro and marco average)
'''
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from scipy import interp

save = True

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

gt_categorical = to_categorical(gt, num_classes=len(unique_labels))
n_classes = len(unique_labels)
lw = 2

for idx, m1 in enumerate(models):
    m = m1['model']
    fpr[m] = dict()
    tpr[m] = dict()
    roc_auc[m] = dict()

    pred_logits = ensemble_pred_logits[idx]


    # ¤¤¤¤¤¤¤¤¤¤¤ micro-average roc
    for i in range(n_classes):
        fpr[m][i], tpr[m][i], _ = roc_curve(gt_categorical[:,i], pred_logits[:,i])
        roc_auc[m][i] = auc(fpr[m][i], tpr[m][i])


    # Compute micro-average ROC curve and ROC area
    fpr[m]["micro"], tpr[m]["micro"], _ = roc_curve(gt_categorical.ravel(), pred_logits.ravel())
    roc_auc[m]["micro"] = auc(fpr[m]["micro"], tpr[m]["micro"])

    # ¤¤¤¤¤¤¤¤¤¤ macro-average roc

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[m][i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[m][i], tpr[m][i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr[m]["macro"] = all_fpr
    tpr[m]["macro"] = mean_tpr
    roc_auc[m]["macro"] = auc(fpr[m]["macro"], tpr[m]["macro"])

    # Plot all ROC curves for this model and save
    fig, ax = plt.subplots(figsize=(10,10))
    ax.plot(fpr[m]["micro"], tpr[m]["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc[m]["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    ax.plot(fpr[m]["macro"], tpr[m]["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc[m]["macro"]),
            color='navy', linestyle=':', linewidth=4)

    colors = cycle(['blue', 'orange', 'green', 'red','purple','brown','pink','gray','olive','cyan','teal'])
    for i, color in zip(range(n_classes), colors):
        ax.plot(fpr[m][i], tpr[m][i], color=color, lw=lw,
                label='ROC curve of class {} (area = {:0.2f})'
                ''.format(class_labels[i], roc_auc[m][i]))

    ax.plot([0, 1], [0, 1], 'k--', lw=lw)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate', fontsize=25)
    ax.set_ylabel('True Positive Rate', fontsize=25)
    ax.set_title('Model {} - multi-class ROC (OneVsAll)'.format(m), fontsize=20)
    plt.legend(loc="lower right", fontsize=12)

    # ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤ work on the zummed-in image
    colors = cycle(['blue', 'orange', 'green', 'red','purple','brown','pink','gray','olive','cyan','teal'])
    axins = zoomed_inset_axes(ax, zoom=1, loc=7, bbox_to_anchor=(0,0,0.99,0.9), bbox_transform=ax.transAxes)

    axins.plot(fpr[m]["micro"], tpr[m]["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc[m]["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    axins.plot(fpr[m]["macro"], tpr[m]["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc[m]["macro"]),
            color='navy', linestyle=':', linewidth=4)

    for i, color in zip(range(n_classes), colors):
        axins.plot(fpr[m][i], tpr[m][i], color=color, lw=lw,
                label='ROC curve of class {} (area = {:0.2f})'
                ''.format(class_labels[i], roc_auc[m][i]))

        # sub region of the original image
        x1, x2, y1, y2 = 0.0, 0.3, 0.7, 1.0
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)

        axins.set_xticks(np.linspace(x1, x2, 4))
        axins.set_yticks(np.linspace(y1, y2, 4))

    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    mark_inset(ax, axins, loc1=1, loc2=3, fc='none', ec='0.5', ls='--')


    plt.draw()

    if save is True:
        savePath = os.path.join(base_folder, m)
        fig.savefig(os.path.join(savePath, 'Multiclass ROC.pdf'), bbox_inches='tight', dpi = 100)
        fig.savefig(os.path.join(savePath, 'Multiclass ROC.png'), bbox_inches='tight', dpi = 100)
        plt.close()
    else:
        plt.show()

## 6 - plot and save Precision-Recall (PR) curves
'''
Check this link for better understanding of micro and macro-averages
https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin

Here computing both the macro-average PR and the micro-average PR.
Using code from https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html with modification

'''
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from itertools import cycle
from scipy import interp

save = True
#line width
lw = 2

# Compute precision and recall for each class
precision = dict()
recall = dict()
average_precision = dict()

# transform labels to categorical
gt_categorical = to_categorical(gt, num_classes=len(unique_labels))
n_classes = len(unique_labels)


for idx, m1 in enumerate(models):
    m = m1['model']
    precision[m] = dict()
    recall[m] = dict()
    average_precision[m] = dict()

    pred_logits = ensemble_pred_logits[idx]


    # ¤¤¤¤¤¤¤¤¤¤¤ micro-average roc
    for i in range(n_classes):
        precision[m][i], recall[m][i], _ = precision_recall_curve(gt_categorical[:,i], pred_logits[:,i])
        average_precision[m][i] = average_precision_score(gt_categorical[:,i], pred_logits[:,i])


    # Compute micro-average ROC curve and ROC area
    precision[m]["micro"], recall[m]["micro"], _ = precision_recall_curve(gt_categorical.ravel(), pred_logits.ravel())
    average_precision[m]["micro"] = average_precision_score(gt_categorical, pred_logits, average='micro')
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision[m]["micro"]))

    # Plot all PR curves for this model and save
    # create iso-f1 curves and plot on top the PR curves for every class
    fig, ax = plt.subplots(figsize=(10,10))

    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = ax.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        ax.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = ax.plot(recall[m]["micro"], precision[m]["micro"], color='gold', lw=lw,
                label='micro-average Precision-recall (area = {0:0.2f})'.format(average_precision[m]["micro"]))
    lines.append(l)
    # labels.append('micro-average Precision-recall (area = {0:0.2f})'
    #             ''.format(average_precision[m]["micro"]))

    colors = cycle(['blue', 'orange', 'green', 'red','purple','brown','pink','gray','olive','cyan','teal'])
    for i, color in zip(range(n_classes), colors):
        ax.plot(recall[m][i], precision[m][i], color=color, lw=lw,
                label='Precision-recall curve of class {:9s} (area = {:0.2f})'.format(class_labels[i], average_precision[m][i]))

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('True positive rate - Recall [TP/(TP+FN)]', fontsize=20)
    ax.set_ylabel('Positive predicted value - Precision [TP/(TP+TN)]', fontsize=20)
    ax.set_title('Model {} - multi-class Precision-recall curve'.format(m), fontsize=20)
    plt.legend(loc="lower right", fontsize=12)

    plt.draw()

    if save is True:
        savePath = os.path.join(base_folder, m)
        fig.savefig(os.path.join(savePath, 'Multiclass PR.pdf'), bbox_inches='tight', dpi = 100)
        fig.savefig(os.path.join(savePath, 'Multiclass PR.png'), bbox_inches='tight', dpi = 100)
        plt.close()
    else:
        plt.show()



##    # ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤ work on the zummed-in image
    colors = cycle(['blue', 'orange', 'green', 'red','purple','brown','pink','gray','olive','cyan','teal'])
    axins = zoomed_inset_axes(ax, zoom=1, loc=7, bbox_to_anchor=(0,0,0.99,0.9), bbox_transform=ax.transAxes)

    axins.plot(fpr[m]["micro"], tpr[m]["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc[m]["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    axins.plot(fpr[m]["macro"], tpr[m]["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc[m]["macro"]),
            color='navy', linestyle=':', linewidth=4)

    for i, color in zip(range(n_classes), colors):
        axins.plot(fpr[m][i], tpr[m][i], color=color, lw=lw,
                label='ROC curve of class {} (area = {:0.2f})'
                ''.format(class_labels[i], roc_auc[m][i]))

        # sub region of the original image
        x1, x2, y1, y2 = 0.0, 0.3, 0.7, 1.0
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)

        axins.set_xticks(np.linspace(x1, x2, 4))
        axins.set_yticks(np.linspace(y1, y2, 4))

    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    mark_inset(ax, axins, loc1=1, loc2=3, fc='none', ec='0.5', ls='--')


    plt.draw()

    if save is True:
        savePath = os.path.join(base_folder, m)
        fig.savefig(os.path.join(savePath, 'Multiclass ROC.pdf'), bbox_inches='tight', dpi = 100)
        fig.savefig(os.path.join(savePath, 'Multiclass ROC.png'), bbox_inches='tight', dpi = 100)
        plt.close()
    else:
        plt.show()

## plot comparison between models using the saved micro and macro-averages
save = True

lw = 2
colors = cycle(['blue', 'orange', 'green', 'red','purple','brown','pink','gray','olive','cyan','teal'])

# ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤ micro average
fig, ax = plt.subplots(figsize=(10,10))
for m, color in zip(range(len(models)), colors):
    ax.plot(fpr[models[m]['model']]['micro'], tpr[models[m]['model']]['micro'], color=color, lw=lw,
            label='Model {} (area = {:0.2f})'.format(models[m]['model'], roc_auc[models[m]['model']]['micro']))

ax.plot([0, 1], [0, 1], 'k--', lw=lw)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
# plt.grid('minor')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Comparison multi-class ROC (OneVsAll) - micro-average')
plt.legend(loc="lower right")

# ¤¤¤¤¤¤¤¤ work on the zoom-in
colors = cycle(['blue', 'orange', 'green', 'red','purple','brown','pink','gray','olive','cyan','teal'])
axins = zoomed_inset_axes(ax, zoom=1, loc=7, bbox_to_anchor=(0,0,0.99,0.9), bbox_transform=ax.transAxes)
for m, color in zip(range(len(models)), colors):
    axins.plot(fpr[models[m]['model']]['micro'], tpr[models[m]['model']]['micro'], color=color, lw=lw,
            label='Model {} (area = {:0.2f})'.format(models[m]['model'], roc_auc[models[m]['model']]['micro']))

# sub region of the original image
x1, x2, y1, y2 = 0.0, 0.3, 0.7, 1.0
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

axins.set_xticks(np.linspace(x1, x2, 4))
axins.set_yticks(np.linspace(y1, y2, 4))

# draw a bbox of the region of the inset axes in the parent axes and
# connecting lines between the bbox and the inset axes area
mark_inset(ax, axins, loc1=1, loc2=3, fc='none', ec='0.5', ls='--')

plt.draw()
if save is True:
    fig.savefig(os.path.join(base_folder, 'Model_comparison_multiclass_ROC_micro_avg_'+datetime.now().strftime("%H:%M:%S")+'.pdf'), bbox_inches='tight', dpi = 100)
    fig.savefig(os.path.join(base_folder, 'Model_comparison_multiclass_ROC_micro_avg_'+datetime.now().strftime("%H:%M:%S")+'.png'), bbox_inches='tight', dpi = 100)
    plt.close()

# ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤ macro average
colors = cycle(['blue', 'orange', 'green', 'red','purple','brown','pink','gray','olive','cyan','teal'])
line_style = cycle([':', '-.', '-'])
fig, ax = plt.subplots(figsize=(10,10))
for m, color, ls in zip(range(len(models)), colors, line_style):
    ax.plot(fpr[models[m]['model']]['macro'], tpr[models[m]['model']]['macro'], color=color, lw=lw, ls=ls,
            label='Model {} (area = {:0.2f})'.format(models[m]['model'], roc_auc[models[m]['model']]['macro']))

ax.plot([0, 1], [0, 1], 'k--', lw=lw)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
# plt.grid('minor')
ax.set_xlabel('False Positive Rate', fontsize=25)
ax.set_ylabel('True Positive Rate', fontsize=25)
ax.set_title('Comparison multi-class ROC (OneVsAll) - macro-average', fontsize=20)
plt.legend(loc="lower right", fontsize=12)

# ¤¤¤¤¤¤¤¤ work on the zoom-in
colors = cycle(['blue', 'orange', 'green', 'red','purple','brown','pink','gray','olive','cyan','teal'])
line_style = cycle([':', '-.', '-'])
axins = zoomed_inset_axes(ax, zoom=1, loc=7, bbox_to_anchor=(0,0,0.99,0.9), bbox_transform=ax.transAxes)
for m, color, ls in zip(range(len(models)), colors, line_style):
    axins.plot(fpr[models[m]['model']]['macro'], tpr[models[m]['model']]['macro'], color=color, lw=lw, ls=ls,
            label='Model {} (area = {:0.2f})'.format(models[m]['model'], roc_auc[models[m]['model']]['macro']))

# sub region of the original image
x1, x2, y1, y2 = 0.0, 0.3, 0.7, 1.0
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

axins.set_xticks(np.linspace(x1, x2, 4))
axins.set_yticks(np.linspace(y1, y2, 4))

# draw a bbox of the region of the inset axes in the parent axes and
# connecting lines between the bbox and the inset axes area
mark_inset(ax, axins, loc1=1, loc2=3, fc='none', ec='0.5', ls='--')

if save is True:
    fig.savefig(os.path.join(base_folder, 'Model_comparison_multiclass_ROC_macro_avg_'+datetime.now().strftime("%H:%M:%S")+'.pdf'), bbox_inches='tight', dpi = 100)
    fig.savefig(os.path.join(base_folder, 'Model_comparison_multiclass_ROC_macro_avg_'+datetime.now().strftime("%H:%M:%S")+'.png'), bbox_inches='tight', dpi = 100)
    plt.close()
else:
    plt.show()



