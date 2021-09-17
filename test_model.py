'''
Script that tests a trained models on its training dataset. It does the same
testing routine as the one in the overall run_training.py script.
It saves
¤ the information about the test for easy later plotting
¤ ROC (per-class and overall using micro and macro average)
¤ PP curve (per-class and overall using micro and macro average)
¤ summary of performance for easy read of the final scores

Steps
1 - get paths and models to test
2 - load testing dataset
3 - get predictions using the test function in the utilities_models_tf.py
4 - plot and save confusion matrix
5 - plot and save ROC curve
6 - save detailed info of the testing and summary
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

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

# local imports
import utilities
import utilities_models_tf

## 1 - get models information and additional files
model_path = '/flush/iulta54/Research/P3-THR_DL/trained_models/LigthOCT_c1_anisotropic_wa'
dataset_path = '/flush/iulta54/Research/Data/OCT/Thyroid_2019_refined_DeepLearning'

# check forlders
if not os.path.isdir(model_path):
    raise ValueError(f'Model not found. Given {model_path}')
else:
    # check that the configuration file is in place
    if not os.path.isfile(os.path.join(model_path,'config.json')):
        raise ValueError(f'Configuration file not found for the given model. Check that the model was configured and trained. Given {os.path.join(model_path,"config.json")}')

if not os.path.isdir(dataset_path):
    raise ValueError(f'Model not found. Given {dataset_path}')

## 2 - load testing dataset
importlib.reload(utilities)

# load configuration file
with open(os.path.join(model_path,'config.json')) as json_file:
    config = json.load(json_file)

    # take one testing
    # make sure that the files point to this system dataset
    test_img = [os.path.join(dataset_path, pathlib.Path(f).parts[-2], pathlib.Path(f).parts[-1]) for f in config['test']]

# create generator based on model specifications
test_dataset = utilities.TFR_2D_dataset(test_img,
                dataset_type = 'test',
                batch_size=100,
                buffer_size=1000,
                crop_size=config['input_size'])

## perform testing for each fold the model was trained on

test_fold_summary = {}
folds = glob.glob(os.path.join(model_path,"fold_*"))

for f in folds:
    # load model
    if os.path.exists(os.path.join(f, 'model.tf')):
        model = tf.keras.models.load_model(os.path.join(model_path, 'fold_' + str(fold+1), 'model.tf'), compile=False)
    else:
        raise Exception('Model not found')

    test_gt, test_prediction, test_time = utilities_models_tf.test(model, test_dataset)
    test_fold_summary[cv]={
            'ground_truth':np.argmax(test_gt.numpy(), axis=-1),
            'prediction':test_prediction.numpy(),
            'test_time':float(test_time)
            }

## save and plot
from collections import OrderedDict
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
'''
Saving overall cross validation test results and images:
- Confisuion matrix
- ROC curve
- Precision-Recall curve

- test summary file with the prediction for every test image (test_summary.txt)
    Here add also the information needed to re-plot the ROC and PP curves (fpr,
    tpr, roc_auc, precision and recall - micro and macro average)
    The test_summary.txt file is a dictionary with the following entries:
    - model_name: string
    - labels: list of the true values fot the tested images
    - fold_test_values: list containing the predictions for every fold (list of lists)

    - test_time: string
    - test date: string

    - accuracy: float

    - false_positive_rate: list containing the fpr for every class (list of lists)
    - false_positive_rate_micro_avg: list containing the micro average fpr (used for the overall roc plot)
    - false_positive_rate_macro_avg: list containing the macro average fpr (used for the overall roc plot)

    - true_positive_rate: list containing the tpr for every class (list of lists)
    - true_positive_rate_micro_avg: list containing the micro average tpr (used for the overall roc plot)
    - true_positive_rate_macro_avg: list containing the macro average tpr (used for the overall roc plot)

    - precision: list containing the precision values for every class to plot the PP (list of lists)
    - precision_micro_avg: list of overall micro average of precision
    - average_precision: average precision value computed using micro average

    - recall: list containing the recall value for every class to plot the PP (list of lists)
    - recall_micro_avg: list of overall micro average of recall

    - F1: list of micro and macro average f1-score

Since the full test_summary file is long to open, the scores are also saved in a separate file for easy access
scores_test_summary.txt
'''
# ############# save the information that is already available
test_summary = OrderedDict()

test_summary['model_name'] = config['model_save_name']
test_summary['labels'] = [int(i) for i in test_fold_summary[0]['ground_truth']]
test_summary['folds_test_logits_values'] = [test_fold_summary[cv]['prediction'].tolist() for cv in range(config['N_FOLDS'])]
test_summary['test_time'] = utilities.tictoc_from_time(np.sum([test_fold_summary[cv]['test_time'] for cv in range(config['N_FOLDS'])]))
test_summary['test_date'] = time.strftime("%Y%m%d-%H%M%S")

# ############ plot and save confucion matrix
ensemble_pred_argmax = []
ensemble_pred_logits = []
# compute ensemble
# compute the logits mean along the folds
ensemble_pred_logits = np.array(test_summary['folds_test_logits_values']).mean(axis=0)
# compute argmax prediction
ensemble_pred_argmax = np.argmax(ensemble_pred_logits, axis=1)

acc = utilities.plotConfusionMatrix(test_summary['labels'], ensemble_pred_argmax, classes=config['label_description'], savePath=config['save_model_path'], draw=False)

# ############ plot and save ROC curve
fpr, tpr, roc_auc = utilities.plotROC(test_summary['labels'], ensemble_pred_logits, classes=config['label_description'], savePath=config['save_model_path'], draw=False)
# make elements of the dictionary to be lists for saving
for key, value in fpr.items():
    fpr[key]=value.tolist()
for key, value in tpr.items():
    tpr[key]=value.tolist()
for key, value in roc_auc.items():
    roc_auc[key]=value.tolist()

# ############ plot and save ROC curve
precision, recall, average_precision, F1 = utilities.plotPR(test_summary['labels'], ensemble_pred_logits, classes=config['label_description'], savePath=config['save_model_path'], draw=False)
# make elements of the dictionary to be lists for saving
for key, value in precision.items():
    precision[key]=value.tolist()
for key, value in recall.items():
    recall[key]=value.tolist()

# save all the information in the test summary
test_summary['accuracy'] = acc

# test_summary['false_positive_rate'] = [fpr[i].tolist() for i in range(len(class_labels))]
test_summary['false_positive_rate'] = fpr
# test_summary['false_positive_rate_micro_avg'] = fpr['micro'].tolist()
# test_summary['false_positive_rate_macro_avg'] = fpr['macro'].tolist()

test_summary['true_positive_rate'] = tpr
# test_summary['true_positive_rate'] = [tpr[i].tolist() for i in range(len(class_labels))]
# test_summary['true_positive_rate_micro_avg'] = tpr['micro'].tolist()
# test_summary['true_positive_rate_macro_avg'] = tpr['macro'].tolist()

test_summary['roc_auc'] = roc_auc

test_summary['precision'] = precision
# test_summary['precision'] = [precision[i].tolist() for i in range(len(class_labels))]
# test_summary['precision_micro_avg'] = precision['micro'].tolist()

test_summary['recall'] = recall
# test_summary['recall'] = [recall[i].tolist() for i in range(len(class_labels))]
# test_summary['recall_micro_avg'] = recall['micro'].tolist()

test_summary['average_precision'] = average_precision
test_summary['F1'] = F1

# save summary file
with open(os.path.join(config['save_model_path'],'test_summary.txt'), 'w') as fp:
    json.dump(test_summary, fp)

## save summary (can be improved, but using the routine from print_model_performance)

labels = np.eye(np.unique(test_summary['labels']).shape[0])[test_summary['labels']]
pred_logits = test_summary['folds_test_logits_values']

# check if we are working on a binary or multi-class classification
binary_classification = True
if labels.shape[-1] > 2:
    binary_classification = False

'''
For binary classification we can compute all the metrics:
- Specificity
- Precision
- Recall (Sensitivity)
- F1-score
- AUC for ROC

For multi-class classification we can only compute:
- Precision
- Recall (Sensitivity)
- F1-score
- AUC for ROC
'''
# Computupe per fold performance
performance_fold = {
            'ROC_AUC':[],
            'Precision': [],
            'Recall':[],
            'F1':[]
                }
if binary_classification:
    performance_fold['Specificity'] = []

for f in range(n_folds):
    performance_fold["Precision"].append(average_precision_score(labels, pred_logits[f],
                                                    average="macro"))
    performance_fold["Recall"].append(recall_score(np.argmax(labels,-1), np.argmax(pred_logits[f],-1),
                                                    average="macro"))
    performance_fold['ROC_AUC'].append(roc_auc_score(labels, pred_logits[f], multi_class='ovr', average='macro'))
    performance_fold['F1'].append(f1_score(np.argmax(labels,-1), np.argmax(pred_logits[f],-1), average='macro'))

    if binary_classification:
        tn, fp, fn, tp = confusion_matrix(np.argmax(labels,-1), np.argmax(pred_logits[f], -1)).ravel()
        performance_fold['Specificity'].append(tn / (tn + fp))

# compute ensamble performance
# compute the logits mean along the folds
ensemble_pred_logits = np.array(pred_logits).mean(axis=0)
# compute argmax prediction
ensemble_pred_argmax = np.argmax(ensemble_pred_logits, axis=1)

performance_ensamble = {}

performance_ensamble["Precision"] = average_precision_score(labels, ensemble_pred_logits,
                                                average="macro")
performance_ensamble["Recall"] = recall_score(np.argmax(labels,-1), ensemble_pred_argmax,
                                                average="macro")
performance_ensamble['ROC_AUC'] = roc_auc_score(labels, ensemble_pred_logits, multi_class='ovr', average='macro')
performance_ensamble['F1'] = f1_score(np.argmax(labels,-1), ensemble_pred_argmax, average='macro')

if binary_classification:
    tn, fp, fn, tp = confusion_matrix(np.argmax(labels,-1), ensemble_pred_argmax).ravel()
    performance_ensamble['Specificity'] = tn / (tn + fp)

# ######################### printing on file

summary = open(os.path.join(model_path,"short_test_summary.txt"), 'w')

summary.write(f'\nModel Name: {os.path.basename(model_path)}\n')
# add test time overall and per image
average_test_time = utilities.tictoc_from_time(np.mean([test_fold_summary[cv]['test_time'] for cv in range(config['N_FOLDS'])]))
average_test_time_per_image = eaverate_test_time/labels.shape[0]
summary.write(f'Overall model test time (average over folds): {utilities.tictoc_from_time(average_test_time)}')
summary.write(f'Average test time per image (average over folds): {utilities.tictoc_from_time(average_test_time_per_image)}\n')
summary.write(f'{"¤"*21}')
summary.write(f'¤ Per-fold metrics ¤')
summary.write(f'{"¤"*21}\n')

if binary_classification:
    keys = ['Specificity','Recall','Precision', 'F1', 'ROC_AUC']

    summary.write(f'{"Fold":^7}{keys[0]:^11}{keys[1]:^11}{keys[2]:^11}{keys[3]:^11}{keys[4]:^11}')

    for i in range(n_folds):
        summary.write(f'{i+1:^7}{performance_fold[keys[0]][i]:^11.3f}{performance_fold[keys[1]][i]:^11.3f}{performance_fold[keys[2]][i]:^11.3f}{performance_fold[keys[3]][i]:^11.3f}{performance_fold[keys[-1]][i]:^11.3f}')
    summary.write(f'{"-"*60}')
    summary.write(f'{"Average":^7}{np.mean(performance_fold[keys[0]]):^11.3f}{np.mean(performance_fold[keys[1]]):^11.3f}{np.mean(performance_fold[keys[2]]):^11.3f}{np.mean(performance_fold[keys[3]]):^11.3f}{np.mean(performance_fold[keys[4]]):^11.3f}')
    summary.write(f'{"STD":^7}{np.std(performance_fold[keys[0]]):^11.3f}{np.std(performance_fold[keys[1]]):^11.3f}{np.std(performance_fold[keys[2]]):^11.3f}{np.std(performance_fold[keys[3]]):^11.3f}{np.std(performance_fold[keys[4]]):^11.3f}')

else:
    keys = ['Recall','Precision', 'F1', 'ROC_AUC']

    summary.write(f'{"Fold":^7}{keys[0]:^11}{keys[1]:^11}{keys[2]:^11}{keys[3]:^11}')

    for i in range(n_folds):
        summary.write(f'{i+1:^7}{performance_fold[keys[0]][i]:^11.3f}{performance_fold[keys[1]][i]:^11.3f}{performance_fold[keys[2]][i]:^11.3f}{performance_fold[keys[3]][i]:^11.3f}')
    summary.write(f'{"-"*50}')
    summary.write(f'{"Average":^7}{np.mean(performance_fold[keys[0]]):^11.3f}{np.mean(performance_fold[keys[1]]):^11.3f}{np.mean(performance_fold[keys[2]]):^11.3f}{np.mean(performance_fold[keys[3]]):^11.3f}')
    summary.write(f'{"STD":^7}{np.std(performance_fold[keys[0]]):^11.3f}{np.std(performance_fold[keys[1]]):^11.3f}{np.std(performance_fold[keys[2]]):^11.3f}{np.std(performance_fold[keys[3]]):^11.3f}')


summary.write(f'\n{"¤"*20}')
summary.write(f'¤ Ensamble metrics ¤')
summary.write(f'{"¤"*20}\n')

if binary_classification:
    keys = ['Specificity','Recall','Precision', 'F1', 'ROC_AUC']
    summary.write(f'{keys[0]:^11}{keys[1]:^11}{keys[2]:^11}{keys[3]:^11}{keys[4]:^11}')
    summary.write(f'{"-"*53}')
    summary.write(f'{performance_ensamble[keys[0]]:^11.3f}{performance_ensamble[keys[1]]:^11.3f}{performance_ensamble[keys[2]]:^11.3f}{performance_ensamble[keys[3]]:^11.3f}{performance_ensamble[keys[4]]:^11.3f}')
else:
    keys = ['Recall','Precision', 'F1', 'ROC_AUC']
    summary.write(f'{keys[0]:^11}{keys[1]:^11}{keys[2]:^11}{keys[3]:^11}')
    summary.write(f'{"-"*44}')
    summary.write(f'{performance_ensamble[keys[0]]:^11.3f}{performance_ensamble[keys[1]]:^11.3f}{performance_ensamble[keys[2]]:^11.3f}{performance_ensamble[keys[3]]:^11.3f}')

summary.close()
