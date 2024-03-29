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
parser = argparse.ArgumentParser(description='Script that prints a summary of the model perfomance.')
parser.add_argument('-m','--model_path' ,required=True, help='Specify the folder where the trained model is located')
parser.add_argument('-d','--dataset_path' ,required=False, help='Specify where the dataset is located', default=False)
parser.add_argument('-mv','--model_version' ,required=False, help='Specify if to run the training on the best model (best) or the last (last)', default="best")
args = parser.parse_args()

model_path = args.model_path
dataset_path = args.dataset_path
model_version = args.model_version

# # # # DEBUG
# model_path = '/flush/iulta54/Research/P3-THR_DL/trained_models/M4_c6_BatchNorm_dr0.4_lr0.00001_wcce_none_batch64'
# dataset_path = '/flush/iulta54/Research/Data/OCT/Thyroid_2019_refined_DeepLearning'
# model_version = "best"

title="Testing script"
print(f'\n{"-"*len(title)}')
print(f'{title}')
print(f'{"-"*len(title)}\n')

# check forlders
if not os.path.isdir(model_path):
    raise ValueError(f'Model not found. Given {model_path}')
else:
    # check that the configuration file is in place
    if not os.path.isfile(os.path.join(model_path,'config.json')):
        raise ValueError(f'Configuration file not found for the given model. Check that the model was configured and trained. Given {os.path.join(model_path,"config.json")}')
    else:
        print("Model and config file found.")

if not os.path.isdir(dataset_path):
    raise ValueError(f'Dataset path not found. Given {dataset_path}')
else:
    print('Dataset path found.')

# check that the model_version setting is a correct one (best, last, ensamble-not jet implemented)
if not any([model_version==s for s in ["best", "last", "ensamble"]]):
    raise ValueError(f'The given model version for the testing is unknown. Given {model_version}, expected best, last or ensamble')

print(f'Working on model {os.path.basename(model_path)}')
print(f'Model configuration set for testing: {model_version}')

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
importlib.reload(utilities_models_tf)

test_fold_summary = {}
folds = glob.glob(os.path.join(model_path,"fold_*"))

# get the right model based on the model_version_specification
if model_version=="best":
    model_name_version = "model.tf"
elif model_version=="last":
    model_name_version = "last_model.tf"

for idx, f in enumerate(folds):
    print(f'Working on fold {idx+1}/{len(folds)}')
    # load model
    if os.path.exists(os.path.join(f, model_name_version)):
        model = tf.keras.models.load_model(os.path.join(f, model_name_version), compile=False)
    else:
        raise Exception('Model not found')

    test_gt, test_prediction, test_time = utilities_models_tf.test_independent(model, config, test_dataset)
    test_fold_summary[idx]={
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
print(f'Saving information...')
# ############# save the information that is already available
test_summary = OrderedDict()

test_summary['model_name'] = config['model_save_name']
test_summary['labels'] = [int(i) for i in test_fold_summary[0]['ground_truth']]
test_summary['folds_test_logits_values'] = [test_fold_summary[cv]['prediction'].tolist() for cv in range(len(folds))]
test_summary['test_time'] = utilities.tictoc_from_time(np.sum([test_fold_summary[cv]['test_time'] for cv in range(len(folds))]))
test_summary['test_model_version'] = model_version
test_summary['test_date'] = time.strftime("%Y%m%d-%H%M%S")

# ############ plot and save confucion matrix
ensemble_pred_argmax = []
ensemble_pred_logits = []

# compute ensemble
# compute the logits mean along the folds
ensemble_pred_logits = np.array(test_summary['folds_test_logits_values']).mean(axis=0)
# compute argmax prediction
ensemble_pred_argmax = np.argmax(ensemble_pred_logits, axis=1)

acc = utilities.plotConfusionMatrix(test_summary['labels'], ensemble_pred_argmax,
            classes=config['label_description'],
            savePath=model_path,
            saveName=f'ConfusionMatrix_{model_version}_model',
            draw=False)

# ############ plot and save ROC curve
fpr, tpr, roc_auc = utilities.plotROC(test_summary['labels'], ensemble_pred_logits,
            classes=config['label_description'],
            savePath=model_path,
            saveName=f'Multiclass_ROC_{model_version}_model',
            draw=False)

# make elements of the dictionary to be lists for saving
for key, value in fpr.items():
    fpr[key]=value.tolist()
for key, value in tpr.items():
    tpr[key]=value.tolist()
for key, value in roc_auc.items():
    roc_auc[key]=value.tolist()

# ############ plot and save PR curve
precision, recall, average_precision, F1 = utilities.plotPR(test_summary['labels'],
            ensemble_pred_logits,
            classes=config['label_description'],
            savePath=model_path,
            saveName=f'Multiclass_PR_{model_version}_model',
            draw=False)

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
with open(os.path.join(model_path,f'{model_version}_model_version_test_summary.txt'), 'w') as fp:
    json.dump(test_summary, fp)

## save summary (can be improved, but using the routine from print_model_performance)
from sklearn.metrics import average_precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix, accuracy_score

def get_metrics(true_logits, pred_logits, average='macro'):
    '''
    Utility that given confusion matrics, returns a dictionary containing:
    tp tn fp fn : overall TP, TN, FP, FN values for each of the classes
    precision (specificity) : for each of the classes
    recall (sensitivity) : for each of the classes
    f1-score : for each of the classes
    auc : for each of the classes
    overall_acc : over all classes
    overall_specificity : over all classes
    overall_precision : over all classes
    overall_f1-score : over all classes
    overall_auc : over all classes
    '''
    # compute confusion matrix
    cnf_matrix = confusion_matrix(np.argmax(true_logits,-1), np.argmax(pred_logits, -1))

    # compute TP, TN, FP, FN

    FP = (cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)).astype(float)
    FN = (cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)).astype(float)
    TP = (np.diag(cnf_matrix)).astype(float)
    TN = (cnf_matrix.sum() - (FP + FN + TP)).astype(float)

    # compute per class metrics
    summary_dict = {
                'precision': TN / (FP+TN),
                'recall': TP / (TP+FN),
                'accuracy': (TP+TN) / (TP+TN+FP+FN),
                'f1-score': TP / (TP + 0.5*(FP+FN)),
                'auc': roc_auc_score(true_logits, pred_logits, average=None),
                }

    # compute overall metrics
    summary_dict['overall_precision']=average_precision_score(true_logits,
                                        pred_logits,
                                        average=average)
    summary_dict['overall_recall']=recall_score(np.argmax(true_logits,-1),
                                        np.argmax(pred_logits,-1),
                                        average=average)
    summary_dict['overall_accuracy']=accuracy_score(np.argmax(true_logits,-1),
                                        np.argmax(pred_logits,-1))
    summary_dict['overall_f1-score']=f1_score(np.argmax(true_logits,-1),
                                        np.argmax(pred_logits,-1),
                                        average=average)
    summary_dict['overall_auc']=roc_auc_score(true_logits,
                                        pred_logits,
                                        multi_class='ovr',
                                        average=average)

    return summary_dict

n_folds = len(folds)

labels = np.eye(np.unique(test_summary['labels']).shape[0])[test_summary['labels']]
pred_logits = test_summary['folds_test_logits_values']

# Computupe per fold performance
performance_fold = {
            'ROC_AUC':[],
            'Precision': [],
            'Recall':[],
            'F1':[],
            'Accuracy':[]
                }
per_fold_performance = []

for f in range(len(folds)):
    per_fold_performance.append(get_metrics(labels, pred_logits[f]))

# compute ensamble performance
# compute the logits mean along the folds
ensemble_pred_logits = np.array(pred_logits).mean(axis=0)
# compute argmax prediction
ensemble_pred_argmax = np.argmax(ensemble_pred_logits, axis=1)

performance_ensamble = get_metrics(labels, ensemble_pred_logits)

# ######################### printing on file

summary = open(os.path.join(model_path,f'{model_version}_model_version_short_test_summary.txt'), 'w')

summary.write(f'\nModel Name: {os.path.basename(model_path)}\n\n')

# add test time overall and per image
average_test_time = np.mean([test_fold_summary[cv]['test_time'] for cv in range(len(folds))])
average_test_time_per_image = np.mean([test_fold_summary[cv]['test_time'] for cv in range(len(folds))])/labels.shape[0]
summary.write(f'Overall model test time (average over folds): {utilities.tictoc_from_time(average_test_time)}\n')
summary.write(f'Average test time per image (average over folds): {utilities.tictoc_from_time(average_test_time_per_image)}\n\n')

# print a summary of the testing per fold, class and ensamble
keys = ['precision','recall', 'accuracy', 'f1-score', 'auc']
max_len_k=max([len(k) for k in keys])

classes = config['label_description']
max_len_c=max(max([len(k) for k in classes]),len('Overall'))

max_len_f=max([len(s) for s in ["Fold", "Average","STD","Ensamble"]])

# build dict that holds the avg of all metrics
avg_dict = {k:[] for k in keys}

# build headers strings
fold_str = f'{"Fold":^{max_len_f}}'
class_str = f'{"Class":^{max_len_c}}'
keys_str = ''.join([f'{k.capitalize():^{max_len_k+2}}' for k in keys])

# start printing
summary.write(f'\n{"¤"*(max_len_f+max_len_c+len(keys_str))}'+'\n')
summary.write(f'{"¤ Per-fold metrics ¤":^{(max_len_f+max_len_c+len(keys_str))}}'+'\n')
summary.write(f'{"¤"*(max_len_f+max_len_c+len(keys_str))}\n')

# print header
summary.write(fold_str+class_str+keys_str+'\n')

# print per fold and class metrics
for idx, fp in enumerate(per_fold_performance):
    fold_num_str = f'{str(idx+1):^{max_len_f}}'
    # print per class performance
    for idc, c in enumerate(classes):
        class_metrics = ''.join([f'{str(round(fp[k][idc],3)):^{max_len_k+2}}' for k in keys])
        if idc == 0:
            summary.write(fold_num_str+f'{c:^{max_len_c}}'+class_metrics+'\n')
        else:
            summary.write(f'{" ":^{max_len_f}}'+f'{c:^{max_len_c}}'+class_metrics+'\n')

    # print overall performance
    summary.write(f'{"-"*(max_len_f+max_len_c+len(keys_str))}'+'\n')
    overall_metric_str = ''.join([f'{str(round(fp["overall_"+k],3)):^{max_len_k+2}}' for k in keys])
    summary.write(fold_num_str+f'{"Overall":^{max_len_c}}'+overall_metric_str+'\n')
    summary.write(f'{"-"*(max_len_f+max_len_c+len(keys_str))}'+'\n\n')

    # save overall metrics for later
    [avg_dict[k].append(fp['overall_'+k]) for k in keys]

# print average overall metrics for the folds
avg_overall_metric_str = ''.join([f'{str(round(np.mean(avg_dict[k]),3)):^{max_len_k+2}}' for k in keys])
std_overall_metric_str = ''.join([f'{str(round(np.std(avg_dict[k]),3)):^{max_len_k+2}}' for k in keys])
summary.write(f'{"="*(max_len_f+max_len_c+len(keys_str))}'+'\n')
summary.write(fold_str+class_str+keys_str+'\n')
summary.write(f'{"Average":^{max_len_f}}'+f'{"":^{len(class_str)}}'+avg_overall_metric_str+'\n')
summary.write(f'{"STD":^{max_len_f}}'+f'{"":^{len(class_str)}}'+std_overall_metric_str+'\n')

# plot ensamble metrics
summary.write(f'\n{"¤"*(max_len_f+max_len_c+len(keys_str))}'+'\n')
summary.write(f'{"¤ Ensamble metrics ¤":^{(max_len_f+max_len_c+len(keys_str))}}'+'\n')
summary.write(f'{"¤"*(max_len_f+max_len_c+len(keys_str))}\n')

# print header
summary.write(f'{"Ensamble":^{max_len_f}}'+class_str+keys_str+'\n')
# print per class performance
fp = performance_ensamble
for idc, c in enumerate(classes):
    class_metrics = ''.join([f'{str(round(fp[k][idc],3)):^{max_len_k+2}}' for k in keys])
    summary.write(f'{" ":^{max_len_f}}'+f'{c:^{max_len_c}}'+class_metrics+'\n')

# print overall performance
summary.write(f'{"-"*(max_len_f+max_len_c+len(keys_str))}'+'\n')
overall_metric_str = ''.join([f'{str(round(fp["overall_"+k],3)):^{max_len_k+2}}' for k in keys])
summary.write(f'{" ":^{max_len_f}}'+f'{"Overall":^{max_len_c}}'+overall_metric_str+'\n')

summary.close()