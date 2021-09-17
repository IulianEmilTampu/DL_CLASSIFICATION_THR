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


###




## 4 - plot and save confusion matrix
importlib.reload(utilities)
ensemble_pred_argmax = []
ensemble_pred_logits = []

debug = False
dummy = False
save = False

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
    if save is True:
        acc = utilities.plotConfusionMatrix(gt, aus_ensemble_pred, classes=class_labels, savePath=save_path, draw=True)
    else:
        acc = utilities.plotConfusionMatrix(gt, aus_ensemble_pred, classes=class_labels, savePath=None, draw=True)
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

save = False

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
    ax.set_title('Model {} (OneVsAll)'.format(model_description[idx]), fontsize=20)
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

save = False
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
    ax.set_title('Model {} '.format(model_description[idx]), fontsize=20)
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
            label=f"Model {model_description[m]:{np.max([len(x) for x in model_description])}s} (area = {roc_auc[models[m]['model']]['micro']:0.2f})")

ax.plot([0, 1], [0, 1], 'k--', lw=lw)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])
ax.tick_params(labelsize=20)

# plt.grid('minor')
ax.set_xlabel('False Positive Rate', fontsize=25)
ax.set_ylabel('True Positive Rate', fontsize=25)
ax.set_title('Comparison multi-class ROC (OneVsAll) - micro-average', fontsize=20)
plt.legend(loc="lower right", fontsize=16)

# ¤¤¤¤¤¤¤¤ work on the zoom-in
colors = cycle(['blue', 'orange', 'green', 'red','purple','brown','pink','gray','olive','cyan','teal'])
axins = zoomed_inset_axes(ax, zoom=1, loc=7, bbox_to_anchor=(0,0,0.99,0.9), bbox_transform=ax.transAxes)
for m, color in zip(range(len(models)), colors):
    axins.plot(fpr[models[m]['model']]['micro'], tpr[models[m]['model']]['micro'], color=color, lw=lw)

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
    fig.savefig(os.path.join(base_folder, f'Model_comparison_micro_avg_{datetime.now().strftime("%H:%M:%S")}.pdf'), bbox_inches='tight', dpi = 100)
    fig.savefig(os.path.join(base_folder, f'Model_comparison_micro_avg_{datetime.now().strftime("%H:%M:%S")}.png'), bbox_inches='tight', dpi = 100)
    plt.close()
else:
    plt.show()

# ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤ macro average
colors = cycle(['blue', 'orange', 'green', 'red','purple','brown','pink','gray','olive','cyan','teal'])
line_style = cycle([':', '-.', '-'])
fig, ax = plt.subplots(figsize=(10,10))
for m, color, ls in zip(range(len(models)), colors, line_style):
    ax.plot(fpr[models[m]['model']]['macro'], tpr[models[m]['model']]['macro'], color=color, lw=lw, ls=ls,
            label=f"Model {model_description[m]:{np.max([len(x) for x in model_description])}s} (area = {roc_auc[models[m]['model']]['macro']:0.2f})")

ax.plot([0, 1], [0, 1], 'k--', lw=lw)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])
# plt.grid('minor')
ax.set_xlabel('False Positive Rate', fontsize=25)
ax.set_ylabel('True Positive Rate', fontsize=25)
ax.set_title('Comparison between models - macro-average', fontsize=20)
plt.legend(loc="lower right", fontsize=20)

# ¤¤¤¤¤¤¤¤ work on the zoom-in
colors = cycle(['blue', 'orange', 'green', 'red','purple','brown','pink','gray','olive','cyan','teal'])
line_style = cycle([':', '-.', '-'])
axins = zoomed_inset_axes(ax, zoom=1, loc=7, bbox_to_anchor=(0,0,0.99,0.9), bbox_transform=ax.transAxes)
for m, color, ls in zip(range(len(models)), colors, line_style):
    axins.plot(fpr[models[m]['model']]['macro'], tpr[models[m]['model']]['macro'], color=color, lw=lw, ls=ls)

# sub region of the original image
x1, x2, y1, y2 = 0.0, 0.3, 0.7, 1.0
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
ax.tick_params(labelsize=20)

axins.set_xticks(np.linspace(x1, x2, 4))
axins.set_yticks(np.linspace(y1, y2, 4))

# draw a bbox of the region of the inset axes in the parent axes and
# connecting lines between the bbox and the inset axes area
mark_inset(ax, axins, loc1=1, loc2=3, fc='none', ec='0.5', ls='--')

if save is True:
    fig.savefig(os.path.join(base_folder, f'Model_comparison_macro_avg_{datetime.now().strftime("%H:%M:%S")}.pdf'), bbox_inches='tight', dpi = 100)
    fig.savefig(os.path.join(base_folder, f'Model_comparison_macro_avg_{datetime.now().strftime("%H:%M:%S")}.png'), bbox_inches='tight', dpi = 100)
    plt.close()
else:
    plt.show()