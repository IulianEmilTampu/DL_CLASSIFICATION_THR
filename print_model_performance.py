'''
Script that given the folder to the traned and tested model, prints all the
performance_fold metrics.

Model name (all values are macro-averages across the classes)

Fold number | Sensitivity | Specificity | ROC AUC
-------------------------------------------------
      1     |    0.xyzt   |    0.xyzt   | 0.xyzt
      2     |    0.xyzt   |    0.xyzt   | 0.xyzt
-------------------------------------------------
   mean     |    0.xyzt   |    0.xyzt   | 0.xyzt
   std      |    0.xyzt   |    0.xyzt   | 0.xyzt


Fold number |  Precision  |    Recall   | F1-score
-------------------------------------------------
      1     |    0.xyzt   |    0.xyzt   | 0.xyzt
      2     |    0.xyzt   |    0.xyzt   | 0.xyzt
-------------------------------------------------
   mean     |    0.xyzt   |    0.xyzt   | 0.xyzt
   std      |    0.xyzt   |    0.xyzt   | 0.xyzt

'''


import os
import sys
import cv2
import glob
import json
import pickle
import random
import pathlib
import argparse
import importlib
import numpy as np

from sklearn.metrics import average_precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix

parser = argparse.ArgumentParser(description='Script that prints a summary of the model perfomance.')
parser.add_argument('-m','--model' ,required=True, help='Specify the folder where the trained and tested model is located')
args = parser.parse_args()

model_path = args.model

# Check that model folder exists and that the test_summary.txt file is present
if not os.path.isdir(model_path):
    raise NameError('Model not found. Given {}. Provide a valid model path.'.format(model_path))

if not os.path.isfile(os.path.join(model_path, 'test_summary.txt')):
    raise ValueError('The test_summary.txt file is not present in the model path. Run test first.')

## All good, opening test_summary.txt file


with open(os.path.join(model_path, 'test_summary.txt')) as file:
    test_summary = json.load(file)
    # labels to categorical
    labels = np.eye(np.unique(test_summary['labels']).shape[0])[test_summary['labels']]
    pred_logits = test_summary['folds_test_logits_values']

n_folds = len(pred_logits)

# check if we are working on a binary or multi-class classification
binary_classification = True
if labels.shape[-1] > 2:
    binary_classification = False

'''
For bynary classification we can compute all the metrics:
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

##
# compute encamble performance
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


## printing

print(f'\nModel Name: {os.path.basename(model_path)}\n')
# add test time overall and per image

print(f'{"¤"*21}')
print(f'¤ Per-fold metrics ¤')
print(f'{"¤"*21}\n')

if binary_classification:
    keys = ['Specificity','Recall','Precision', 'F1', 'ROC_AUC']

    print(f'{"Fold":^7}{keys[0]:^11}{keys[1]:^11}{keys[2]:^11}{keys[3]:^11}{keys[4]:^11}')

    for i in range(n_folds):
        print(f'{i+1:^7}{performance_fold[keys[0]][i]:^11.3f}{performance_fold[keys[1]][i]:^11.3f}{performance_fold[keys[2]][i]:^11.3f}{performance_fold[keys[3]][i]:^11.3f}{performance_fold[keys[-1]][i]:^11.3f}')
    print(f'{"-"*60}')
    print(f'{"Average":^7}{np.mean(performance_fold[keys[0]]):^11.3f}{np.mean(performance_fold[keys[1]]):^11.3f}{np.mean(performance_fold[keys[2]]):^11.3f}{np.mean(performance_fold[keys[3]]):^11.3f}{np.mean(performance_fold[keys[4]]):^11.3f}')
    print(f'{"STD":^7}{np.std(performance_fold[keys[0]]):^11.3f}{np.std(performance_fold[keys[1]]):^11.3f}{np.std(performance_fold[keys[2]]):^11.3f}{np.std(performance_fold[keys[3]]):^11.3f}{np.std(performance_fold[keys[4]]):^11.3f}')

else:
    keys = ['Recall','Precision', 'F1', 'ROC_AUC']

    print(f'{"Fold":^7}{keys[0]:^11}{keys[1]:^11}{keys[2]:^11}{keys[3]:^11}')

    for i in range(n_folds):
        print(f'{i+1:^7}{performance_fold[keys[0]][i]:^11.3f}{performance_fold[keys[1]][i]:^11.3f}{performance_fold[keys[2]][i]:^11.3f}{performance_fold[keys[3]][i]:^11.3f}')
    print(f'{"-"*50}')
    print(f'{"Average":^7}{np.mean(performance_fold[keys[0]]):^11.3f}{np.mean(performance_fold[keys[1]]):^11.3f}{np.mean(performance_fold[keys[2]]):^11.3f}{np.mean(performance_fold[keys[3]]):^11.3f}')
    print(f'{"STD":^7}{np.std(performance_fold[keys[0]]):^11.3f}{np.std(performance_fold[keys[1]]):^11.3f}{np.std(performance_fold[keys[2]]):^11.3f}{np.std(performance_fold[keys[3]]):^11.3f}')


print(f'\n{"¤"*20}')
print(f'¤ Ensamble metrics ¤')
print(f'{"¤"*20}\n')

if binary_classification:
    keys = ['Specificity','Recall','Precision', 'F1', 'ROC_AUC']
    print(f'{keys[0]:^11}{keys[1]:^11}{keys[2]:^11}{keys[3]:^11}{keys[4]:^11}')
    print(f'{"-"*53}')
    print(f'{performance_ensamble[keys[0]]:^11.3f}{performance_ensamble[keys[1]]:^11.3f}{performance_ensamble[keys[2]]:^11.3f}{performance_ensamble[keys[3]]:^11.3f}{performance_ensamble[keys[4]]:^11.3f}')
else:
    keys = ['Recall','Precision', 'F1', 'ROC_AUC']
    print(f'{keys[0]:^11}{keys[1]:^11}{keys[2]:^11}{keys[3]:^11}')
    print(f'{"-"*44}')
    print(f'{performance_ensamble[keys[0]]:^11.3f}{performance_ensamble[keys[1]]:^11.3f}{performance_ensamble[keys[2]]:^11.3f}{performance_ensamble[keys[3]]:^11.3f}')






















