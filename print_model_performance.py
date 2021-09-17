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
parser.add_argument('-p','--plot' ,required=False, help='Specify if replotting and saving Confusion matrix, ROC and PP', default=False)
parser.add_argument('-ul','--unique_labels',nargs='+' ,required=False, help='Specify labels to be used in the plot.', default=[])
args = parser.parse_args()

model_path = args.model
plot = args.plot == 'True'
unique_labels = [str(i) for i in args.unique_labels]

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

##
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


##  Confusion matrix

if plot is True:
    from matplotlib import pyplot as plt
    import itertools
    from sklearn.metrics import roc_curve, auc
    from itertools import cycle
    from scipy import interp
    from tensorflow.keras.utils import to_categorical
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset

    import numpy as np

    print('\n\nPlotting...')

    def plotConfusionMatrix(GT, PRED, classes, Labels=None, cmap=plt.cm.Blues, savePath=None, draw=False):
        '''
        Funtion that plots the confision matrix given the ground truths and the predictions
        '''
        # compute confusion matrix

        cm = confusion_matrix(GT, PRED)
        fig = plt.figure(figsize=(10,10))
        plt.imshow(cm, interpolation=None, cmap=cmap)
        # plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45, fontsize=25)
        plt.yticks(tick_marks, classes, fontsize=25)

        thresh = cm.max()/2

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i,j],
                horizontalalignment='center',
                verticalalignment='center',
                color='white' if cm[i,j] > thresh else 'black',
                fontsize=35)

        # plt.tight_layout()
        plt.ylabel('True label', fontsize=20)
        plt.xlabel('Prediction', fontsize=20)

        acc = 100*(np.trace(cm) / np.sum(cm))
        plt.title('Confusion matrix -> ' + 'Accuracy {:05.2f}'.format(acc), fontsize=20)
        fig.tight_layout()

        # save is needed
        if savePath is not None:
            if os.path.isdir(savePath):
                fig.savefig(os.path.join(savePath, 'ConfisionMatrix_ensemble_prediction_from_print_function.pdf'), bbox_inches='tight', dpi = 100)
                fig.savefig(os.path.join(savePath, 'ConfisionMatrix_ensemble_prediction_from_print_function.png'), bbox_inches='tight', dpi = 100)
            else:
                raise ValueError('Invalida save path: {}'.format(savePath))

        if draw is True:
            plt.draw()
        else:
            plt.close()

        return acc


    def plotROC(GT, PRED, classes, savePath=None, draw=False):
        '''
        Funtion that plots the ROC curve given the ground truth and the logits prediction

        INPUT
        - GT: true labels
        - PRED: array of float the identifies the logits prediction
        - classes: list of string that identifies the labels of each class
        - save path: sting that identifies the path where to save the ROC plots
        - draw: bool if to print or not the ROC curve

        RETURN
        - fpr: dictionary that contains the false positive rate for every class and
            the overall micro and marco averages
        - trp: dictionary that contains the true positive rate for every class and
            the overall micro and marco averages
        - roc_auc: dictionary that contains the area under the curve for every class and
            the overall micro and marco averages

        Check this link for better understanding of micro and macro-averages
        https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin

        Here computing both the macro-average ROC and the micro-average ROC.
        Using code from https://scikit-learn.org/dev/auto_examples/model_selection/plot_roc.html with modification
        '''
        # define variables
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        n_classes = len(classes)
        lw = 2 # line width

        # make labels categorical
        GT = to_categorical(GT, num_classes=n_classes)

        # ¤¤¤¤¤¤¤¤¤¤¤ micro-average roc
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(GT[:,i], PRED[:,i])
            roc_auc[i] = auc(fpr[i], tpr[i])


        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(GT.ravel(), PRED.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # ¤¤¤¤¤¤¤¤¤¤ macro-average roc

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves and save
        fig, ax = plt.subplots(figsize=(10,10))
        ax.plot(fpr["micro"], tpr["micro"],
                label='micro-average ROC curve (area = {0:0.2f})'
                    ''.format(roc_auc["micro"]),
                color='red', linestyle='-', linewidth=4)

        # ax.plot(fpr["macro"], tpr["macro"],
        #         label='macro-average ROC curve (area = {0:0.2f})'
        #             ''.format(roc_auc["macro"]),
        #         color='navy', linestyle=':', linewidth=4)

        # colors = cycle(['blue', 'orange', 'green', 'red','purple','brown','pink','gray','olive','cyan','teal'])
        # for i, color in zip(range(n_classes), colors):
        #     ax.plot(fpr[i], tpr[i], color=color, lw=lw,
        #             label='ROC curve of class {} (area = {:0.2f})'
        #             ''.format(classes[i], roc_auc[i]))

        ax.plot([0, 1], [0, 1], 'k--', lw=lw)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_xlabel('False Positive Rate', fontsize=25)
        ax.set_ylabel('True Positive Rate', fontsize=25)
        ax.set_title('Multi-class ROC (OneVsAll)', fontsize=20)
        plt.setp(ax.get_xticklabels(), fontsize=15)
        plt.setp(ax.get_yticklabels(), fontsize=15)
        plt.legend(loc="lower right", fontsize=20)

        # ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤ work on the zummed-in image
        colors = cycle(['blue', 'orange', 'green', 'red','purple','brown','pink','gray','olive','cyan','teal'])
        axins = zoomed_inset_axes(ax, zoom=1, loc=7, bbox_to_anchor=(0,0,0.99,0.9), bbox_transform=ax.transAxes)

        axins.plot(fpr["micro"], tpr["micro"],
                label='micro-average ROC curve (area = {0:0.2f})'
                    ''.format(roc_auc["micro"]),
                color='red', linestyle='-', linewidth=4)

        # axins.plot(fpr["macro"], tpr["macro"],
        #         label='macro-average ROC curve (area = {0:0.2f})'
        #             ''.format(roc_auc["macro"]),
        #         color='navy', linestyle=':', linewidth=4)

        # for i, color in zip(range(n_classes), colors):
        #     axins.plot(fpr[i], tpr[i], color=color, lw=lw,
        #             label='ROC curve of class {} (area = {:0.2f})'
        #             ''.format(classes[i], roc_auc[i]))

        # sub region of the original image
        x1, x2, y1, y2 = 0.0, 0.3, 0.7, 1.0
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)

        axins.set_xticks(np.linspace(x1, x2, 4))
        axins.set_yticks(np.linspace(y1, y2, 4))
        plt.setp(axins.get_xticklabels(), fontsize=13)
        plt.setp(axins.get_yticklabels(), fontsize=13)

        # draw a bbox of the region of the inset axes in the parent axes and
        # connecting lines between the bbox and the inset axes area
        mark_inset(ax, axins, loc1=1, loc2=3, fc='none', ec='0.5', ls='--')

        # save is needed
        if savePath is not None:
            if os.path.isdir(savePath):
                fig.savefig(os.path.join(savePath, 'Multiclass ROC_from_print_function.pdf'), bbox_inches='tight', dpi = 100)
                fig.savefig(os.path.join(savePath, 'Multiclass ROC_from_print_function.png'), bbox_inches='tight', dpi = 100)
            else:
                raise ValueError('Invalid save path: {}'.format(savePath))

        if draw is True:
            plt.draw()
        else:
            plt.close()

        return fpr, tpr, roc_auc

    plotConfusionMatrix(np.argmax(labels,-1), ensemble_pred_argmax.ravel(), classes=unique_labels, savePath=model_path, draw=False)

    plotROC(np.argmax(labels,-1), ensemble_pred_logits, classes=unique_labels, savePath=model_path, draw=False)




















