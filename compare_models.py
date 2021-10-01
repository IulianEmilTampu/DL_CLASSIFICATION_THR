'''
Script that given a list of models, uses the summary test file to plot the ROC,
the PP comparing the models. The training performance is also plotted to show
how the different models trained (looking for overfitting).

Steps
1 - get models, models' paths. Check that all the models have the test summary file.
2 - load the values needed to plot the ROCs for comparison
3 - loop through all the models and for each get the training curves (tr and val
    loss, accuracy and F1-score) for each fold.
4 - plot overall curves (one graph for each parameter).
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
from itertools import cycle
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from sklearn.metrics import average_precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix, auc

## 1 - get model path and check that everything is in place

# parser = argparse.ArgumentParser(description='Script that compares models ROC and training curves.')
# parser.add_argument('-m','--models' ,required=True, help='List of model names that should be compared.')
# parser.add_argument('-tmp','--trained_model_path' ,required=True, help='Path of where the trained models are located.', default=False)
# args = parser.parse_args()
#
# model_path = args.model
# trained_model_path = args.trained_model_path
# models = [str(i) for i in args.models]

# ############ for debug
wcce = "none"
lr = 0.0001
dr = 0.2
trained_model_path = "/flush/iulta54/Research/P3-THR_DL/trained_models"
models = [f"M4_c6_BatchNorm_dr{dr}_lr{lr}_wcce_{wcce}_batch16",
          f"M4_c6_BatchNorm_dr{dr}_lr{lr}_wcce_{wcce}_batch64",
          f"M4_c6_BatchNorm_dr{dr}_lr{lr}_wcce_{wcce}_batch128"]

# models = ["M4_c6_BatchNorm_dr0.2_lr0.001_wcce_none_batch16",
#           "M4_c6_BatchNorm_dr0.2_lr0.005_wcce_none_batch128",
#           "M4_c6_BatchNorm_dr0.2_lr0.0005_wcce_weights_batch128",
#           "M4_c6_BatchNorm_dr0.2_lr0.0001_wcce_none_batch64",
#           "M4_c6_BatchNorm_dr0.2_lr0.001_wcce_weights_batch64"]

# Check that model folder exists and that the test_summary.txt file is present
for m in models:
    if not os.path.isdir(os.path.join(trained_model_path, m)):
        raise NameError('Model not found. Given {os.path.join(trained_model_path, m)}. Provide a valid model path.')
    else:
        # check that the test_summary.txt file is present
        if not os.path.isfile(os.path.join(trained_model_path, m, "test_summary.txt")):
            raise ValueError(f'The test_summary.txt file is not present in the model path. Run test first. Given {os.path.join(trained_model_path, m, "test_summary.txt")}')

# get the true positive and false positive rates for all the models
fpr = dict()
tpr = dict()
roc_auc = dict()
acc = dict()
f1 = dict()

for m in models:
    # load the test_summary.txt file and get information
    with open(os.path.join(trained_model_path, m, 'test_summary.txt')) as file:
        test_summary = json.load(file)
        fpr[m] = test_summary['false_positive_rate']
        tpr[m] = test_summary['true_positive_rate']
        # check if roc_auc is saved, if not compute (in older version was not saved)
        if "roc_auc" in test_summary:
            roc_auc[m] = test_summary['roc_auc']
        else:
            roc_auc[m] = dict()
            roc_auc[m]['micro'] = auc(fpr[m]["micro"], tpr[m]["micro"])
            roc_auc[m]['macro'] = auc(fpr[m]["macro"], tpr[m]["macro"])
        # save also acc and f1 acores
        acc[m] = test_summary["accuracy"]
        f1[m]= {"macro" : test_summary["F1"]["macro"],
                "micro" : test_summary["F1"]["micro"]}

# plot the comparicon ROC between models (micro and macro average separately)

# overall settings
tick_font_size=20
title_font_size=20
label_font_size=25
legend_font_size=8
line_width=2
save = False
mml = np.max([len(i) for i in models])

# ########## MACRO AVERAGE
fig, ax = plt.subplots(figsize=(10,10))
colors = cycle(['blue', 'orange', 'green', 'red','purple','brown','pink','gray','olive','cyan','teal'])
for m, color in zip(models, colors):
    ax.plot(fpr[m]['macro'], tpr[m]['macro'], color=color, lw=line_width,
            label=f'{m:{mml+2}s}(AUC:{roc_auc[m]["macro"]:0.3f}, F1:{f1[m]["macro"]:0.3f}, ACC:{acc[m]}')

ax.plot([0, 1], [0, 1], 'k--', lw=line_width)
major_ticks = np.arange(0, 1, 0.1)
minor_ticks = np.arange(0, 1, 0.05)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])
ax.tick_params(labelsize=tick_font_size)
plt.grid(color='b', linestyle='-.', linewidth=0.1, which='both')


ax.set_xlabel('False Positive Rate', fontsize=label_font_size)
ax.set_ylabel('True Positive Rate', fontsize=label_font_size)
ax.set_title('Comparison multi-class ROC - macro-average', fontsize=title_font_size)
ax.legend(loc="lower right", fontsize=legend_font_size)

# ¤¤¤¤¤¤¤¤ work on the zoom-in
colors = cycle(['blue', 'orange', 'green', 'red','purple','brown','pink','gray','olive','cyan','teal'])
axins = zoomed_inset_axes(ax, zoom=1, loc=7, bbox_to_anchor=(0,0,0.99,0.9), bbox_transform=ax.transAxes)
for m, color in zip(models, colors):
    axins.plot(fpr[m]['macro'], tpr[m]['macro'], color=color, lw=line_width)

# sub region of the original image
x1, x2, y1, y2 = 0.0, 0.3, 0.7, 1.0
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.grid(color='b', linestyle='--', linewidth=0.1)

axins.set_xticks(np.linspace(x1, x2, 4))
axins.set_yticks(np.linspace(y1, y2, 4))

# draw a bbox of the region of the inset axes in the parent axes and
# connecting lines between the bbox and the inset axes area
mark_inset(ax, axins, loc1=1, loc2=3, fc='none', ec='0.5', ls='--')

if save is True:
    fig.savefig(os.path.join(trained_model_path, f'Model_comparison_macro_avg_{datetime.now().strftime("%H:%M:%S")}.pdf'), bbox_inches='tight', dpi = 100)
    fig.savefig(os.path.join(trained_model_path, f'Model_comparison_macro_avg_{datetime.now().strftime("%H:%M:%S")}.png'), bbox_inches='tight', dpi = 100)
    plt.close()
else:
    plt.show()

# ########## MICRO AVERAGE
fig, ax = plt.subplots(figsize=(10,10))
colors = cycle(['blue', 'orange', 'green', 'red','purple','brown','pink','gray','olive','cyan','teal'])
for m, color in zip(models, colors):
    ax.plot(fpr[m]['micro'], tpr[m]['micro'], color=color, lw=line_width,
            label=f'{m:{mml+2}s}(AUC:{roc_auc[m]["micro"]:0.3f}, F1:{f1[m]["micro"]:0.3f}, ACC:{acc[m]}')

ax.plot([0, 1], [0, 1], 'k--', lw=line_width)
major_ticks = np.arange(0, 1.1, 0.1)
minor_ticks = np.arange(0, 1.1, 0.05)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])
ax.tick_params(labelsize=tick_font_size)
plt.grid(color='b', linestyle='-.', linewidth=0.1, which='both')


ax.set_xlabel('False Positive Rate', fontsize=label_font_size)
ax.set_ylabel('True Positive Rate', fontsize=label_font_size)
ax.set_title('Comparison multi-class ROC - micro-average', fontsize=title_font_size)
ax.legend(loc="lower right", fontsize=legend_font_size)

# ¤¤¤¤¤¤¤¤ work on the zoom-in
colors = cycle(['blue', 'orange', 'green', 'red','purple','brown','pink','gray','olive','cyan','teal'])
axins = zoomed_inset_axes(ax, zoom=1, loc=7, bbox_to_anchor=(0,0,0.99,0.9), bbox_transform=ax.transAxes)
for m, color in zip(models, colors):
    axins.plot(fpr[m]['micro'], tpr[m]['micro'], color=color, lw=line_width)

# sub region of the original image
x1, x2, y1, y2 = 0.0, 0.3, 0.7, 1.0
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.grid(color='b', linestyle='--', linewidth=0.1)

axins.set_xticks(np.linspace(x1, x2, 4))
axins.set_yticks(np.linspace(y1, y2, 4))

# draw a bbox of the region of the inset axes in the parent axes and
# connecting lines between the bbox and the inset axes area
mark_inset(ax, axins, loc1=1, loc2=3, fc='none', ec='0.5', ls='--')

if save is True:
    fig.savefig(os.path.join(trained_model_path, f'Model_comparison_micro_avg_{datetime.now().strftime("%H:%M:%S")}.pdf'), bbox_inches='tight', dpi = 100)
    fig.savefig(os.path.join(trained_model_path, f'Model_comparison_micro_avg_{datetime.now().strftime("%H:%M:%S")}.png'), bbox_inches='tight', dpi = 100)
    plt.close()
else:
    plt.show()

## get per model and per fold training curve values

tr_loss = dict()
val_loss = dict()

tr_acc = dict()
val_acc = dict()

tr_f1 = dict()
val_f1 = dict()

for m in models:
    tr_loss[m]=dict()
    val_loss[m]=dict()
    tr_acc[m]=dict()
    val_acc[m]=dict()
    tr_f1[m]=dict()
    val_f1[m]=dict()
    for idx, f in enumerate(glob.glob(os.path.join(trained_model_path, m, "fold_*"))):
        # get fold values
        with open(os.path.join(f, "model_summary_json.txt")) as json_file:
            fold_info=json.load(json_file)
            tr_loss[m][idx]=fold_info["TRAIN_LOSS_HISTORY"]
            val_loss[m][idx]=fold_info["VALIDATION_LOSS_HISTORY"]
            tr_acc[m][idx]=fold_info["TRAIN_ACC_HISTORY"]
            val_acc[m][idx]=fold_info["VALIDATION_ACC_HISTORY"]
            tr_f1[m][idx]=fold_info["TRAIN_F1_HISTORY"]
            val_f1[m][idx]=fold_info["VALIDATION_F1_HISTORY"]

## plot training curves

# ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤ utility function
def get_mean_and_std(parameter_dict):
    '''
    Utility that given the dictionary containing the training history values for
    one model, returns the mean and std across the epochs. It also returns until
    where each epoch has trained (used for marking in the plot).
    '''
    # get all the fold values for the parameter
    per_epoch_values = [value for key, value in parameter_dict.items()]
    # get number of epochs for each fold
    n_epochs = [len(e) for e in per_epoch_values]
    # create a masked array and fill in all the values
    arr = np.ma.empty((len(n_epochs), np.max(n_epochs)))
    arr.mask = True
    for idx, v in enumerate(per_epoch_values):
        arr[idx, :len(v)]=v

    return arr.mean(axis=0), arr.std(axis=0), [n-1 for n in n_epochs]
# ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤ utility function


# general settings
tick_font_size=20
title_font_size=16
label_font_size=12
legend_font_size=8
line_width=2
alpha_fillin = 0.1
save = True

y_labels = ["loss", "accuracy", "F1-score"]
x_label = "epochs"
parameters = [[tr_loss, val_loss], [tr_acc, val_acc], [tr_f1, val_f1]]

for parameter, y_label in zip(parameters,y_labels):
    colors = cycle(['blue', 'orange', 'green', 'red','purple','brown','pink','gray','olive','cyan','teal'])
    markers = cycle(["v", "D", "<", "s", "*", "P", "X", "p", "d", "+"])
    # create figure
    fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(15,10))
    aus_epochs = []
    # loop through the models and get mean and std for tr and validation
    for m, color, marker in zip(models, colors, markers):
        tr_mu, tr_std, tr_ne = get_mean_and_std(parameter[0][m])
        val_mu, val_std, val_ne = get_mean_and_std(parameter[1][m])

        # get some parametrs useful for both tr and val
        epochs=np.arange(0, tr_mu.shape[0], 1)
        marker_on = np.full_like(epochs, False, dtype=bool)
        marker_on[tr_ne] = True
        aus_epochs.append(tr_mu.shape[0])

        # plot training
        ax = axes[0]
        ax.plot(epochs, tr_mu, lw=line_width, color=color, markevery=marker_on, marker=marker, label=m)
        ax.fill_between(epochs, tr_mu+tr_std, tr_mu-tr_std, facecolor=color, alpha=alpha_fillin)
        ax.set_title(f'Training {y_label} curves', fontsize=title_font_size)

        # plot validation
        ax = axes[1]
        ax.plot(epochs, val_mu, lw=line_width, color=color, markevery=marker_on, marker=marker, label=m)
        ax.fill_between(epochs, val_mu+val_std, val_mu-val_std, facecolor=color, alpha=alpha_fillin)
        ax.set_title(f'Validation {y_label} curves', fontsize=title_font_size)

    # final settings on the axes
    max_epochs = np.max(aus_epochs)
    major_ticks = np.arange(0, max_epochs, 5)
    minor_ticks = np.arange(0, max_epochs, 1)
    for ax in axes:
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)

        ax.set_xlabel(x_label, fontsize=label_font_size)
        ax.set_ylabel(y_label, fontsize=label_font_size)

        ax.grid(which="both", color='k', linestyle='--', linewidth=0.1, alpha=0.5)

        if y_label=='loss':
            ax.legend(loc='upper right')
        else:
            ax.legend(loc='lower right')

    # save figure if needed
    if save is True:
        fig.savefig(os.path.join(trained_model_path, f'Model_comparison_{y_label}_{datetime.now().strftime("%H:%M:%S")}.pdf'), bbox_inches='tight', dpi = 100)
        fig.savefig(os.path.join(trained_model_path, f'Model_comparison_{y_label}_{datetime.now().strftime("%H:%M:%S")}.png'), bbox_inches='tight', dpi = 100)
        plt.close()
    else:
        plt.show()


##

wcce = ["none", "weights"]
lr = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
dr = [0.2, 0.3, 0.3]
batch = [16, 64, 128]

max_acc = [{"model":"", "value" : 0} for i in range(4)]

max_micro_auc = [{"model":"", "value" : 0} for i in range(4)]
max_macro_auc = [{"model":"", "value" : 0} for i in range(4)]

max_micro_f1 = [{"model":"", "value" : 0} for i in range(4)]
max_macro_f1 = [{"model":"", "value" : 0} for i in range(4)]

all_acc = []
all_micro_auc = []
all_macro_auc = []

all_micro_f1 = []
all_macro_f1 = []

all_model_names = []

trained_model_path = "/flush/iulta54/Research/P3-THR_DL/trained_models"

for d in dr:
    for l in lr:
        for w in wcce:
            for b in batch:
                # build model name
                model = f"M4_c6_BatchNorm_dr{d}_lr{l}_wcce_{w}_batch{b}"

                # check if model exists
                if os.path.isdir(os.path.join(trained_model_path, model)):
                    # check that the test_summary.txt file is present
                    if os.path.isfile(os.path.join(trained_model_path, model, "test_summary.txt")):
                        # save information
                        all_model_names.append(model)

                        # open test_summary and save information
                        with open(os.path.join(trained_model_path, model, 'test_summary.txt')) as file:
                            test_summary = json.load(file)
                            # check if roc_auc is saved, if not compute (in older version was not saved)
                            if "roc_auc" in test_summary:
                                all_micro_auc.append(test_summary['roc_auc']['micro'])
                                all_macro_auc.append(test_summary['roc_auc']['macro'])

                                all_micro_f1.append(test_summary['F1']['micro'])
                                all_macro_f1.append(test_summary['F1']['macro'])

                                all_acc.append(test_summary["accuracy"])
##
# get the first 4 best models in the different metrics
# accuracy
all_best = {"accuracy":[],
            "micro_auc":[],
            "macro_auc":[],
            "micro_f1":[],
            "macro_f1":[]}


metric = np.array(all_acc)
indexes = (-metric).argsort()[:4]
all_best["accuracy"] = [{"model":all_model_names[i], "value" : metric[i]} for i in indexes]

metric = np.array(all_macro_auc)
indexes = (-metric).argsort()[:4]
all_best["macro_auc"] = [{"model":all_model_names[i], "value" : metric[i]} for i in indexes]

metric = np.array(all_micro_auc)
indexes = (-metric).argsort()[:4]
all_best["micro_auc"] = [{"model":all_model_names[i], "value" : metric[i]} for i in indexes]

metric = np.array(all_macro_f1)
indexes = (-metric).argsort()[:4]
all_best["macro_f1"] = [{"model":all_model_names[i], "value" : metric[i]} for i in indexes]

metric = np.array(all_micro_f1)
indexes = (-metric).argsort()[:4]
all_best["micro_f1"] = [{"model":all_model_names[i], "value" : metric[i]} for i in indexes]

# print results
for metric, values in all_best.items():
    print(f'{metric.upper()}')
    print(f'{"-"*10}')
    for idx, m in enumerate(values):
        print(f'    {idx} - {m["model"]:56s}: {m["value"]:2.3f}')
    print(f'{"-"*10}\n')




