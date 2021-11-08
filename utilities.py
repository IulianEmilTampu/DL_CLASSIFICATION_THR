import os
import cv2
import math
import time
import random
import numbers
import itertools
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Model

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

## fix label function

def fix_labels_v2(labels, classification_type, unique_labels, categorical=True):
    '''
    Prepares the labels for training using the specifications in unique_labels.
    This was initially done in the data generator, but given that working with
    TFrecords does not allow some operations, labels have to be fixed here.

    Args:
    labels : numpy array
        Contains the labels for each classification type
    classification_type : str
        Specifies the classification type as described in the
        create_dataset_v2.py script.
    unique_labels (list): list of the wanted labels and their organization
        # For example:
        [
        0,
        [1, 3],
        [2, 4, 5],
        6
        ]

    will return categorical labels where labels are 0, 1, 2 and 3 with:
        - 0 having images from class 0;
        - 1 having images from classes 1 and 3
        - 2 having images from classes 2, 4, 5
        - 3 having images from class 6

    categorical: if True a categorical verison of the labels is returned.
                    If False, the labels are returned as a 1D numpy array.
    '''
    # check inputs
    assert type(labels) is np.ndarray, 'Labels should be np.ndarray. Given {}'.format(type(labels))
    assert type(unique_labels) is list, 'unique_labels should be list. Given {}'.format(tyep(unique_labels))

    if not isinstance(classification_type, str):
        raise TypeError(f'classification_type expected to be a list, but give {type(classification_type)}')
    else:
        # check that it specifies a know classification type
        if not (classification_type=='c1' or classification_type=='c2' or classification_type=='c3'):
            # raise Warning(f'classification_type expected to be c1, c2 or c3. Instead was given {classification_type}\n')
            # print('Setting classification type to c3 to be able to fix the labels. * Check that this is correct!')
            classification_type = 'c3'

    # get the right label list based on the classification type
    if classification_type=='c1':
        labels = labels[:,0]
    elif classification_type=='c2':
        labels = labels[:,1]
    elif classification_type=='c3':
        labels = labels[:,2]

    # get the appropriate label based on the unique label specification
    for idy, label in enumerate(labels):
        for idx, u_label in enumerate(unique_labels):
            if type(u_label) is list:
                for i in u_label:
                    if label == i:
                        labels[idy] = int(idx)
                        # break
            elif type(u_label) is not list:
                if label == u_label:
                    labels[idy] = int(idx)
                    # break
    if categorical == True:
        # convert labels to categorical
        return tf.convert_to_tensor(to_categorical(labels, num_classes=len(unique_labels)), dtype=tf.float32)
    else:
        return tf.convert_to_tensor(labels, dtype=tf.float32)

##

def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot


def get_organized_files(file_names, classification_type,
                        return_labels=True,
                        categorical=False,
                        custom=False,
                        custom_labels=None):
    '''
    Utility that given a list of file names using the convention described in
    the create_dataset_v2.py script, returns three things:
    1 - list of files that does not contain the file marked as to be excluded (9)
    2 - list of labels corresponding to the files above (categprical or not)
    3 - a list of lists that contains the files organised per aggregation class

    Parameters
    ----------
    file_names : list of str
        Identifies the file names
    classification_type : str
        Specifies the classification type as described in the
        create_dataset_v2.py script.
    custom : bool
        Specifies if the labels need to be aggregated in a different way from
        default. The default is that every labels is an independent class. If
        False, the labels will be clastered based on the specifications given in
        the custom_labels. Default is False.
    custom_labels : list
        Specifies the way the labels should be clustered. Used if custom
        parameter is set to True.
        # For example:
        [
        0,
        [1, 3],
        [2, 4, 5],
        6
        ]

    will return categorical labels where labels are 0, 1, 2 and 3 with:
        - 0 having images from class 0;
        - 1 having images from classes 1 and 3
        - 2 having images from classes 2, 4, 5
        - 3 having images from class 6

    categorical : bool
        If True, returns the labels in categorical form.
    '''

    # check that the inputs are correct
    if isinstance(file_names, list):
        # loop through all the elements in the list and make sure they are
        # strings and they match the convention
        for file in file_names:
            # try to get the labels
            c1 = int(file[file.find('c1')+3])
            c2 = int(file[file.find('c2')+3])
            c3 = int(file[file.find('c3')+3])
    else:
        raise TypeError(f'file_name expected to be a list, but give {type(file_names)}')

    if not isinstance(classification_type, str):
        raise TypeError(f'classification_type expected to be a list, but give {type(classification_type)}')
    else:
        if not custom:
            # check that it specifies a know classification type
            if not (classification_type=='c1' or classification_type=='c2' or classification_type=='c3'):
                raise ValueError(f'Not custom classification was set. classification_type expected to be c1, c2 or c3. Instead was given {classification_type}')

    if custom:
        # set lassification type to c3 to be able to get the last label
        classification_type = 'c3'
        # custom label aggregation given, thus checking if custom_labels is given
        if custom_labels:
            # check that is a list
            if not isinstance(custom_labels, list):
                raise TypeError(f'custom_labels expected to be a list, but given {type(custom_labels)}')
        else:
            raise ValueError('custom was set to True, but no custom_labels specification was given.')

    # get labels for the specified classification type and exclude label 9
    # (flags volumes to not be used)
    raw_labels = []
    filtered_file_names = []
    for file in file_names:
        label = int(file[file.find(classification_type)+3])
        if label != 9:
            raw_labels.append(label)
            filtered_file_names.append(file)

    # use custom aggregation
    final_file_names = []
    organized_files = [[] for i in range(len(custom_labels))]
    labels = []
    for f, c in zip(filtered_file_names, raw_labels):
        for idx, l in enumerate(custom_labels):
            if type(l) is list:
                for ll in l:
                    if c == ll:
                        organized_files[idx].append(f)
                        labels.append(idx)
                        final_file_names.append(f)
            elif c == l:
                organized_files[idx].append(f)
                labels.append(idx)
                final_file_names.append(f)


    if categorical == True:
        # convert labels to categorical
        labels = to_categorical(labels, num_classes=np.unique(labels).shape[0])

    return final_file_names, labels, organized_files

## TENSORFLOW DATA GENERATOR

'''
 ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤ TFRecord dataset functions

There are two main functions here:
1 - _parse_function -> opens the TFRecord files using the format used during the
                      creation of the TFRecord files. Whithin this function one
                      can manuputale the data in the record to prepare the images
                      and the labels used by the model e.g. add extra channels or crop.
2 - create_dataset -> this looks at all the TFRecords files specified in the dataset
                      and retrievs, shuffles, buffers the data for the model.
                      One here can even implement augmentation if needed. This
                      returns a dataset that the model will use (image, label) format.
 The hyper-parameters needed for the preparation of the dataset are:
- batch_size: how many samples at the time should be fed into the model.
- number of parallel loaders: how many files are read at the same time.
- buffer_size: number of samples (individual subjects) that will be used for the
              shuffling procedure!
'''
def _parse_function_2D(proto, crop_size):
  '''
  Parse the TFrecord files. In this function one can change the 'structure' of
  the input or output based on the model requirements. Note that no boolean
  operators are accepted if the function should be ingluded in the graph.
  '''

  key_features = {
    'xdim' : tf.io.FixedLenFeature([], tf.int64),
    'zdim' : tf.io.FixedLenFeature([], tf.int64),
    'nCh'  : tf.io.FixedLenFeature([], tf.int64),
    'image' : tf.io.FixedLenFeature([], tf.string),
    'label_c1' : tf.io.FixedLenFeature([], tf.int64),
    'label_c2' : tf.io.FixedLenFeature([], tf.int64),
    'label_c3' : tf.io.FixedLenFeature([], tf.int64)
  }
  parsed_features = tf.io.parse_single_example(proto, key_features)

  # parse input dimentions
  xdim = parsed_features['xdim']
  zdim = parsed_features['zdim']
  nCh = parsed_features['nCh']
  # parse image
  image = tf.io.parse_tensor(parsed_features['image'], out_type=tf.float32)
  image = tf.reshape(image, shape=[xdim,zdim,nCh])
  image = tf.image.crop_to_bounding_box(tf.expand_dims(image, axis=0), 0, 0, crop_size[0], crop_size[1])

  # parse lable
  c1 = parsed_features['label_c1']
  c2 = parsed_features['label_c2']
  c3 = parsed_features['label_c3']

  return tf.squeeze(image, axis=0), [c1, c2, c3]

def preprocess_augment(dataset):
    image, label = dataset[0], dataset[1]
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return image, label

'''
uset the above to create the dataset
'''
def TFR_2D_dataset(filepath, dataset_type, batch_size, buffer_size=100, crop_size=(200, 200), classification_type=None, unique_labels=None):
    # point to the files of the dataset
    dataset = tf.data.TFRecordDataset(filepath)
    # parse sample
    dataset = dataset.map(lambda x: _parse_function_2D(x, crop_size=crop_size),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # shuffle the training dataset
    if dataset_type == 'train':
        dataset = dataset.shuffle(buffer_size=buffer_size)

    # set bach_size
    dataset = dataset.batch(batch_size=batch_size)

    # # augmentation
    # if dataset_type == 'train':
    #     dataset = dataset.map(lambda x: preprocess_augment(),
    #             num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # prefetch batches
    dataset = dataset.prefetch(100)

    return dataset


## METRICS

def acc_metric(pred, y):
    '''
    computes the accuracy number of correct classified / total number of samples
    INPUT
     - pred: numpy array of categoricals [B, H, W, C]
     - y : numpyarray of ground truth. This can be both categorical or
            the argmax of the ground truth categorical [B, H, W] or [B, H, W, C]
    '''
    if y.squeeze().shape != pred.squeeze().shape:
        # here we have the prediction in categorical while the gt in argmax
        return (pred.squeeze().argmax(axis=-1) == y.squeeze()).astype(float).mean()
    else:
        # both are categoricals
        return (pred.squeeze().argmax(axis=-1) == y.squeeze().argmax(axis=-1)).astype(float).mean()

## TIME

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

    # form a string in the format d:h:m:s
    # return str(days)+delimiter+str(hours)+delimiter+str(minutes)+delimiter+str(round(seconds,0))
    return "%2dd:%02dh:%02dm:%02ds:%02dms" % (days, hours, minutes, seconds, milliseconds)

def tictoc_from_time(elapsed=1):
    '''
    # Returns a string that contains the number of days, hours, minutes and
    seconds given the elapsed time
    '''
    days, rem = np.divmod(elapsed, 86400)
    hours, rem = np.divmod(rem, 3600)
    minutes, rem = np.divmod(rem, 60)
    seconds, rem = np.divmod(rem, 1)
    milliseconds = rem*1000

    # form a string in the format d:h:m:s
    # return str(days)+delimiter+str(hours)+delimiter+str(minutes)+delimiter+str(round(seconds,0))
    return "%2dd:%02dh:%02dm:%02ds:%02dms" % (days, hours, minutes, seconds, milliseconds)

## CONFISION MATRIX

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
    plt.xticks(tick_marks, classes, rotation=45, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    thresh = cm.max()/2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i,j],
            horizontalalignment='center',
            verticalalignment='center',
            color='white' if cm[i,j] > thresh else 'black',
            fontsize=25)

    # plt.tight_layout()
    plt.ylabel('True label', fontsize=15)
    plt.xlabel('Prediction', fontsize=15)

    acc = 100*(np.trace(cm) / np.sum(cm))
    plt.title('Confusion matrix -> ' + 'Accuracy {:05.2f}'.format(acc), fontsize=20)
    fig.tight_layout()

    # save is needed
    if savePath is not None:
        if os.path.isdir(savePath):
            fig.savefig(os.path.join(savePath, 'ConfisionMatrix_ensemble_prediction.pdf'), bbox_inches='tight', dpi = 100)
            fig.savefig(os.path.join(savePath, 'ConfisionMatrix_ensemble_prediction.png'), bbox_inches='tight', dpi = 100)
        else:
            raise ValueError('Invalid save path: {}'.format(os.path.join(savePath, 'ConfisionMatrix_ensemble_prediction.pdf')))

    if draw is True:
        plt.draw()
    else:
        plt.close()

    return acc

## PLOT ROC

def plotROC(GT, PRED, classes, savePath=None, draw=False):
    from sklearn.metrics import roc_curve, auc
    from itertools import cycle
    from scipy import interp
    from tensorflow.keras.utils import to_categorical
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
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
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

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
            color='deeppink', linestyle=':', linewidth=4)

    ax.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    colors = cycle(['blue', 'orange', 'green', 'red','purple','brown','pink','gray','olive','cyan','teal'])
    for i, color in zip(range(n_classes), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {} (area = {:0.2f})'
                ''.format(classes[i], roc_auc[i]))

    ax.plot([0, 1], [0, 1], 'k--', lw=lw)

    major_ticks = np.arange(0, 1, 0.1)
    minor_ticks = np.arange(0, 1, 0.05)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    plt.grid(color='b', linestyle='-.', linewidth=0.1, which='both')

    ax.set_xlabel('False Positive Rate', fontsize=25)
    ax.set_ylabel('True Positive Rate', fontsize=25)
    ax.set_title('Multi-class ROC (OneVsAll)', fontsize=20)
    plt.legend(loc="lower right", fontsize=12)

    # ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤ work on the zummed-in image
    colors = cycle(['blue', 'orange', 'green', 'red','purple','brown','pink','gray','olive','cyan','teal'])
    axins = zoomed_inset_axes(ax, zoom=1, loc=7, bbox_to_anchor=(0,0,0.99,0.9), bbox_transform=ax.transAxes)

    axins.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    axins.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    for i, color in zip(range(n_classes), colors):
        axins.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {} (area = {:0.2f})'
                ''.format(classes[i], roc_auc[i]))

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

    # save is needed
    if savePath is not None:
        if os.path.isdir(savePath):
            fig.savefig(os.path.join(savePath, 'Multiclass ROC.pdf'), bbox_inches='tight', dpi = 100)
            fig.savefig(os.path.join(savePath, 'Multiclass ROC.png'), bbox_inches='tight', dpi = 100)
        else:
            raise ValueError('Invalida save path: {}'.format(savePath))

    if draw is True:
        plt.draw()
    else:
        plt.close()

    return fpr, tpr, roc_auc

## PLOR PR (precision and recall) curves

def plotPR(GT, PRED, classes, savePath=None, draw=False):
    from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
    from sklearn.metrics import average_precision_score
    from itertools import cycle
    from scipy import interp
    from tensorflow.keras.utils import to_categorical
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset

    '''
    Funtion that plots the PR (precision and recall) curve given the ground truth and the logits prediction

    INPUT
    - GT: true labels
    - PRED: array of float the identifies the logits prediction
    - classes: list of string that identifies the labels of each class
    - save path: sting that identifies the path where to save the ROC plots
    - draw: bool if to print or not the ROC curve

    RETURN
    - precision: dictionary that contains the precision every class and micro average
    - recall: dictionary that contains the recall for every class and micro average
    - average_precision: float of the average precision
    - F1: dictionare containing the micro and marco average f1-score
    '''
    # define variables
    precision = dict()
    recall = dict()
    average_precision = dict()
    n_classes = len(classes)
    lw = 2 # line width

    # ¤¤¤¤¤¤¤¤¤¤¤ f1_score
    F1 = {
        'micro':f1_score(GT, np.argmax(PRED, axis=-1), average='micro'),
        'macro':f1_score(GT, np.argmax(PRED, axis=-1), average='macro')
    }
    print('F1-score (micro and macro): {0:0.2f} and {0:0.2f}'.format(F1['micro'], F1['macro']))

    # make labels categorical
    GT = to_categorical(GT, num_classes=n_classes)

    # ¤¤¤¤¤¤¤¤¤¤¤ micro-average roc
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(GT[:,i], PRED[:,i])
        average_precision[i] = average_precision_score(GT[:,i], PRED[:,i])


    # Compute micro-average ROC curve and ROC area
    precision["micro"], recall["micro"], _ = precision_recall_curve(GT.ravel(), PRED.ravel())
    average_precision["micro"] = average_precision_score(GT, PRED, average='micro')
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))


    # Plot all PR curves and save

    # create iso-f1 curves and plot on top the PR curves for every class
    fig, ax = plt.subplots(figsize=(10,10))

    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for idx, f_score in enumerate(f_scores):
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        if idx == 0:
            l, = ax.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2, label='iso-f1 curves')
        else:
            l, = ax.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        ax.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    # labels.append('iso-f1 curves')
    l, = ax.plot(recall["micro"], precision["micro"], color='gold', lw=lw,
                        label='micro-average Precision-recall (area = {0:0.2f})'.format(average_precision["micro"]))
    lines.append(l)
    # labels.append('micro-average Precision-recall (area = {0:0.2f})'.format(average_precision["micro"]))

    colors = cycle(['blue', 'orange', 'green', 'red','purple','brown','pink','gray','olive','cyan','teal'])
    for i, color in zip(range(n_classes), colors):
        ax.plot(recall[i], precision[i], color=color, lw=lw,
                label='Precision-recall curve of class {:9s} (area = {:0.2f})'
                ''.format(classes[i], average_precision[i]))

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('True positive rate - Recall [TP/(TP+FN)]', fontsize=20)
    ax.set_ylabel('Positive predicted value - Precision [TP/(TP+TN)]', fontsize=20)
    ax.set_title('Multi-class Precision-recall curve', fontsize=20)
    plt.legend(loc="lower right", fontsize=12)

    # save is needed
    if savePath is not None:
        if os.path.isdir(savePath):
            fig.savefig(os.path.join(savePath, 'Multiclass PR.pdf'), bbox_inches='tight', dpi = 100)
            fig.savefig(os.path.join(savePath, 'Multiclass PR.png'), bbox_inches='tight', dpi = 100)
        else:
            raise ValueError('Invalida save path: {}'.format(savePath))

    if draw is True:
        plt.draw()
    else:
        plt.close()

    return precision, recall, average_precision, F1


## OTHER PLOTTING

# Helper function to show a batch
def show_batch_2D(sample_batched, title=None, img_per_row=10):

    from mpl_toolkits.axes_grid1 import ImageGrid
    """
    Creates a grid of images with the samples contained in a batch of data.

    Parameters
    ----------
    sample_batch : tuple
        COntains the actuall images (sample_batch[0]) and their label
        (sample_batch[1]).
    title : str
        Title of the created image
    img_per_row : int
        number of images per row in the created grid of images.
    """
    batch_size = len(sample_batched[0])
    nrows = batch_size//img_per_row
    if batch_size%img_per_row > 0:
        nrows += 1
    ncols = img_per_row

    # make figure grid
    fig = plt.figure(figsize=(4., 4.))
    grid = ImageGrid(fig, 111,
                    nrows_ncols=(nrows, ncols),
                    axes_pad=0.3,  # pad between axes in inch.
                    label_mode='L',
                    )

    # fill in the axis
    for i in range(batch_size):
        img = np.squeeze(sample_batched[0][i,:,:])
        grid[i].imshow(img, cmap='gray', interpolation=None)
        grid[i].set_xticks([])
        grid[i].set_yticks([])
        grid[i].set_title(sample_batched[1][i])
    if title:
        fig.suptitle(title, fontsize=20)
    else:
        fig.suptitle('Batch of data', fontsize=20)
    plt.show()

# Helper function to show a batch
def show_batch_2D_with_histogram(sample_batched, title=None):

    """
    Creates a grid of images with the samples contained in a batch of data.
    Here showing 5 random examples along with theirplt histogram.

    Parameters
    ----------
    sample_batch : tuple
        COntains the actuall images (sample_batch[0]) and their label
        (sample_batch[1]).
    title : str
        Title of the created image
    img_per_row : int
        number of images per row in the created grid of images.
    """
    n_images_to_show = 5
    random.seed(29092019)
    index_samples = random.sample(range(len(sample_batched[0])), n_images_to_show)

    # make figure grid
    fig , ax = plt.subplots(nrows=2, ncols=n_images_to_show, figsize=(15,10))

    # fill in the axis with the images and histograms
    for i, img_idx in zip(range(n_images_to_show),index_samples) :
        img = np.squeeze(sample_batched[0][img_idx,:,:])
        ax[0][i].imshow(img, cmap='gray', interpolation=None)
        ax[0][i].set_xticks([])
        ax[0][i].set_yticks([])
        ax[0][i].set_title(sample_batched[1][img_idx])

        # add histogram
        bins = np.histogram(img, bins=100)[1] #get the bin edges
        ax[1][i].hist(img.flatten(), bins=bins)
        # ax[1][i].set_xlim([-1.1,1.1])
    if title:
        fig.suptitle(title, fontsize=20)
    else:
        fig.suptitle('Batch of data', fontsize=20)

    plt.show()

##

'''
Grad-CAM implementation [1] as described in post available at [2].

[1] Selvaraju RR, Cogswell M, Das A, Vedantam R, Parikh D, Batra D. Grad-cam:
    Visual explanations from deep networks via gradient-based localization.
    InProceedings of the IEEE international conference on computer vision 2017
    (pp. 618-626).

[2] https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/

'''

class gradCAM:
    def __init__(self, model, classIdx, layerName=None, use_image_prediction=True, ViT=False, debug=False):
        '''
        model: model to inspect
        classIdx: index of the class to ispect
        layerName: which layer to visualize
        '''
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        self.debug = debug
        self.use_image_prediction = use_image_prediction
        self.is_ViT = ViT

        # if the layerName is not provided, find the last conv layer in the model
        if self.layerName is None:
            self.layerName = self.find_target_layer()
        else:
            if self.debug is True:
                print('GradCAM - using layer {}'.format(self.model.get_layer(self.layerName).name))

    def find_target_layer(self):
        '''
        Finds the last convolutional layer in the model by looping throught the
        available layers
        '''
        for layer in reversed(self.model.layers):
            # check if it is a 2D conv layer (which means that needs to have
            # 4 dimensions [batch, width, hight, channels])
            if len(layer.output_shape) == 4:
                # check that is a conv layer
                if layer.name.find('conv') != -1:
                    if self.debug is True:
                        print('GradCAM - using layer {}'.format(layer.name))
                    return layer.name

        if self.layerName is None:
            # if no convolutional layer have been found, rase an error since
            # Grad-CAM can not work
            raise ValueError('Could not find a 4D layer. Cannot apply GradCAM')

    def compute_heatmap(self, image, eps=1e-6):
        '''
        Compute the L_grad-cam^c as defined in the original article, that is the
        weighted sum over feature maps in the given layer with weights based on
        the importance of the feature map on the classsification on the inspected
        class.

        This is done by supplying
        1 - an input to the pre-trained model
        2 - the output of the selected conv layer
        3 - the final softmax activation of the model
        '''
        # this is a gradient model that we will use to obtain the gradients from
        # with respect to an image to construct the heatmaps
        gradModel = Model(
                inputs=[self.model.inputs],
                outputs=[self.model.get_layer(self.layerName).output,
                self.model.output])

        # replacing softmax with linear activation
        gradModel.layers[-1].activation = tf.keras.activations.linear

        if self.debug is True:
            gradModel.summary()

        # use the tensorflow gradient tape to store the gradients
        with tf.GradientTape() as tape:
            '''
            cast the image tensor to a float-32 data type, pass the
            image through the gradient model, and grab the loss
            associated with the specific class index.
            '''
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            # check if the prediction is a list (VAE)
            if type(predictions) is list:
                # the model is a VEA, taking only the prediction
                predictions = predictions[4]
            pred = tf.argmax(predictions, axis=1)
            loss = predictions[:, self.classIdx]

        grads = tape.gradient(loss, convOutputs)
        # sometimes grads becomes NoneType
        if grads is None:
            grads = tf.zeros_like(convOutputs)
        '''
        compute the guided gradients.
         - positive gradients if the classIdx matches the prediction (I want to
            know which values make the probability of that class to be high)
         - negative gradients if the classIdx != the predicted class (I want to
            know which gradients pushed down the probability for that class)
        '''
        if self.use_image_prediction == True:
            if self.classIdx == pred:
                castConvOutputs = tf.cast(convOutputs > 0, tf.float32)
                castGrads = tf.cast(grads > 0, tf.float32)
            else:
                castConvOutputs = tf.cast(convOutputs <= 0, tf.float32)
                castGrads = tf.cast(grads <= 0, tf.float32)
        else:
            castConvOutputs = tf.cast(convOutputs > 0, tf.float32)
            castGrads = tf.cast(grads > 0, tf.float32)
        guidedGrads = castConvOutputs * castGrads * grads

        # remove teh batch dimension
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        # compute the weight value for each feature map in the conv layer based
        # on the guided gradient
        weights = tf.reduce_mean(guidedGrads, axis=(0,1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        # now that we have the astivation map for the specific layer, we need
        # to resize it to be the same as the input image
        if self.is_ViT:
            dim = int(np.sqrt(cam.shape[0]))
            (w, h) = (image.shape[2], image.shape[1])
            heatmap = cam.numpy().reshape((dim, dim))
            heatmap = cv2.resize(heatmap,(w, h))
        else:
            (w, h) = (image.shape[2], image.shape[1])
            heatmap = cv2.resize(cam.numpy(),(w, h))

        # normalize teh heat map in [0,1] and rescale to [0, 255]
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = (numer/denom)
        heatmap_raw = (heatmap * 255).astype('uint8')

        # create heatmap based ont he colormap setting
        heatmap_rgb = cv2.applyColorMap(heatmap_raw, cv2.COLORMAP_VIRIDIS).astype('float32')

        return heatmap_raw, heatmap_rgb

    def overlay_heatmap(self, heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_VIRIDIS):

        # create heatmap based ont he colormap setting
        heatmap = cv2.applyColorMap(heatmap, colormap).astype('float32')

        if image.shape[-1] == 1:
            # convert image from grayscale to RGB
            image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB).astype('float32')

        output = cv2.addWeighted(image, alpha, heatmap, (1 - alpha), 0)

        # return both the heatmap and the overlayed output
        return (heatmap, output)

## normalization

def data_normalization(dataset, quantile=0.995):
    '''
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
    '''

    x_min = np.quantile(dataset, 1-quantile)
    x_max = np.quantile(dataset, quantile)

    return 2.0*(dataset.astype('float32') - x_min)/(x_max - x_min) - 1.0

