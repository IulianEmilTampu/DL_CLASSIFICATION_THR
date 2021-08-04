import os
import time
import math
import numbers
import itertools
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.metrics import confusion_matrix


import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torchvision import transforms, utils

import tensorflow as tf

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")



class OCTDataset2D_classification(Dataset):
    '''Creates a pytorch compatible dataset of OCT images. Data is expected to
    be in nifti format such that the loader can load it.'''

    def __init__(self, image_paths, unique_labels, augmentor=None, transform=None):
        '''
        Args:
            image_paths (list): list of all the files to be included in the dataset.
                                Note that the file name specifies the class to which
                                the image belongs to. The file name in in the format

                                some-itenfities_label_number.nii.gz
                                e.g. TH01_0001_0001_label_1

            unique_labels (list): list of the wanted labels and their organization
                For example:
                [
                'class_0',
                ['class_1', 'class_3'],
                ['class_2', 'class_4', 'class_5'],
                'class_6'
                ]

                will return tuple of image, labels where labels are 0, 1, 2 or 3 with:
                - 0 having images from class_0;
                - 1 having images from class_1 and class_3
                - 2 having images from class_2, class_4, class_5
                - 3 having images from class_6
            transform (callable, optional): Optional transform to be applied
            on a sample.
            seed: set in case need for deterministic dataset creation

        returns image, label
        '''
        self.image_paths = image_paths
        self.unique_labels = unique_labels
        self.transform = transform
        self.augmentor = augmentor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        '''
        This retrieves one sample from the dataset
        Args
            idx: index of the element in the dataset to retrieve
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = nib.load(self.image_paths[idx]).get_fdata().astype('float32')
        # intensity normalization [0,1] in the ergion of the image that is not -1 (interpolation)
        image[image!=-1] = (image[image!=-1]  - np.min(image[image!=-1] ))/(np.max(image[image!=-1] ) - np.min(image[image!=-1] ))

        # work on the label
        start = os.path.basename(self.image_paths[idx]).index('label_') + len('label_')
        end = os.path.basename(self.image_paths[idx]).index('.nii.gz')
        label = 'class_'+ os.path.basename(self.image_paths[idx])[start:end]

        # get the appropriate label based on the unique label specification
        for idx, u_label in enumerate(self.unique_labels):
            if type(u_label) is list:
                for i in u_label:
                    if label == i:
                        label = int(idx)
                        break
            elif type(u_label) is not list:
                if label == u_label:
                    label = int(idx)
                    break

        # apply augmentation if needed
        if self.augmentor:
            # note that the augmentor need imput in the channel first convention
            image = self.augmentor(images=image[np.newaxis,:,:])
            image = image.squeeze()

        # apply transformations if needed
        if self.transform:
            image, label  = self.transform((image, label))

        return image, label

class RandomCrop2D(object):
    '''
    Crop randomply the sample (image, label) in sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    '''

    def __init__(self, output_size, seed):
        assert isinstance(output_size, (int, tuple)), 'Invalid output size. Expected an int or tuple'
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2 , 'Invalid input size. Expected a tuple of len=3'
            self.output_size = output_size

        self.seed = seed

    def __call__(self, sample):
        '''
        Performing the actuall rescale
        '''
        # open sample into image and label
        image = sample[0]
        label = sample[1]
        h, w = image.shape
        new_h, new_w = self.output_size

        # check that the requred input is valid
        assert h >= new_h, 'Invalid crop size: old {}, new {}'.format(h, new_h)
        assert w >= new_w, 'Invalid crop size: old {}, new {}'.format(w, new_w)

        # make deterministic if required
        if self.seed is not None:
            assert isinstance(self.seed, int) , 'Invalid seed value. Expected integer'
            np.random.seed(self.seed)


        top_corner = np.random.randint(0, h - new_h + 1)
        left_corner = np.random.randint(0, w - new_w + 1)

        image = image[top_corner:top_corner + new_h,
                       left_corner:left_corner + new_w]
        label = label[top_corner:top_corner + new_h,
                       left_corner:left_corner + new_w]

        return image, label

class ChannelFix(object):
    '''
    Returns image and labes with channel last or channel first convention

    Args:
        channel_order (string): 'last' [X, Y, C] or 'first' [C, X, Y]
    '''
    def __init__(self, channel_order='last'):
        super().__init__()
        self.channel_order =  channel_order

    def __call__(self, sample):
        # fix channel order
        if self.channel_order == 'last':
            # return numpy [X, Y, C]
            image = np.expand_dims(sample[0], axis=-1)
            # check if working on both image and label
            if type(sample[1]) == int:
                label = sample[1]
            else:
                label = np.expand_dims(sample[1], axis=-1)
            return image, label
        elif self.channel_order == 'first':
            # return numpy [C, X, Y]
            image = np.expand_dims(sample[0], axis=0)

            # check if working on both image and label
            if type(sample[1]) == int:
                label = sample[1]
            else:
                label = np.expand_dims(sample[1], axis=-1)
            return image, label
        else:
            raise TypeError('Invalid convention. given {} but expecting last or first'.format(self.channel_order))

class RandomHorizontalFlip(torch.nn.Module):
    """Horizontally flip the given image randomly with a given probability.
    The image can be a PIL Image or a torch Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, sample):
        """
        Args:
            sample (a Tensor): sample to be flipped (image, label).

        Returns:
            aTensor: Randomly flipped image, label.
        """
        if torch.rand(1) < self.p:
            # check if working on both image and label
            if type(sample[1]) == int:
                return transforms.functional.hflip(sample[0]), sample[1]
            else:
                return transforms.functional.hflip(sample[0]), transforms.functional.hflip(sample[1])

        return sample[0], sample[1]

class RandomRotate(torch.nn.Module):
    """Clockwise rotation by random angle in the given range. Rotation is
    performed with given probability p

    Args:
        p (float): probability of the image being rotated. Default value is 0.5
        angle (tuple): range of angles for being rotated. Default is (0,90)
    """

    def __init__(self, p=0.5, angle=(0,90)):
        super().__init__()
        self.p = p
        self.angle = angle

    def forward(self, sample):
        """
        Args:
            sample (a Tensor): sample to be rotated (image, label).

        Returns:
            aTensor: Randomly flipped image, label.
        """
        if torch.rand(1) < self.p:
            ang = float(((self.angle[0] - self.angle[1]) * torch.rand(1) + self.angle[1])[0].numpy())
            image = transforms.functional.rotate(sample[0], angle=ang)

            # check if working on both image and label
            if type(sample[1]) == int:
                label = sample[1]
            else:
                label = transforms.functional.rotate(sample[1], angle=ang)
            return image, label

        return sample[0], sample[1]



def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot

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
    'label' : tf.io.FixedLenFeature([], tf.int64)
  }
  parsed_features = tf.io.parse_single_example(proto, key_features)

  # parse input dimentions
  xdim = parsed_features['xdim']
  zdim = parsed_features['zdim']
  nCh = parsed_features['nCh']
  # parse image
  image = tf.io.parse_tensor(parsed_features['image'], out_type=tf.float32)
  image = tf.reshape(image, shape=[xdim,zdim,nCh])
  # image = tf.image.resize_with_crop_or_pad(tf.expand_dims(image, axis=0), crop_size[0], crop_size[1])
  image = tf.image.resize(tf.expand_dims(image, axis=0), [crop_size[0], crop_size[1]])

  # normalize
  image = tf.image.per_image_standardization(image)
  # image = (image - tf.math.reduce_mean(image)) / tf.math.reduce_std(image)
  # image = 2.0 * (image - tf.math.reduce_min(image))/(tf.math.reduce_max(image) - tf.math.reduce_min(image)) - 1.0
  # image = (image - tf.math.reduce_min(image))/(tf.math.reduce_max(image) - tf.math.reduce_min(image))

  # augment
  image = tf.image.random_flip_up_down(image)
  image = tf.image.random_flip_left_right(image)

  # parse lable
  label = parsed_features['label']

  return tf.squeeze(image, axis=0), label

def preprocess_augment(dataset):
    image, label = dataset[0], dataset[1]
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return image, label

'''
uset the above to create the dataset
'''
def TFR_2D_dataset(filepath, dataset_type, batch_size, buffer_size=100, crop_size=(250, 250)):
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

##

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
            raise ValueError('Invalida save path: {}'.format(savePath))

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
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
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


## PLOTTING

# Helper function to show a batch
def show_batch_2D(sample_batched, title=''):
    """
    Show sample image (z,x) for a batch of oct images.
    """
    batch_size = len(sample_batched[0])
    nrows = batch_size//10
    if batch_size%3 > 0:
        nrows += 1

    # convert to pythorch if numpy
    if type(sample_batched[0]) is np.ndarray:
        # numpy arrays are given, convert to pytorch tensors with [B,C,D,W] convention
        x = torch.tensor(sample_batched[0]).permute(0,3,1,2)
        y = torch.tensor(sample_batched[1])
        sample_batched = (x, y)

    image_grid = utils.make_grid(sample_batched[:][0], nrow=nrows)

    if not list(sample_batched[1][1].size()):
        # print('Empty {}'.format(list(sample_batched[1][1].size())))
        label_grid = [int(x) for x in sample_batched[:][1]]
    else:
        # print('Not empty {}'.format(list(sample_batched[1][1].size())))
        label_grid = utils.make_grid(sample_batched[:][1], nrow=nrows)

    # reshape from [C, D, W] to [D, W, C]
    plt.imshow(image_grid.numpy().transpose(1,2,0)[:,:,0], cmap='gray', interpolation=None)

    if not list(sample_batched[1][1].size()):
        plt.title(label_grid)
    else:
        plt.imshow(label_grid.numpy().transpose(1,2,0)[:,:,0], cmap = 'Dark2', norm=colors.Normalize(vmin=0, vmax=4), alpha=0.5)
        plt.title('Batch from dataloader')
    plt.show()

## CHECK OCT DATASET
def check_dataset(dataset_folder):
    '''
    Ausiliary function that checks that the dataset provided by the user is in
    required format:
    .../dataset_folder/
    ├── Train
    │   ├── class_1
    │   │   ├── TH01_0001_0001_label_1.nii.gz
    │   │   ├── TH01_0002_0001_label_1.nii.gz
    │   │   ├── TH01_0003_0001_label_1.nii.gz
    │   ├── class_2
    │   │   ├── TH01_0001_0001_label_2.nii.gz
    │   │   ├── TH01_0002_0001_label_2.nii.gz
    │   │   ├── TH01_0003_0001_label_2.nii.gz
    │   ├── ...
    ├── Test
    │   ├── class_1
    │   │   ├── TH02_0001_0001_label_1.nii.gz
    │   │   ├── TH02_0002_0001_label_1.nii.gz
    │   │   ├── TH02_0003_0001_label_1.nii.gz
    │   ├── class_2
    │   │   ├── TH03_0001_0001_label_2.nii.gz
    │   │   ├── TH03_0002_0001_label_2.nii.gz
    │   │   ├── TH03_0003_0001_label_2.nii.gz
    │   ├── ...

    Steps:
    1 - check that the main folder exists
    2 - check that the Train, Test and dataset_info.json exist
    3 - in each Train and Test, check the subfolders and that these ara named
        class_integer-value
    4 - check that each file in a sfecific class_x is a nifti file and ends with
        label_x
    '''
    # 1 check main folder
    if not os.path.isdir(dataset_folder):
        print('Dataset folder does not exist. Given {}'.format(dataset_folder))
        sys.exit()
    else:
        # 2 - check the existance of Test, Train and dataset_info.json
        if not os.path.isfile(os.path.join(dataset_folder, 'dataset_info.json')):
            print('dataset infromation file not found. Input a valid dataset directory. Given {}'.format(dataset_folder))
            sys.exit()
        if not os.path.isdir(os.path.join(dataset_folder, 'Train')):
            print('Train folder not found. Input a valid dataset directory. Given {}'.format(dataset_folder))
            sys.exit()
        if not os.path.isdir(os.path.join(dataset_folder, 'Test')):
            print('Test folder not found. Input a valid dataset directory. Given {}'.format(dataset_folder))
            sys.exit()
        # 3 - in Train and Test, check that each subfolder is named correctly -> class_x where x is an integer
        folder1 = next(os.walk(dataset_folder))[1]
        for f1 in folder1:
            folder2 = next(os.walk(os.path.join(dataset_folder, f1)))[1]
            for f2 in folder2:
                # check folder name
                if (f2[0:6] == 'class_' and f2[6::].isdigit()):
                    # all in order, check all files
                    files = next(os.walk(os.path.join(dataset_folder, f1, f2)))[2]
                    for f3 in files:
                        # check that are nifti files and that the the file name
                        # ends with label_x
                        if not (f3[-7::]=='.nii.gz' and f3[f3.index('label_')+len('label_'):-7]==f2[6::]):
                            print('Found invalid file in {} folder. {}'.format(f2, f3))
                            sys.exit()

                else:
                    print('Invalid class folder. Found {} in {}.'.format(f2, f1))
                    sys.exit()
        # all good.
        print('Dataset folder checked. All good!')

## fix labels for TF base dataset

def fix_labels(labels, unique_labels, categorical=True):
    '''
    Prepares the labels for training using the specifications in unique_labels.
    This was initially done in the data generator, but given that working with
    TFrecords does not allow some operations, labels have to be fixed here.

    Args:
    labels: numpy array containing the labels
    unique_labels (list): list of the wanted labels and their organization
        For example:
        [
        'class_0',
        ['class_1', 'class_3'],
        ['class_2', 'class_4', 'class_5'],
        'class_6'
        ]

    will return categorical labels where labels are 0, 1, 2 or 3 with:
        - 0 having images from class_0;
        - 1 having images from class_1 and class_3
        - 2 having images from class_2, class_4, class_5
        - 3 having images from class_6

    categorical: if True a categorical verison of the labels is returned.
                    If False, the labels are returned as a 1D numpy array.
    '''
    # check inputs
    assert type(labels) is np.ndarray, 'Labels should be np.ndarray. Given {}'.format(type(labels))
    assert type(unique_labels) is list, 'unique_labels should be list. Given {}'.format(tyep(unique_labels))

    # get the appropriate label based on the unique label specification
    for idy, label in enumerate(labels):
        for idx, u_label in enumerate(unique_labels):
            if type(u_label) is list:
                for i in u_label:
                    if label == int(i[-1]):
                        labels[idy] = int(idx)
                        # break
            elif type(u_label) is not list:
                if label == int(u_label[-1]):
                    labels[idy] = int(idx)
                    # break
    if categorical == True:
        # convert labels to categorical
        return to_categorical(labels, num_classes=len(unique_labels))
    else:
        return labels