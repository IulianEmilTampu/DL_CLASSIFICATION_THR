'''
General methods used by the models created in the models_tf.py
'''

import glob # Unix style pathname pattern expansion
import os
import sys
import json
import time
import random
import numpy as np
import matplotlib.pyplot as plt


import tensorflow as tf
from tensorflow.keras.utils import to_categorical



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

    will return categorical labels where labels are 0, 1, 2 and 3 with:
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


def leraningRateScheduler(lr_start, current_epoch, max_epochs, power):
    '''
    Implements a polynimial leraning rate decay based on the:
    - lr_start: initial learning rate
    - current_epoch: current epoch
    - max_epochs: max epochs
    - power: polynomial power
    '''

    decay = (1 - (current_epoch / float(max_epochs))) ** power
    return lr_start * decay


def accuracy(y_true, y_pred):
    '''
    Computes the accuracy number of correct classified / total number of samples
    '''
    count = tf.reduce_sum(tf.cast(tf.argmax(y_true, -1) == tf.argmax(y_pred, -1),dtype=tf.int64))
    total = tf.cast(tf.reduce_sum(y_true),dtype=tf.int64)
    return count/total

def plotModelPerformance(tr_loss, tr_acc, val_loss, val_acc, save_path, display=False):
    '''
    Saves training and validation curves.
    INPUTS
    - tr_loss: training loss history
    - tr_acc: training accuracy history
    - val_loss: validation loss history
    - val_acc: validation accuracy history
    - save_path: path to where to save the model
    '''

    fig, ax1 = plt.subplots(figsize=(15, 10))
    colors = ['blue', 'orange', 'green', 'red','purple','brown','pink','gray','olive','cyan','teal']
    line_style = [':', '-.', '--', '-']
    ax1.set_xlabel('Epochs', fontsize=15)
    ax1.set_ylabel('Loss', fontsize=15)
    l1 = ax1.plot(tr_loss, colors[0], ls=line_style[0])
    l2 = ax1.plot(val_loss, colors[1], ls=line_style[1])
    plt.legend(['Training loss', 'Validation loss'])

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel('Accuracy', fontsize=15)
    ax2.set_ylim(bottom=0, top=1)
    l3 = ax2.plot(tr_acc, colors[2], ls=line_style[2])
    l4 = ax2.plot(val_acc, colors[3], ls=line_style[3])

    # add legend
    lns = l1+l2+l3+l4
    labs = ['Training loss', 'Validation loss', 'Training accuracy', 'Validation accuracy']
    ax1.legend(lns, labs, loc=7, fontsize=15)

    ax1.set_title('Training loss and accuracy trends', fontsize=20)
    ax1.grid()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.savefig(os.path.join(save_path, 'perfomance.pdf'), bbox_inches='tight', dpi = 100)
    fig.savefig(os.path.join(save_path, 'perfomance.png'), bbox_inches='tight', dpi = 100)
    plt.close()

    if display is True:
        plt.show()
    else:
        plt.close()

def plotLearningRate(lr_history, save_path, display=False):
    '''
    Plots and saves a figure of the learning rate history
    '''
    fig = plt.figure(figsize=(20,20))
    plt.plot(lr_history)
    plt.title('Learning rate trend')
    plt.legend(['Learning rate'])
    fig.savefig(os.path.join(save_path, 'learning_rate.pdf'), bbox_inches='tight', dpi = 100)
    fig.savefig(os.path.join(save_path, 'learning_rate.png'), bbox_inches='tight', dpi = 100)

    if display is True:
        plt.show()
    else:
        plt.close()


def plotVAEreconstruction(original, reconstructed, epoch, save_path, display=False):
    '''
    Plots the reconstructed imageg from the VAE along with the original input image.
    '''

    fig , ax = plt.subplots(nrows=3, ncols=2, figsize=(15,10))
    ax[0][0].imshow(original[0,:,:,0], cmap='gray', interpolation=None)
    ax[0][0].set_title('Mean {:.03f}, min {:.03f}, max {:.03f}'
            .format(original[0,:,:,0].mean(),
                    original[0,:,:,0].min(),
                    original[0,:,:,0].max()))
    ax[0][0].set_xticks([])
    ax[0][0].set_yticks([])

    ax[0][1].imshow(original[1,:,:,0], cmap='gray', interpolation=None)
    ax[0][1].set_title('Mean {:.03f}, min {:.03f}, max {:.03f}'
            .format(original[1,:,:,0].mean(),
                    original[1,:,:,0].min(),
                    original[1,:,:,0].max()))
    ax[0][1].set_xticks([])
    ax[0][1].set_yticks([])

    ax[1][0].imshow(reconstructed[0,:,:,0].numpy(), cmap='gray', interpolation=None)
    ax[1][0].set_title('Mean {:.03f}, min {:.03f}, max {:.03f}'
            .format(reconstructed[0,:,:,0].numpy().mean(),
                    reconstructed[0,:,:,0].numpy().min(),
                    reconstructed[0,:,:,0].numpy().max()))
    ax[1][0].set_xticks([])
    ax[1][0].set_yticks([])

    ax[1][1].imshow(reconstructed[1,:,:,0].numpy(), cmap='gray', interpolation=None)
    ax[1][1].set_title('Mean {:.03f}, min {:.03f}, max {:.03f}'
            .format(reconstructed[1,:,:,0].numpy().mean(),
                    reconstructed[1,:,:,0].numpy().min(),
                    reconstructed[1,:,:,0].numpy().max()))
    ax[1][1].set_xticks([])
    ax[1][1].set_yticks([])

    ax[2][0].hist(original[0,:,:,0].flatten(), bins=256)
    ax[2][0].hist(reconstructed[0,:,:,0].numpy().flatten(), bins=256)

    ax[2][1].hist(original[1,:,:,0].flatten(), bins=256)
    ax[2][1].hist(reconstructed[1,:,:,0].numpy().flatten(), bins=256)

    fig.savefig(os.path.join(save_path, 'reconstruction_'+str(epoch+1)+'.pdf'),
                    bbox_inches='tight', dpi = 100)
    fig.savefig(os.path.join(save_path, 'reconstruction_'+str(epoch+1)+'.png'),
                    bbox_inches='tight', dpi = 100)

    if display is True:
        plt.show()
    else:
        plt.close()


def tictoc(tic=0, toc=1):
    '''
    Returns a string and a dictionary that contains the number of days, hours, minutes and
    seconds elapsed between tic and toc
    '''
    elapsed = toc-tic
    days, rem = np.divmod(elapsed, 86400)
    hours, rem = np.divmod(rem, 3600)
    minutes, rem = np.divmod(rem, 60)
    seconds, rem = np.divmod(rem, 1)
    milliseconds = rem*1000

    dictionary = {}
    dictionary['days']=days
    dictionary['hours']=hours
    dictionary['minutes']=minutes
    dictionary['seconds']=seconds
    dictionary['milliseconds']=milliseconds

    # form a string in the format d:h:m:s
    return "%2dd:%02dh:%02dm:%02ds:%02dms" % (days, hours, minutes, seconds, milliseconds), dictionary