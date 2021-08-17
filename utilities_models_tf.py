'''
General methods used by the models to be used by the classes in models_tf.py
'''

import glob # Unix style pathname pattern expansion
import os
import sys
import json
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, precision_score


import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


def fix_labels(labels, unique_labels, categorical=True):
    '''
    Prepares the labels for training using the specifications in unique_labels.
    This was initially done in the data generator, but given that working with
    TFrecords does not allow some operations, labels have to be fixed here.

    Args:
    labels: numpy array containing the labels
    unique_labels (list): list of the wanted labels and their organization
        # For example:
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
        # chack that it specifies a know classification type
        if not (classification_type=='c1' or classification_type=='c2' or classification_type=='c3'):
            raise ValueError(f'classification_type expected to be c1, c2 or c3. Instead was given {classification_type}')

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

def f1Score(y_true, y_pred):
    '''
    Computes the f1 score defined as:
    f1-score = 2 * (Precision * Recall) / (Precision + Recall)

    where
    Precision = tp / (tp + fp)
    Recall = tp / (tp + fn)
    '''
    return f1_score(tf.argmax(y_true, -1), tf.argmax(y_pred, -1), average='macro')


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

def plotModelPerformance_v2(tr_loss, tr_acc, val_loss, val_acc, tr_f1, val_f1, save_path, display=False):
    '''
    Saves training and validation curves.
    INPUTS
    - tr_loss: training loss history
    - tr_acc: training accuracy history
    - tr_f1 : training f1-score history
    - val_loss: validation loss history
    - val_acc: validation accuracy history
    - val_f1 : validation f1-score history
    - save_path: path to where to save the model
    '''

    fig, ax1 = plt.subplots(figsize=(15, 10))
    colors = ['blue', 'orange', 'green', 'red','pink','gray','purple','brown','olive','cyan','teal']
    line_style = [':', '-.', '--', '-']
    ax1.set_xlabel('Epochs', fontsize=15)
    ax1.set_ylabel('Loss', fontsize=15)
    l1 = ax1.plot(tr_loss, colors[0], ls=line_style[0])
    l2 = ax1.plot(val_loss, colors[1], ls=line_style[1])
    plt.legend(['Training loss', 'Validation loss'])

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel('Accuracy and F1-score', fontsize=15)
    ax2.set_ylim(bottom=0, top=1)
    l3 = ax2.plot(tr_acc, colors[2], ls=line_style[2])
    l4 = ax2.plot(val_acc, colors[3], ls=line_style[3])
    l5 = ax2.plot(tr_f1, colors[4], ls=line_style[2])
    l6 = ax2.plot(val_f1, colors[5], ls=line_style[3])

    # add legend
    lns = l1+l2+l3+l4+l5+l6
    labs = ['Training loss', 'Validation loss', 'Training accuracy', 'Validation accuracy', 'Training F1-score', 'Validation F1-score']
    ax1.legend(lns, labs, loc=7, fontsize=15)

    ax1.set_title('Training loss, accuracy and F1-score trends', fontsize=20)
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

def train(self, training_dataloader,
                validation_dataloader,
                classification_type,
                unique_labels,
                loss=('cee'),
                start_learning_rate = 0.001,
                scheduler='polynomial',
                power=0.1,
                max_epochs=200,
                early_stopping=True,
                patience=20,
                save_model_path=os.getcwd(),
                save_model_architecture_figure=True,
                vae_kl_weight=0.1,
                vae_reconst_weight=0.1,
                verbose=1):
    '''

    Training function used to train models implemented in the models_tf.py file
    in the context of the project for deep learning-based OCT image classsification.

    Parameters
    ----------
    self : models_tf.model
        Model to train (see available in models_tf.py)
    training_dataloader : tensorflow.python.data.ops.dataset_ops.PrefetchDataset
            Tensorflow dataloader that outputs a tuple of (image, label)
    validation_dataloader : tensorflow.python.data.ops.dataset_ops.PrefetchDataset
            Tensorflow dataloader that outputs a tuple of (image, label)
    classification_type : str
        Specifies the type of classification. it can be c1, c2 or c3. Needed to fix the
        labels using the unique_labels
    unique_labels : list
        List that specifies the classes to be included in every class. This is
        needed by the fix_label function to adjust the image labels based on the
        classification type.
    loss : str
        Type of loss function to use when comparing the model prediction with
        the ground truth. Available 'cce' - categorical cross entropy and
        'wcce' - weighted categorical cross entropy.
    start_learning_rate : float
        Starting learning rate (default=0.001).
    scheduler : str
        Specifies the type of learning rate scheduler to use during training.
        Available linear and polynomial (defauls linear).
    power : (float)
        Power of the polynomial learning rate scheduler (default=0.1).
    max_epochs : int
        Maximum number of epochs to run during training (default=200).
    early_stopping : bool
        Specifies if early stopping is used (True - default) or not (False).
    patience : int
        If early_stopping is True, specifies the number of epochs to waint the
        validation to increase before stopping the training (default=20).
    save_model_path : str
        Path to where the model and the training progress are saved (default=current
        working directory).
    vae_kl_weight : float
        Used when a Variational Auto Encoder model is used. This specifies the
        weight of the KL loss in the total model loss (defaul=0.1).
    vae_reconst_weight : float
        Used when a Variational Auto Encoder model is used. This specifies the
        weight of the reconstruction loss in the total model loss (default=0.1).
    verbose : int
        Specifies how much is printed during the training (0=nothing, 1=every
        epoch - default, 2=every batch)

    The training loop goes as follows for one epoch
    1 - for all the batches of data in the dataloader
    2 - fix labels based on the unique labels specification
    3 - compute training logits
    4 - compute loss and accuracy
    5 - update weights using the optimizer

    When the training batches are finished run validation

    6 - compute validation logits
    7 - compute validation loss and accuracy
    8 - check for early stopping
    9 - save perfromance curves and loss decay curve (if VEA model, save also
        reconstructed image)
    10 - save model if early stopping or the max number of epochs is reached
        in the specified directory
    '''

    # define parameters useful to store training and validation information
    self.train_loss_history, self.val_loss_history = [], []
    self.train_acc_history, self.val_acc_history = [], []
    self.train_f1_history, self.val_f1_history = [], []
    self.initial_learning_rate = start_learning_rate
    self.scheduler = scheduler
    self.maxEpochs = max_epochs
    self.learning_rate_history = []
    self.loss = loss
    self.num_validation_samples = 0
    self.num_training_samples = 0
    self.vae_kl_weight=vae_kl_weight
    self.vae_reconst_weight=vae_reconst_weight
    self.unique_labels = unique_labels
    self.classification_type = classification_type


    if verbose <= 2 and isinstance(verbose, int):
        self.verbose=verbose
    else:
        print('Invalid verbose parameter. Given {} but expected 0, 1 or 2. Setting to default 1'.format(verbose))

    # save model architecture figure
    if save_model_architecture_figure is True:
        try:
            tf.keras.utils.plot_model(self.model, to_file=os.path.join(self.save_model_path, 'model_architecture.png'), show_shapes=True)
        except:
            print('Cannot save model architecture as figure. Printing instead.')
            self.model.summary()

    if early_stopping:
        self.best_acc = 0.0
        self.best_f1 = 0.0
        n_wait = 0

    start = time.time()
    # start looping through the epochs
    for epoch in range(self.maxEpochs):
        # initialize the variables
        epoch_train_loss, epoch_val_loss = [], []
        epoch_train_acc, epoch_val_acc = [], []
        epoch_train_f1, epoch_val_f1 = [], []

        # compute learning rate based on the scheduler
        if self.scheduler == 'linear':
            self.power = 1
        elif self.scheduler == 'polynomial':
            self.power = power
        else:
            raise TypeError('Invalid scheduler. given {} but expected linear or polynomial'.format(self.scheduler))

        lr = leraningRateScheduler(self.initial_learning_rate, epoch, self.maxEpochs, power)
        self.learning_rate_history.append(lr)

        # set optimizer - using ADAM by default
        optimizer = Adam(lr=lr)

        # ####### TRAINING
        step = 0
        for x, y in training_dataloader:
            step += 1

            # make data usable
            x = x.numpy()
            y = fix_labels_v2(y.numpy(), self.classification_type, self.unique_labels)

            # save information about training image size
            if epoch == 0 and step == 1:
                self.batch_size = x.shape[0]
                self.input_size = (x.shape[1], x.shape[2])

            # Open a GradientTape to record the operations run
            # during the forward pass, which enables autodifferentiation.
            with tf.GradientTape() as tape:
                # Logits for this minibatch
                if 'VAE' in self.model_name:
                    train_logits, reconstruction, augmented_norm, z_mean, z_log_var, z = self.model(x, training=True)
                else:
                    train_logits = self.model(x, training=True)

                # classification loss
                for l in loss:
                    if l == 'cce':
                        # compute categorical cross entropy
                        cce = tf.keras.losses.CategoricalCrossentropy()
                        classification_loss = cce(y, train_logits)
                        # compute weighted categorical cross entropy
                    elif l == 'wcce':
                        weights = tf.constant(self.class_weights, dtype=tf.float32)
                        # one weight for each sample [batch_size, 1]
                        weights = tf.reduce_sum(weights * y, axis=1)
                        # weighted loss
                        cce = tf.keras.losses.CategoricalCrossentropy()
                        classification_loss = cce(y, train_logits, sample_weight=weights)
                    elif l == 'sfce':
                        sfce = tfa.losses.SigmoidFocalCrossEntropy(from_logits=True)
                        classification_loss = tf.reduce_mean(sfce(y, train_logits))
                    else:
                        raise TypeError('Invalid loss. given {} but expected cee, wcce or sfce'.format(l))

                if 'VAE' in self.model_name:
                    # reconstruction loss
                    mse = tf.keras.losses.MeanSquaredError()
                    reconstruction_loss = mse(x, reconstruction)*1000

                    #k-1 loss: Kulback-Leibler divergence that tries to make the latent space (z)
                    # of the encoded vector as regular as possible (N(0,1))
                    kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                    kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

                    # COMPUTE TOTAL LOSS
                    # it is very important for the loss to be a tensor! If not
                    # the gradient tape will not be able to use it during the
                    # backpropagation.

                    train_loss = classification_loss + self.vae_kl_weight*kl_loss + self.vae_reconst_weight*reconstruction_loss
                else:
                    # COMPUTE TOTAL LOSS
                    # it is very important for the loss to be a tensor! If not
                    # the gradient tape will not be able to use it during the
                    # backpropagation.
                    train_loss = classification_loss


            # save metrics
            epoch_train_loss.append(float(train_loss))
            train_acc = accuracy(y, train_logits)
            train_f1 = f1Score(y, train_logits)
            epoch_train_acc.append(float(train_acc))
            epoch_train_f1.append(float(train_f1))

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            if 'VAE' in self.model_name:
                grads = tape.gradient(train_loss, self.model.trainable_weights,
                                      unconnected_gradients=tf.UnconnectedGradients.ZERO)
            else:
                grads = tape.gradient(train_loss, self.model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

            # print values
            if self.verbose == 2:
                if epoch == 0:
                    print('\r', end='')
                    print('Epoch {:04d} training (counting training steps) -> {:04d} -> tr_loss:{:.4f}, tr_acc:{:.4f}, tr_f1:{:.4f} \r'
                            .format(epoch+1, step, train_loss, train_acc, train_f1),end='')
                else:
                    print('Epoch {:04d} training -> {:04d}/{:04d} -> tr_loss:{:.4f}, tr_acc:{:.4f}, tr_f1:{:.4f} \r'
                            .format(epoch+1,
                                    step,
                                    self.num_training_samples//self.batch_size,
                                    train_loss,
                                    train_acc,
                                    train_f1)
                            ,end='')

        # finisced all the training batches -> save training loss
        self.train_loss_history.append(np.mean(np.array(epoch_train_loss), axis=0))
        self.train_acc_history.append(np.mean(np.array(epoch_train_acc), axis=0))
        self.train_f1_history.append(np.mean(np.array(epoch_train_f1), axis=0))
        if epoch == 0:
            self.num_training_samples = self.batch_size*step

        # ########### VALIDATION
        step = 0
        for x, y in validation_dataloader:
            step += 1

            # make data usable
            x = x.numpy()
            y = fix_labels_v2(y.numpy(), self.classification_type, self.unique_labels)

            # logits for this validation batch
            if 'VAE' in self.model_name:
                val_logits, reconstruction, augmented_norm, z_mean, z_log_var, z = self.model(x, training=False)
            else:
                val_logits = self.model(x, training=False)

            # classification loss
            for l in loss:
                if l == 'cce':
                    # compute categorical cross entropy
                    cce = tf.keras.losses.CategoricalCrossentropy()
                    classification_loss = cce(y, val_logits)
                    # compute weighted categorical cross entropy
                elif l == 'wcce':
                    weights = tf.constant(self.class_weights, dtype=tf.float32)
                    # one weight for each sample [batch_size, 1]
                    weights = tf.reduce_sum(weights * y, axis=1)
                    # weighted loss
                    cce = tf.keras.losses.CategoricalCrossentropy()
                    classification_loss = cce(y, val_logits, sample_weight=weights)
                elif l == 'sfce':
                    sfce = tfa.losses.SigmoidFocalCrossEntropy(from_logits=True)
                    classification_loss = tf.reduce_mean(sfce(y, val_logits))
                else:
                    raise TypeError('Invalid loss. given {} but expected cee, wcce or sfce'.format(l))

            if 'VAE' in self.model_name:
                # reconstruction loss
                mse = tf.keras.losses.MeanSquaredError()
                reconstruction_loss = mse(x, reconstruction)*1000

                #k-1 loss: Kulback-Leibler divergence that tries to make the latent space (z)
                # of the encoded vector as regular as possible (N(0,1))
                kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

                # COMPUTE TOTAL LOSS
                # it is very important for the loss to be a tensor! If not
                # the gradient tape will not be able to use it during the
                # backpropagation.
                val_loss = classification_loss + self.vae_kl_weight*kl_loss + self.vae_reconst_weight*reconstruction_loss
            else:
                val_loss = classification_loss

            epoch_val_loss.append(float(val_loss))
            val_acc = accuracy(y, val_logits)
            val_f1 = f1Score(y, val_logits)
            epoch_val_acc.append(float(val_acc))
            epoch_val_f1.append(float(val_f1))

            # print values
            if self.verbose == 2:
                if epoch == 0:
                    print('\r', end='')
                    print('Epoch {:04d} validation (counting validation steps) -> {:04d} -> val_loss:{:.4f}, val_acc:{:.4f}, val_f1:{:.4f} \r'
                            .format(epoch+1, step, val_loss, val_acc, val_f1),
                            end='')
                else:
                    print('Epoch {:04d} validation -> {:04d}/{:04d} -> val_loss:{:.4f}, val_acc:{:.4f}, val_f1:{:.4f} \r'
                            .format(epoch+1, step, self.num_validation_samples//self.batch_size, val_loss, val_acc, val_f1),
                            end='')


        # finisced all the batches in the validation
        self.val_loss_history.append(np.mean(np.array(epoch_val_loss), axis=0))
        self.val_acc_history.append(np.mean(np.array(epoch_val_acc), axis=0))
        self.val_f1_history.append(np.mean(np.array(epoch_val_f1), axis=0))

        if self.verbose == 1 or self.verbose == 2:
            print('Epoch {:04d} -> tr_loss:{:.4f}, tr_acc:{:.4f}, val_loss:{:.4f}, val_acc:{:.4f}, val_f1:{:.04} {:10s}'.format(epoch+1,
                            self.train_loss_history[-1], self.train_acc_history[-1],
                            self.val_loss_history[-1], self.val_acc_history[-1],
                            self.val_f1_history[-1],
                            ''*10))
        if epoch == 0:
            self.num_validation_samples = self.batch_size*step

        if epoch % 2 == 0:
            # plotModelPerformance(self.train_loss_history,
            #                         self.train_acc_history,
            #                         self.val_loss_history,
            #                         self.val_acc_history,
            #                         self.save_model_path,
            #                         display=False)

            plotModelPerformance_v2(self.train_loss_history,
                                    self.train_acc_history,
                                    self.val_loss_history,
                                    self.val_acc_history,
                                    self.train_f1_history,
                                    self.val_f1_history,
                                    self.save_model_path,
                                    display=False)

            plotLearningRate(self.learning_rate_history, self.save_model_path, display=False)

            if 'VAE' in self.model_name:
                plotVAEreconstruction(augmented_norm.numpy(), reconstruction, epoch, self.save_model_path)

        if early_stopping:
            # check if model accurary improved, and update counter if needed
            if self.val_f1_history[-1] > self.best_f1:
                # save model checkpoint
                if self.verbose == 1 or self.verbose == 2:
                    print(' - Saving model checkpoint in {}'.format(self.save_model_path))
                # save some extra parameters

                stop = time.time()
                self.training_time, _ = tictoc(start, stop)
                # self.training_time = 1234
                self.training_epochs = epoch
                self.best_acc = self.val_acc_history[-1]
                self.best_f1 = self.val_f1_history[-1]

                # save model
                save_model(self)

                # reset counter
                n_wait = 0
            else:
                n_wait += 1
            # check max waiting is reached
            if n_wait == patience:
                if self.verbose == 1 or self.verbose == 2:
                    print(' -  Early stopping patient reached. Last model saved in {}'.format(self.save_model_path))
                break

def test(self, test_dataloader):
    '''

    Function that given a trained model outputs the prediction on the test dataset

    Parameters
    ----------

    self : models_tf.model
        Trained model
    test_dataloader : tensorflow.python.data.ops.dataset_ops.PrefetchDataset
            Tensorflow dataloader that outputs a tuple of (image, label)

    Outputs
    -------
    test_gt : tensorflow.tensor
        Ground truth values
    test_logits : tensorflow.tensor
        Logits predicted values
    test_time : float
        Time spent for testing

    Testing loop
    1 - for all the batches of data in the dataloader
        - fix labels based on the unique labels specification
        - compute prediction on all the test images
    2 - compute accuracy
    3 - print class accuracy and overall accuracy
    4 - return groud truth, predicted values and test time
    '''
    # initialize variables
    test_logits = tf.zeros([0, len(self.unique_labels)])
    test_gt = tf.zeros([0, len(self.unique_labels)])

    step = 0
    test_start = time.time()
    for x, y in test_dataloader:
        x = x.numpy()
        test_gt = tf.concat([test_gt, fix_labels_v2(y.numpy(), self.classification_type, self.unique_labels)], axis=0)
        if 'VAE' in self.model_name:
            aus_logits, _, _, _, _, _ = self.model(x, training=False)
        else:
            aus_logits= self.model(x, training=False)
        test_logits = tf.concat([test_logits, aus_logits], axis=0)

    test_stop = time.time()
    print('Model accuracy: {:.02f}'.format(accuracy(test_gt, test_logits)))

    # return test predictions
    return test_gt, test_logits, test_stop-test_start

def save_model(self):
    '''

    This functionsaves the model along with its weights and additional
    information about its architecture and training performance.

    Parameters
    ---------
    self : models_tf.model
        Model to save
    '''

    # save model and weights
    self.model.save(os.path.join(self.save_model_path, 'model'+'.tf'))
    self.model.save_weights(os.path.join(self.save_model_path, 'model_weights.tf'))

    # save extra information
    model_summary = {
        'Model_name' : self.model_name,
        'Num_input_channels': self.number_of_input_channels,
        'Num_classes' : self.num_classes,
        'Input_size' : self.input_size,
        'Unique_labels': self.unique_labels,
        'Class_weights': self.class_weights.tolist(),
        'Custom_model': self.custom_model,
        'Model_depth' : int(self.depth),
        'Num_filter_start' : self.num_filter_start,
        'Kernel_size' : list(self.kernel_size),
        'Num_training_samples' : int(self.num_training_samples),
        'nVALIDATION' :  int(self.num_validation_samples),
        'BATCH_SIZE' : int(self.batch_size),
        'EPOCHS' : int(self.training_epochs),
        'TRAINING_TIME' : self.training_time,
        'BEST_ACC' : self.best_acc,
        'LOSS': self.loss,
        'INITIAL_LERANING_RATE':self.initial_learning_rate,
        'SCHEDULER':self.scheduler,
        'POWER':self.power,
        'TRAIN_LOSS_HISTORY':self.train_loss_history,
        'VALIDATION_LOSS_HISTORY':self.val_loss_history,
        'TRAIN_ACC_HISTORY':self.train_acc_history,
        'VALIDATION_ACC_HISTORY':self.val_acc_history,
        'TRAIN_F1_HISTORY':self.train_f1_history,
        'VALIDATION_F1_HISTORY':self.val_f1_history,
        'KL_LOSS_WEIGHT':self.vae_kl_weight if 'VAE' in self.model_name else None,
        'RECONSTRUCTION_LOSS_WEIGHT':self.vae_reconst_weight if 'VAE' in self.model_name else None,
        'VAE_LATENT_SPACE_DIM':self.vae_latent_dim if 'VAE' in self.model_name else None
        }
    with open(os.path.join(self.save_model_path, 'model_summary_json.txt'), 'w') as outfile:
        json.dump(model_summary, outfile)
























