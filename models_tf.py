'''
Implementation of different models used for OCT image classification. Each model
has a custom training function that, for all models, saves the training and
validation curves, model summary and model check-point. For the Variational
Auto-Encoder models, the training function saves also the reconstructed image.
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
import tensorflow_addons as tfa
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Conv2DTranspose, Dropout, LeakyReLU, Dense, GlobalMaxPooling2D, Flatten, Reshape
import tensorflow.keras.layers as layers
import tensorflow_addons as tfa

# custom imports
import utilities_models_tf

class LightOCT(object):
    '''
    Implementation of the LightOCT described in https://arxiv.org/abs/1812.02487
    used for OCT image classification.
    The model architecture is:
    conv(5x5, 8 filters) - ReLU - MaxPool(2x2) - conv(5x5, 32) - ReLU - Flatten - Softmax - outpul Layer
    '''
    def __init__(self, number_of_input_channels,
                    num_classes,
                    input_size,
                    data_augmentation=True,
                    class_weights=None,
                    kernel_size=(5,5),
                    model_name='LightOCT',
                    debug=False):

        self.number_of_input_channels = number_of_input_channels
        self.num_classes = num_classes
        self.input_size=input_size
        self.debug = debug
        if class_weights is None:
            self.class_weights = np.ones([1, self.num_classes])
        else:
            self.class_weights = class_weights
        self.model_name = model_name
        self.kernel_size = kernel_size

        inputs = Input(shape=[None, None, self.number_of_input_channels])

        # Create a data augmentation stage with normalization, horizontal flipping and rotations
        if data_augmentation is True:
            augmentor = tf.keras.Sequential([
                    layers.experimental.preprocessing.Normalization(),
                    layers.experimental.preprocessing.RandomFlip("horizontal"),
                    layers.experimental.preprocessing.RandomRotation(0.1),
                    layers.experimental.preprocessing.RandomCrop(input_size[0], input_size[0])],
                    name='NormalizationAugmentationCropping')
        else: # perform only normalization
            augmentor = tf.keras.Sequential([
                    layers.experimental.preprocessing.Normalization(),
                    layers.experimental.preprocessing.RandomCrop(input_size[0], input_size[0])],
                    name='NormalizationCrop')

        x = augmentor(inputs)

        # building LightOCT model
        x = Conv2D(filters=8,
                        kernel_size=self.kernel_size,
                        activation='relu',
                        padding='same',
                        )(x)
        x = MaxPooling2D(pool_size=(2,2),
                        strides=2
                        )(x)
        x = Conv2D(filters=32,
                        kernel_size=self.kernel_size,
                        activation='relu',
                        padding='same',
                        )(x)
        x = MaxPooling2D(pool_size=(2,2),
                        strides=2
                        )(x)
        # FCN
        x = GlobalMaxPooling2D()(x)
        # x = Flatten()(x)
        final = Dense(units=self.num_classes, activation='softmax')(x)

        # save model paramenters
        self.num_filter_start = 8
        self.depth = 2
        self.num_filter_per_layer = [8, 32]
        self.custom_model = False

        # finally make the model and return
        self.model = Model(inputs=inputs, outputs=final, name=model_name)

        # print model if needed
        if self.debug is True:
            print(self.model.summary())


    def train(self, training_dataloader,
                    validation_dataloader,
                    unique_labels=None,
                    loss=('cee'),
                    start_learning_rate = 0.001,
                    scheduler='polynomial',
                    power=0.1,
                    max_epochs=200,
                    early_stopping=True,
                    patience=20,
                    save_model_path=None,
                    save_model_architecture_figure=False,
                    verbose=1):
        '''
        The training loop goes as follows for one epoch
        1 - for all the batches of data in the dataloader
        2 - fix labels based on the unique labels specification
        3 - compute training logits
        4 - compute loss and accuracy
        5 - update weights using the optimizer

        When the training batches are finished run validation

        6 - compute validation logits
        7 - check for early stopping
        8 - same model if early stopping is reached or the max number of epochs
            int eh specified directory
        '''
        # define parameters useful for saving the model
        self.save_model_path = save_model_path

        # define parameters useful to store training and validation information
        self.train_loss_history, self.val_loss_history = [], []
        self.train_acc_history, self.val_acc_history = [], []
        self.initial_learning_rate = start_learning_rate
        self.scheduler = scheduler
        self.maxEpochs = max_epochs
        self.learning_rate_history = []
        self.loss = loss
        self.unique_labels = unique_labels
        self.num_validation_samples = 0
        self.num_training_samples = 0
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
            n_wait = 0

        start = time.time()
        # start looping through the epochs
        for epoch in range(self.maxEpochs):
            # initialize the variables
            epoch_train_loss, epoch_val_loss = [], []
            epoch_train_acc, epoch_val_acc = [], []

            running_loss = 0.0
            running_acc = 0.0

            # compute learning rate based on the scheduler
            if self.scheduler == 'linear':
                self.power = 1
            elif self.scheduler == 'polynomial':
                self.power = power
            else:
                raise TypeError('Invalid scheduler. given {} but expected linear or polynomial'.format(self.scheduler))

            lr = utilities_models_tf.leraningRateScheduler(self.initial_learning_rate, epoch, self.maxEpochs, power)
            self.learning_rate_history.append(lr)

            # set optimizer - using ADAM by default
            optimizer = Adam(lr=lr)

            # consume data from the dataloader
            step = 0
            for x, y in training_dataloader:
                step += 1

                # make data usable
                x = x.numpy()
                y = utilities_models_tf.fix_labels(y.numpy(), self.unique_labels)

                # save information about training patch size
                if epoch == 0 and step == 1:
                    self.batch_size = x.shape[0]

                # Open a GradientTape to record the operations run
                # during the forward pass, which enables autodifferentiation.
                with tf.GradientTape() as tape:
                    # Logits for this minibatch
                    train_logits = self.model(x, training=True)
                    # Compute the loss value for this minibatch.
                    train_loss = []
                    for l in loss:
                        if l == 'cce':
                            # compute categorical cross entropy
                            cce = tf.keras.losses.CategoricalCrossentropy()
                            train_loss.append(cce(y, train_logits))
                            # compute weighted categorical cross entropy
                        elif l == 'wcce':
                            weights = tf.constant(self.class_weights, dtype=tf.float32)
                            # one weight for each sample [batch_size, 1]
                            weights = tf.reduce_sum(weights * y, axis=1)
                            # weighted loss
                            cce = tf.keras.losses.CategoricalCrossentropy()
                            train_loss.append(cce(y, train_logits, sample_weight=weights))
                        elif l == 'sfce':
                            sfce = tfa.losses.SigmoidFocalCrossEntropy(from_logits=True)
                            train_loss.append(tf.reduce_mean(sfce(y, train_logits)))
                        else:
                            raise TypeError('Invalid loss. given {} but expected cee, wcce or sfce'.format(l))
                    # it is very important for the loss to be a tensor! If not
                    # the gradient tape will not be able to use it during the
                    # backpropagation.
                    train_loss = tf.add_n(train_loss)

                    # save metrics
                    epoch_train_loss.append(float(train_loss))
                    train_acc = utilities_models_tf.accuracy(y, train_logits)
                    epoch_train_acc.append(float(train_acc))

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                grads = tape.gradient(train_loss, self.model.trainable_weights)
                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

                # print values
                if self.verbose == 2:
                    # print progress in the epoch
                    if epoch == 0:
                        print('\n', end='')
                        print('Epoch {:04d} training (counting training steps) -> {:04d} -> tr_loss:{:.4f}, tr_acc:{:.4f} \r'
                                .format(epoch+1, step, train_loss, train_acc),end='')
                    else:
                        print('Epoch {:04d} training -> {:04d}/{:04d} -> tr_loss:{:.4f}, tr_acc:{:.4f} \r'
                                .format(epoch+1,
                                        step,
                                        self.num_training_samples//self.batch_size,
                                        train_loss,
                                        train_acc)
                                ,end='')

            # finisced all the training batches -> save training loss
            self.train_loss_history.append(np.mean(np.array(epoch_train_loss), axis=0))
            self.train_acc_history.append(np.mean(np.array(epoch_train_acc), axis=0))
            if epoch == 0:
                self.num_training_samples = self.batch_size*step

            # running validation loop
            step = 0
            for x, y in validation_dataloader:
                step += 1

                # make data usable
                x = x.numpy()
                y = utilities_models_tf.fix_labels(y.numpy(), self.unique_labels)

                # logits for this validation batch
                val_logits = self.model(x, training=False)  # Logits for this minibatch

                # Compute the loss value for this minibatch.
                val_loss = []
                for l in loss:
                        if l == 'cce':
                            # compute categorical cross entropy
                            cce = tf.keras.losses.CategoricalCrossentropy()
                            val_loss.append(cce(y, val_logits))
                        elif l == 'wcce':
                            # compute weighted categorical cross entropy
                            weights = tf.constant(self.class_weights, dtype=tf.float32)
                            weights = tf.reduce_sum(weights * y, axis=1)
                            cce = tf.keras.losses.CategoricalCrossentropy()
                            val_loss.append(cce(y, val_logits, sample_weight=weights))
                        elif l == 'sfce':
                            sfce = tfa.losses.SigmoidFocalCrossEntropy(from_logits=True)
                            val_loss.append(tf.reduce_mean(sfce(y, val_logits)))
                        else:
                            raise TypeError('Invalid loss. given {} but expected cee, wcce or sfce'.format(l))

                val_loss = tf.add_n(val_loss)
                epoch_val_loss.append(float(val_loss))
                val_acc = utilities_models_tf.accuracy(y, val_logits)
                epoch_val_acc.append(float(val_acc))

                # print values
                if self.verbose == 2:
                    # print epoch progress
                    if epoch == 0:
                        print('\r', end='')
                        print('Epoch {:04d} validation (counting validation steps) -> {:04d} -> val_loss:{:.4f}, val_acc:{:.4f} \r'
                                .format(epoch+1, step, val_loss, val_acc),
                                end='')
                    else:
                        print('Epoch {:04d} validation-> {:04d}/{:04d} -> val_loss:{:.4f}, val_acc:{:.4f} \r'
                                .format(epoch+1, step, self.num_validation_samples//self.batch_size, val_loss, val_acc),
                                end='')


            # finisced all the batches in the validation
            self.val_loss_history.append(np.mean(np.array(epoch_val_loss), axis=0))
            self.val_acc_history.append(np.mean(np.array(epoch_val_acc), axis=0))

            if self.verbose == 1 or self.verbose == 2:
                print('Epoch {:04d} -> tr_loss:{:.4f}, tr_acc:{:.4f}, val_loss:{:.4f}, val_acc:{:.4f}'.format(epoch+1,
                                    self.train_loss_history[-1], self.train_acc_history[-1],
                                    self.val_loss_history[-1], self.val_acc_history[-1]))
            if epoch == 0:
                self.num_validation_samples = self.batch_size*step

            if epoch % 5 == 0:
                utilities_models_tf.plotModelPerformance(self.train_loss_history,
                                     self.train_acc_history,
                                     self.val_loss_history,
                                     self.val_acc_history,
                                     self.save_model_path,
                                     display=False)

                utilities_models_tf.plotLearningRate(self.learning_rate_history, self.save_model_path, display=False)

            if early_stopping:
                # check if model accurary improved, and update counter if needed
                if self.val_acc_history[-1] > self.best_acc:
                    # save model checkpoint
                    if self.verbose == 1 or self.verbose == 2:
                        print(' - Saving model checkpoint in {}'.format(self.save_model_path))

                    # save some extra parameters
                    stop = time.time()
                    self.training_time, _ = utilities_models_tf.tictoc(start, stop)
                    # self.training_time = 1234
                    self.training_epochs = epoch
                    self.best_acc = self.val_acc_history[-1]

                    # save model
                    self.save_model()

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
            test_gt = tf.concat([test_gt, utilities_models_tf.fix_labels(y.numpy(), self.unique_labels)], axis=0)
            test_logits = tf.concat([test_logits, self.model(x, training=False)], axis=0)

        test_stop = time.time()
        print('Overall model accuracy: {:.02f}'.format(utilities_models_tf.accuracy(test_gt, test_logits)))

        # return test predictions
        return test_gt, test_logits, test_stop-test_start

    def save_model(self):
        '''
        This saves the model along with its weights and additional information
        about its architecture and training performance
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
            'VALIDATION_ACC_HISTORY':self.val_acc_history
            }
        with open(os.path.join(self.save_model_path, 'model_summary_json.txt'), 'w') as outfile:
            json.dump(model_summary, outfile)

## CUSTOM MODEL M2

class M2(object):
    '''
    Implementation of custom model for OCT image classification. Model architercture:
    3 convolutional layers (32, 64, 128) filters with ReLU activation and followed by MaxPooling.
    After the last conv layer, GlobalAveragePooling is used to obtain a one dimensional vector.
    The FCN is made by a dense layer of 60 nodes with ReLU activation and dropout, and final softmax.
    '''
    def __init__(self, number_of_input_channels,
                    num_classes,
                    data_augmentation=True,
                    class_weights=None,
                    kernel_size=(5,5),
                    model_name='M2',
                    debug=False):

        self.number_of_input_channels = number_of_input_channels
        self.num_classes = num_classes
        self.debug = debug
        if class_weights is None:
            self.class_weights = np.ones([1, self.num_classes])
        else:
            self.class_weights = class_weights
        self.model_name = model_name
        self.kernel_size = kernel_size

        inputs = Input(shape=[None, None, self.number_of_input_channels])

        # Create a data augmentation stage with normalization, horizontal flipping and rotations
        if data_augmentation is True:
            augmentor = tf.keras.Sequential([
                    layers.experimental.preprocessing.Normalization(),
                    layers.experimental.preprocessing.RandomFlip("horizontal"),
                    layers.experimental.preprocessing.RandomRotation(0.1)],
                    name='NormalizationAugmentation')
        else: # perform only normalization
            augmentor = tf.keras.Sequential([
                    layers.experimental.preprocessing.Normalization()],
                    name='Normalization')

        x = augmentor(inputs)

        # build model
        n_filters = [32, 64, 128]
        for i in n_filters:
            x = Conv2D(filters=i,
                        kernel_size=self.kernel_size)(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            x = MaxPooling2D(pool_size=(2,2),
                        strides=2
                        )(x)
        # FCN
        x = GlobalMaxPooling2D()(x)
        x = Dropout(rate=0.2)(x)
        x = Dense(units=60, activation='relu')(x)
        final = Dense(units=self.num_classes, activation='softmax')(x)

        # save model paramenters
        self.num_filter_start = n_filters[0]
        self.depth = len(n_filters)
        self.num_filter_per_layer = n_filters
        self.custom_model = False

        # finally make the model and return
        self.model = Model(inputs=inputs, outputs=final, name=model_name)

        # print model if needed
        if self.debug is True:
            print(self.model.summary())

    def train(self, training_dataloader,
                    validation_dataloader,
                    unique_labels=None,
                    loss=('cee'),
                    start_learning_rate = 0.001,
                    scheduler='polynomial',
                    power=0.1,
                    max_epochs=200,
                    early_stopping=True,
                    patience=20,
                    save_model_path=None,
                    save_model_architecture_figure=False):
        '''
        The training loop goes as follows for one epoch
        1 - for all the batches of data in the dataloader
        2 - fix labels based on the unique labels specification
        3 - compute training logits
        4 - compute loss and accuracy
        5 - update weights using the optimizer

        When the training batches are finished run validation

        6 - compute validation logits
        7 - check for early stopping
        8 - same model if early stopping is reached or the max number of epochs
            int eh specified directory
        '''
        # define parameters useful for saving the model
        self.save_model_path = save_model_path

        # define parameters useful to store training and validation information
        self.train_loss_history, self.val_loss_history = [], []
        self.train_acc_history, self.val_acc_history = [], []
        self.initial_learning_rate = start_learning_rate
        self.scheduler = scheduler
        self.maxEpochs = max_epochs
        self.learning_rate_history = []
        self.loss = loss
        self.unique_labels = unique_labels
        self.num_validation_samples = 0
        self.num_training_samples = 0

        # save model architecture figure
        if save_model_architecture_figure is True:
            try:
                tf.keras.utils.plot_model(self.model, to_file=os.path.join(self.save_model_path, 'model_architecture.png'), show_shapes=True)
            except:
                print('Cannot save model architecture as figure. Printing instead.')
                self.model.summary()

        if early_stopping:
            self.best_acc = 0.0
            n_wait = 0

        start = time.time()
        # start looping through the epochs
        for epoch in range(self.maxEpochs):
            # initialize the variables
            epoch_train_loss, epoch_val_loss = [], []
            epoch_train_acc, epoch_val_acc = [], []

            running_loss = 0.0
            running_acc = 0.0

            # compute learning rate based on the scheduler
            if self.scheduler == 'linear':
                self.power = 1
            elif self.scheduler == 'polynomial':
                self.power = power
            else:
                raise TypeError('Invalid scheduler. given {} but expected linear or polynomial'.format(self.scheduler))

            lr = utilities_models_tf.leraningRateScheduler(self.initial_learning_rate, epoch, self.maxEpochs, power)
            self.learning_rate_history.append(lr)

            # set optimizer - using ADAM by default
            optimizer = Adam(lr=lr)

            # consume data from the dataloader
            step = 0
            for x, y in training_dataloader:
                step += 1

                # make data usable
                x = x.numpy()
                y = utilities_models_tf.fix_labels(y.numpy(), self.unique_labels)

                # save information about training patch size
                if epoch == 0 and step == 1:
                    self.batch_size = x.shape[0]
                    self.input_size = (x.shape[1], x.shape[2])

                # Open a GradientTape to record the operations run
                # during the forward pass, which enables autodifferentiation.
                with tf.GradientTape() as tape:
                    # Logits for this minibatch
                    train_logits = self.model(x, training=True)
                    # Compute the loss value for this minibatch.
                    train_loss = []
                    for l in loss:
                        if l == 'cce':
                            # compute categorical cross entropy
                            cce = tf.keras.losses.CategoricalCrossentropy()
                            train_loss.append(cce(y, train_logits))
                            # compute weighted categorical cross entropy
                        elif l == 'wcce':
                            weights = tf.constant(self.class_weights, dtype=tf.float32)
                            # one weight for each sample [batch_size, 1]
                            weights = tf.reduce_sum(weights * y, axis=1)
                            # weighted loss
                            cce = tf.keras.losses.CategoricalCrossentropy()
                            train_loss.append(cce(y, train_logits, sample_weight=weights))
                        elif l == 'sfce':
                            sfce = tfa.losses.SigmoidFocalCrossEntropy(from_logits=True)
                            train_loss.append(tf.reduce_mean(sfce(y, train_logits)))
                        else:
                            raise TypeError('Invalid loss. given {} but expected cee, wcce or sfce'.format(l))
                    # it is very important for the loss to be a tensor! If not
                    # the gradient tape will not be able to use it during the
                    # backpropagation.
                    train_loss = tf.add_n(train_loss)

                    # save metrics
                    epoch_train_loss.append(float(train_loss))
                    train_acc = utilities_models_tf.accuracy(y, train_logits)
                    epoch_train_acc.append(float(train_acc))

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                grads = tape.gradient(train_loss, self.model.trainable_weights)
                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

                # print values
                if epoch == 0:
                    print('\r', end='')
                    print('Epoch {:04d} training (counting training steps) -> {:04d} -> tr_loss:{:.4f}, tr_acc:{:.4f} \r'
                            .format(epoch+1, step, train_loss, train_acc),end='')
                else:
                    print('Epoch {:04d} training -> {:04d}/{:04d} -> tr_loss:{:.4f}, tr_acc:{:.4f} \r'
                            .format(epoch+1,
                                    step,
                                    self.num_training_samples//self.batch_size,
                                    train_loss,
                                    train_acc)
                            ,end='')

            # finisced all the training batches -> save training loss
            self.train_loss_history.append(np.mean(np.array(epoch_train_loss), axis=0))
            self.train_acc_history.append(np.mean(np.array(epoch_train_acc), axis=0))
            if epoch == 0:
                self.num_training_samples = self.batch_size*step

            # running validation loop
            step = 0
            for x, y in validation_dataloader:
                step += 1

                # make data usable
                x = x.numpy()
                y = utilities_models_tf.fix_labels(y.numpy(), self.unique_labels)

                # logits for this validation batch
                val_logits = self.model(x, training=False)  # Logits for this minibatch

                # Compute the loss value for this minibatch.
                val_loss = []
                for l in loss:
                        if l == 'cce':
                            # compute categorical cross entropy
                            cce = tf.keras.losses.CategoricalCrossentropy()
                            val_loss.append(cce(y, val_logits))
                        elif l == 'wcce':
                            # compute weighted categorical cross entropy
                            weights = tf.constant(self.class_weights, dtype=tf.float32)
                            weights = tf.reduce_sum(weights * y, axis=1)
                            cce = tf.keras.losses.CategoricalCrossentropy()
                            val_loss.append(cce(y, val_logits, sample_weight=weights))
                        elif l == 'sfce':
                            sfce = tfa.losses.SigmoidFocalCrossEntropy(from_logits=True)
                            val_loss.append(tf.reduce_mean(sfce(y, val_logits)))
                        else:
                            raise TypeError('Invalid loss. given {} but expected cee, wcce or sfce'.format(l))

                val_loss = tf.add_n(val_loss)
                epoch_val_loss.append(float(val_loss))
                val_acc = utilities_models_tf.accuracy(y, val_logits)
                epoch_val_acc.append(float(val_acc))

                # print values
                if epoch == 0:
                    print('\r', end='')
                    print('Epoch {:04d} validation (counting validation steps) -> {:04d} -> val_loss:{:.4f}, val_acc:{:.4f} \r'
                            .format(epoch+1, step, val_loss, val_acc),
                            end='')
                else:
                    print('Epoch {:04d} validation-> {:04d}/{:04d} -> val_loss:{:.4f}, val_acc:{:.4f} \r'
                            .format(epoch+1, step, self.num_validation_samples//self.batch_size, val_loss, val_acc),
                            end='')


            # finisced all the batches in the validation
            self.val_loss_history.append(np.mean(np.array(epoch_val_loss), axis=0))
            self.val_acc_history.append(np.mean(np.array(epoch_val_acc), axis=0))

            print('Epoch {:04d} -> tr_loss:{:.4f}, tr_acc:{:.4f}, val_loss:{:.4f}, val_acc:{:.4f}'.format(epoch+1,
                                self.train_loss_history[-1], self.train_acc_history[-1],
                                self.val_loss_history[-1], self.val_acc_history[-1]))
            if epoch == 0:
                self.num_validation_samples = self.batch_size*step

            if epoch % 5 == 0:
                utilities_models_tf.plotModelPerformance(self.train_loss_history,
                                     self.train_acc_history,
                                     self.val_loss_history,
                                     self.val_acc_history,
                                     self.save_model_path,
                                     display=False)

                utilities_models_tf.plotLearningRate(self.learning_rate_history, self.save_model_path, display=False)

            if early_stopping:
                # check if model accurary improved, and update counter if needed
                if self.val_acc_history[-1] > self.best_acc:
                    # save model checkpoint
                    print(' - Saving model checkpoint in {}'.format(self.save_model_path))
                    # save some extra parameters

                    stop = time.time()
                    self.training_time, _ = utilities_models_tf.tictoc(start, stop)
                    # self.training_time = 1234
                    self.training_epochs = epoch
                    self.best_acc = self.val_acc_history[-1]

                    # save model
                    self.save_model()

                    # reset counter
                    n_wait = 0
                else:
                    n_wait += 1
                # check max waiting is reached
                if n_wait == patience:
                    print(' -  Early stopping patient reached. Last model saved in {}'.format(self.save_model_path))
                    break

    def test(self, test_dataloader):
        '''
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
            test_gt = tf.concat([test_gt, utilities_models_tf.fix_labels(y.numpy(), self.unique_labels)], axis=0)
            test_logits = tf.concat([test_logits, self.model(x, training=False)], axis=0)

        test_stop = time.time()
        print('Overall model accuracy: {:.02f}'.format(utilities_models_tf.accuracy(test_gt, test_logits)))

        # return test predictions
        return test_gt, test_logits, test_stop-test_start

    def save_model(self):
        '''
        This saves the model along with its weights and additional information
        about its architecture and training performance
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
            'VALIDATION_ACC_HISTORY':self.val_acc_history
            }
        with open(os.path.join(self.save_model_path, 'model_summary_json.txt'), 'w') as outfile:
            json.dump(model_summary, outfile)

## CUSTOM MODEL M3

class M3(object):
    '''
    Implementation of custom model for OCT image classification. Model architercture:
    2 convolutional layer (8-8, 32-32) filters with ReLU activation, followed by MaxPooling
    After the last conv layer, GlobalAveragePooling is used to obtain a one dimensional vector.
    The FCN is made by a dense layer of 60 nodes with ReLU activation and dropout, and final softmax.
    '''
    def __init__(self, number_of_input_channels,
                    num_classes,
                    data_augmentation=True,
                    class_weights=None,
                    kernel_size=(5,5),
                    model_name='M3',
                    debug=False):

        self.number_of_input_channels = number_of_input_channels
        self.num_classes = num_classes
        self.debug = debug
        if class_weights is None:
            self.class_weights = np.ones([1, self.num_classes])
        else:
            self.class_weights = class_weights
        self.model_name = model_name
        self.kernel_size = kernel_size

        inputs = Input(shape=[None, None, self.number_of_input_channels])

        # Create a data augmentation stage with normalization, horizontal flipping and rotations
        if data_augmentation is True:
            augmentor = tf.keras.Sequential([
                    layers.experimental.preprocessing.Normalization(),
                    layers.experimental.preprocessing.RandomFlip("horizontal"),
                    layers.experimental.preprocessing.RandomRotation(0.1)],
                    name='NormalizationAugmentation')
        else: # perform only normalization
            augmentor = tf.keras.Sequential([
                    layers.experimental.preprocessing.Normalization()],
                    name='Normalization')

        x = augmentor(inputs)

        # build model
        n_filters = [8, 32]
        for i in n_filters:
            x = Conv2D(filters=i,
                        kernel_size=self.kernel_size)(x)
            x = Conv2D(filters=i,
                        kernel_size=self.kernel_size)(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            x = MaxPooling2D(pool_size=(2,2),
                        strides=2
                        )(x)
        # FCN
        x = GlobalMaxPooling2D()(x)
        x = Dropout(rate=0.2)(x)
        x = Dense(units=60, activation='relu')(x)
        final = Dense(units=self.num_classes, activation='softmax')(x)

        # save model paramenters
        self.num_filter_start = n_filters[0]
        self.depth = len(n_filters)
        self.num_filter_per_layer = n_filters
        self.custom_model = False

        # finally make the model and return
        self.model = Model(inputs=inputs, outputs=final, name=model_name)

        # print model if needed
        if self.debug is True:
            print(self.model.summary())

    def train(self, training_dataloader,
                    validation_dataloader,
                    unique_labels=None,
                    loss=('cee'),
                    start_learning_rate = 0.001,
                    scheduler='polynomial',
                    power=0.1,
                    max_epochs=200,
                    early_stopping=True,
                    patience=20,
                    save_model_path=None,
                    save_model_architecture_figure=False):
        '''
        The training loop goes as follows for one epoch
        1 - for all the batches of data in the dataloader
        2 - fix labels based on the unique labels specification
        3 - compute training logits
        4 - compute loss and accuracy
        5 - update weights using the optimizer

        When the training batches are finished run validation

        6 - compute validation logits
        7 - check for early stopping
        8 - same model if early stopping is reached or the max number of epochs
            int eh specified directory
        '''
        # define parameters useful for saving the model
        self.save_model_path = save_model_path

        # define parameters useful to store training and validation information
        self.train_loss_history, self.val_loss_history = [], []
        self.train_acc_history, self.val_acc_history = [], []
        self.initial_learning_rate = start_learning_rate
        self.scheduler = scheduler
        self.maxEpochs = max_epochs
        self.learning_rate_history = []
        self.loss = loss
        self.unique_labels = unique_labels
        self.num_validation_samples = 0
        self.num_training_samples = 0

        # save model architecture figure
        if save_model_architecture_figure is True:
            try:
                tf.keras.utils.plot_model(self.model, to_file=os.path.join(self.save_model_path, 'model_architecture.png'), show_shapes=True)
            except:
                print('Cannot save model architecture as figure. Printing instead.')
                self.model.summary()

        if early_stopping:
            self.best_acc = 0.0
            n_wait = 0

        start = time.time()
        # start looping through the epochs
        for epoch in range(self.maxEpochs):
            # initialize the variables
            epoch_train_loss, epoch_val_loss = [], []
            epoch_train_acc, epoch_val_acc = [], []

            running_loss = 0.0
            running_acc = 0.0

            # compute learning rate based on the scheduler
            if self.scheduler == 'linear':
                self.power = 1
            elif self.scheduler == 'polynomial':
                self.power = power
            else:
                raise TypeError('Invalid scheduler. given {} but expected linear or polynomial'.format(self.scheduler))

            lr = utilities_models_tf.leraningRateScheduler(self.initial_learning_rate, epoch, self.maxEpochs, power)
            self.learning_rate_history.append(lr)

            # set optimizer - using ADAM by default
            optimizer = Adam(lr=lr)

            # consume data from the dataloader
            step = 0
            for x, y in training_dataloader:
                step += 1

                # make data usable
                x = x.numpy()
                y = utilities_models_tf.fix_labels(y.numpy(), self.unique_labels)

                # save information about training patch size
                if epoch == 0 and step == 1:
                    self.batch_size = x.shape[0]
                    self.input_size = (x.shape[1], x.shape[2])

                # Open a GradientTape to record the operations run
                # during the forward pass, which enables autodifferentiation.
                with tf.GradientTape() as tape:
                    # Logits for this minibatch
                    train_logits = self.model(x, training=True)
                    # Compute the loss value for this minibatch.
                    train_loss = []
                    for l in loss:
                        if l == 'cce':
                            # compute categorical cross entropy
                            cce = tf.keras.losses.CategoricalCrossentropy()
                            train_loss.append(cce(y, train_logits))
                            # compute weighted categorical cross entropy
                        elif l == 'wcce':
                            weights = tf.constant(self.class_weights, dtype=tf.float32)
                            # one weight for each sample [batch_size, 1]
                            weights = tf.reduce_sum(weights * y, axis=1)
                            # weighted loss
                            cce = tf.keras.losses.CategoricalCrossentropy()
                            train_loss.append(cce(y, train_logits, sample_weight=weights))
                        elif l == 'sfce':
                            sfce = tfa.losses.SigmoidFocalCrossEntropy(from_logits=True)
                            train_loss.append(tf.reduce_mean(sfce(y, train_logits)))
                        else:
                            raise TypeError('Invalid loss. given {} but expected cee, wcce or sfce'.format(l))
                    # it is very important for the loss to be a tensor! If not
                    # the gradient tape will not be able to use it during the
                    # backpropagation.
                    train_loss = tf.add_n(train_loss)

                    # save metrics
                    epoch_train_loss.append(float(train_loss))
                    train_acc = utilities_models_tf.accuracy(y, train_logits)
                    epoch_train_acc.append(float(train_acc))

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                grads = tape.gradient(train_loss, self.model.trainable_weights)
                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

                # print values
                if epoch == 0:
                    print('\r', end='')
                    print('Epoch {:04d} training (counting training steps) -> {:04d} -> tr_loss:{:.4f}, tr_acc:{:.4f} \r'
                            .format(epoch+1, step, train_loss, train_acc),end='')
                else:
                    print('Epoch {:04d} training -> {:04d}/{:04d} -> tr_loss:{:.4f}, tr_acc:{:.4f} \r'
                            .format(epoch+1,
                                    step,
                                    self.num_training_samples//self.batch_size,
                                    train_loss,
                                    train_acc)
                            ,end='')

            # finisced all the training batches -> save training loss
            self.train_loss_history.append(np.mean(np.array(epoch_train_loss), axis=0))
            self.train_acc_history.append(np.mean(np.array(epoch_train_acc), axis=0))
            if epoch == 0:
                self.num_training_samples = self.batch_size*step

            # running validation loop
            step = 0
            for x, y in validation_dataloader:
                step += 1

                # make data usable
                x = x.numpy()
                y = utilities_models_tf.fix_labels(y.numpy(), self.unique_labels)

                # logits for this validation batch
                val_logits = self.model(x, training=False)  # Logits for this minibatch

                # Compute the loss value for this minibatch.
                val_loss = []
                for l in loss:
                        if l == 'cce':
                            # compute categorical cross entropy
                            cce = tf.keras.losses.CategoricalCrossentropy()
                            val_loss.append(cce(y, val_logits))
                        elif l == 'wcce':
                            # compute weighted categorical cross entropy
                            weights = tf.constant(self.class_weights, dtype=tf.float32)
                            weights = tf.reduce_sum(weights * y, axis=1)
                            cce = tf.keras.losses.CategoricalCrossentropy()
                            val_loss.append(cce(y, val_logits, sample_weight=weights))
                        elif l == 'sfce':
                            sfce = tfa.losses.SigmoidFocalCrossEntropy(from_logits=True)
                            val_loss.append(tf.reduce_mean(sfce(y, val_logits)))
                        else:
                            raise TypeError('Invalid loss. given {} but expected cee, wcce or sfce'.format(l))

                val_loss = tf.add_n(val_loss)
                epoch_val_loss.append(float(val_loss))
                val_acc = utilities_models_tf.accuracy(y, val_logits)
                epoch_val_acc.append(float(val_acc))

                # print values
                if epoch == 0:
                    print('\r', end='')
                    print('Epoch {:04d} validation (counting validation steps) -> {:04d} -> val_loss:{:.4f}, val_acc:{:.4f} \r'
                            .format(epoch+1, step, val_loss, val_acc),
                            end='')
                else:
                    print('Epoch {:04d} validation-> {:04d}/{:04d} -> val_loss:{:.4f}, val_acc:{:.4f} \r'
                            .format(epoch+1, step, self.num_validation_samples//self.batch_size, val_loss, val_acc),
                            end='')


            # finisced all the batches in the validation
            self.val_loss_history.append(np.mean(np.array(epoch_val_loss), axis=0))
            self.val_acc_history.append(np.mean(np.array(epoch_val_acc), axis=0))

            print('Epoch {:04d} -> tr_loss:{:.4f}, tr_acc:{:.4f}, val_loss:{:.4f}, val_acc:{:.4f}'.format(epoch+1,
                                self.train_loss_history[-1], self.train_acc_history[-1],
                                self.val_loss_history[-1], self.val_acc_history[-1]))
            if epoch == 0:
                self.num_validation_samples = self.batch_size*step

            if epoch % 5 == 0:
                utilities_models_tf.plotModelPerformance(self.train_loss_history,
                                     self.train_acc_history,
                                     self.val_loss_history,
                                     self.val_acc_history,
                                     self.save_model_path,
                                     display=False)

                utilities_models_tf.plotLearningRate(self.learning_rate_history, self.save_model_path, display=False)

            if early_stopping:
                # check if model accurary improved, and update counter if needed
                if self.val_acc_history[-1] > self.best_acc:
                    # save model checkpoint
                    print(' - Saving model checkpoint in {}'.format(self.save_model_path))
                    # save some extra parameters

                    stop = time.time()
                    self.training_time, _ = utilities_models_tf.tictoc(start, stop)
                    # self.training_time = 1234
                    self.training_epochs = epoch
                    self.best_acc = self.val_acc_history[-1]

                    # save model
                    self.save_model()

                    # reset counter
                    n_wait = 0
                else:
                    n_wait += 1
                # check max waiting is reached
                if n_wait == patience:
                    print(' -  Early stopping patient reached. Last model saved in {}'.format(self.save_model_path))
                    break

    def test(self, test_dataloader):
        '''
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
            test_gt = tf.concat([test_gt, utilities_models_tf.fix_labels(y.numpy(), self.unique_labels)], axis=0)
            test_logits = tf.concat([test_logits, self.model(x, training=False)], axis=0)

        test_stop = time.time()
        print('Overall model accuracy: {:.02f}'.format(utilities_models_tf.accuracy(test_gt, test_logits)))

        # return test predictions
        return test_gt, test_logits, test_stop-test_start

    def save_model(self):
        '''
        This saves the model along with its weights and additional information
        about its architecture and training performance
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
            'VALIDATION_ACC_HISTORY':self.val_acc_history
            }
        with open(os.path.join(self.save_model_path, 'model_summary_json.txt'), 'w') as outfile:
            json.dump(model_summary, outfile)

## CUSTOM MODEL M3

class ResNet50(object):
    '''
    Imports the ResNEt50 architecture available in tensorflow.
    The FCN is made by a dense layer of 60 nodes with ReLU activation and dropout, and final softmax.
    '''
    def __init__(self, number_of_input_channels,
                    num_classes,
                    data_augmentation=True,
                    class_weights=None,
                    model_name='ResNet50',
                    debug=False):

        self.number_of_input_channels = number_of_input_channels
        self.num_classes = num_classes
        self.debug = debug
        if class_weights is None:
            self.class_weights = np.ones([1, self.num_classes])
        else:
            self.class_weights = class_weights
        self.model_name = model_name

        inputs = Input(shape=[None, None, self.number_of_input_channels])

        # Create a data augmentation stage with normalization, horizontal flipping and rotations
        if data_augmentation is True:
            augmentor = tf.keras.Sequential([
                    layers.experimental.preprocessing.Normalization(),
                    layers.experimental.preprocessing.RandomFlip("horizontal"),
                    layers.experimental.preprocessing.RandomRotation(0.1)],
                    name='NormalizationAugmentation')
        else: # perform only normalization
            augmentor = tf.keras.Sequential([
                    layers.experimental.preprocessing.Normalization()],
                    name='Normalization')

        x = augmentor(inputs)

        # import model
        resnet = tf.keras.applications.ResNet50(include_top=False,
                                weights=None,
                                input_tensor=x,
                                input_shape=(None, None, self.number_of_input_channels))
        # FCN
        x = GlobalMaxPooling2D()(resnet.output)
        x = Dropout(rate=0.2)(x)
        x = Dense(units=60, activation='relu')(x)
        final = Dense(units=self.num_classes, activation='softmax')(x)

        # save model paramenters
        self.num_filter_start = 64
        self.kernel_size = (3,3)
        self.depth = 50
        self.num_filter_per_layer = [64, 64, 256, 64, 64, 256, 64, 64, 256, 128, 128, 512, 128, 128, 512, 128, 128, 512, 128, 128, 512, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 512, 512, 2048, 512, 512, 2048, 512, 512, 2048]
        self.custom_model = False

        # finally make the model and return
        self.model = Model(inputs=inputs, outputs=final, name=model_name)

        # print model if needed
        if self.debug is True:
            print(self.model.summary())

    def train(self, training_dataloader,
                    validation_dataloader,
                    unique_labels=None,
                    loss=('cee'),
                    start_learning_rate = 0.001,
                    scheduler='polynomial',
                    power=0.1,
                    max_epochs=200,
                    early_stopping=True,
                    patience=20,
                    save_model_path=None,
                    save_model_architecture_figure=False):
        '''
        The training loop goes as follows for one epoch
        1 - for all the batches of data in the dataloader
        2 - fix labels based on the unique labels specification
        3 - compute training logits
        4 - compute loss and accuracy
        5 - update weights using the optimizer

        When the training batches are finished run validation

        6 - compute validation logits
        7 - check for early stopping
        8 - same model if early stopping is reached or the max number of epochs
            int eh specified directory
        '''
        # define parameters useful for saving the model
        self.save_model_path = save_model_path

        # define parameters useful to store training and validation information
        self.train_loss_history, self.val_loss_history = [], []
        self.train_acc_history, self.val_acc_history = [], []
        self.initial_learning_rate = start_learning_rate
        self.scheduler = scheduler
        self.maxEpochs = max_epochs
        self.learning_rate_history = []
        self.loss = loss
        self.unique_labels = unique_labels
        self.num_validation_samples = 0
        self.num_training_samples = 0

        # save model architecture figure
        if save_model_architecture_figure is True:
            try:
                tf.keras.utils.plot_model(self.model, to_file=os.path.join(self.save_model_path, 'model_architecture.png'), show_shapes=True)
            except:
                print('Cannot save model architecture as figure. Printing instead.')
                self.model.summary()

        if early_stopping:
            self.best_acc = 0.0
            n_wait = 0

        start = time.time()
        # start looping through the epochs
        for epoch in range(self.maxEpochs):
            # initialize the variables
            epoch_train_loss, epoch_val_loss = [], []
            epoch_train_acc, epoch_val_acc = [], []

            running_loss = 0.0
            running_acc = 0.0

            # compute learning rate based on the scheduler
            if self.scheduler == 'linear':
                self.power = 1
            elif self.scheduler == 'polynomial':
                self.power = power
            else:
                raise TypeError('Invalid scheduler. given {} but expected linear or polynomial'.format(self.scheduler))

            lr = utilities_models_tf.leraningRateScheduler(self.initial_learning_rate, epoch, self.maxEpochs, power)
            self.learning_rate_history.append(lr)

            # set optimizer - using ADAM by default
            optimizer = Adam(lr=lr)

            # consume data from the dataloader
            step = 0
            for x, y in training_dataloader:
                step += 1

                # make data usable
                x = x.numpy()
                y = utilities_models_tf.fix_labels(y.numpy(), self.unique_labels)

                # save information about training patch size
                if epoch == 0 and step == 1:
                    self.batch_size = x.shape[0]
                    self.input_size = (x.shape[1], x.shape[2])

                # Open a GradientTape to record the operations run
                # during the forward pass, which enables autodifferentiation.
                with tf.GradientTape() as tape:
                    # Logits for this minibatch
                    train_logits = self.model(x, training=True)
                    # Compute the loss value for this minibatch.
                    train_loss = []
                    for l in loss:
                        if l == 'cce':
                            # compute categorical cross entropy
                            cce = tf.keras.losses.CategoricalCrossentropy()
                            train_loss.append(cce(y, train_logits))
                            # compute weighted categorical cross entropy
                        elif l == 'wcce':
                            weights = tf.constant(self.class_weights, dtype=tf.float32)
                            # one weight for each sample [batch_size, 1]
                            weights = tf.reduce_sum(weights * y, axis=1)
                            # weighted loss
                            cce = tf.keras.losses.CategoricalCrossentropy()
                            train_loss.append(cce(y, train_logits, sample_weight=weights))
                        elif l == 'sfce':
                            sfce = tfa.losses.SigmoidFocalCrossEntropy(from_logits=True)
                            train_loss.append(tf.reduce_mean(sfce(y, train_logits)))
                        else:
                            raise TypeError('Invalid loss. given {} but expected cee, wcce or sfce'.format(l))
                    # it is very important for the loss to be a tensor! If not
                    # the gradient tape will not be able to use it during the
                    # backpropagation.
                    train_loss = tf.add_n(train_loss)

                    # save metrics
                    epoch_train_loss.append(float(train_loss))
                    train_acc = utilities_models_tf.accuracy(y, train_logits)
                    epoch_train_acc.append(float(train_acc))

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                grads = tape.gradient(train_loss, self.model.trainable_weights)
                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

                # print values
                if epoch == 0:
                    print('\r', end='')
                    print('Epoch {:04d} training (counting training steps) -> {:04d} -> tr_loss:{:.4f}, tr_acc:{:.4f} \r'
                            .format(epoch+1, step, train_loss, train_acc),end='')
                else:
                    print('Epoch {:04d} training -> {:04d}/{:04d} -> tr_loss:{:.4f}, tr_acc:{:.4f} \r'
                            .format(epoch+1,
                                    step,
                                    self.num_training_samples//self.batch_size,
                                    train_loss,
                                    train_acc)
                            ,end='')

            # finisced all the training batches -> save training loss
            self.train_loss_history.append(np.mean(np.array(epoch_train_loss), axis=0))
            self.train_acc_history.append(np.mean(np.array(epoch_train_acc), axis=0))
            if epoch == 0:
                self.num_training_samples = self.batch_size*step

            # running validation loop
            step = 0
            for x, y in validation_dataloader:
                step += 1

                # make data usable
                x = x.numpy()
                y = utilities_models_tf.fix_labels(y.numpy(), self.unique_labels)

                # logits for this validation batch
                val_logits = self.model(x, training=False)  # Logits for this minibatch

                # Compute the loss value for this minibatch.
                val_loss = []
                for l in loss:
                        if l == 'cce':
                            # compute categorical cross entropy
                            cce = tf.keras.losses.CategoricalCrossentropy()
                            val_loss.append(cce(y, val_logits))
                        elif l == 'wcce':
                            # compute weighted categorical cross entropy
                            weights = tf.constant(self.class_weights, dtype=tf.float32)
                            weights = tf.reduce_sum(weights * y, axis=1)
                            cce = tf.keras.losses.CategoricalCrossentropy()
                            val_loss.append(cce(y, val_logits, sample_weight=weights))
                        elif l == 'sfce':
                            sfce = tfa.losses.SigmoidFocalCrossEntropy(from_logits=True)
                            val_loss.append(tf.reduce_mean(sfce(y, val_logits)))
                        else:
                            raise TypeError('Invalid loss. given {} but expected cee, wcce or sfce'.format(l))

                val_loss = tf.add_n(val_loss)
                epoch_val_loss.append(float(val_loss))
                val_acc = utilities_models_tf.accuracy(y, val_logits)
                epoch_val_acc.append(float(val_acc))

                # print values
                if epoch == 0:
                    print('\r', end='')
                    print('Epoch {:04d} validation (counting validation steps) -> {:04d} -> val_loss:{:.4f}, val_acc:{:.4f} \r'
                            .format(epoch+1, step, val_loss, val_acc),
                            end='')
                else:
                    print('Epoch {:04d} validation-> {:04d}/{:04d} -> val_loss:{:.4f}, val_acc:{:.4f} \r'
                            .format(epoch+1, step, self.num_validation_samples//self.batch_size, val_loss, val_acc),
                            end='')


            # finisced all the batches in the validation
            self.val_loss_history.append(np.mean(np.array(epoch_val_loss), axis=0))
            self.val_acc_history.append(np.mean(np.array(epoch_val_acc), axis=0))

            print('Epoch {:04d} -> tr_loss:{:.4f}, tr_acc:{:.4f}, val_loss:{:.4f}, val_acc:{:.4f}'.format(epoch+1,
                                self.train_loss_history[-1], self.train_acc_history[-1],
                                self.val_loss_history[-1], self.val_acc_history[-1]))
            if epoch == 0:
                self.num_validation_samples = self.batch_size*step

            if epoch % 5 == 0:
                utilities_models_tf.plotModelPerformance(self.train_loss_history,
                                     self.train_acc_history,
                                     self.val_loss_history,
                                     self.val_acc_history,
                                     self.save_model_path,
                                     display=False)

                utilities_models_tf.plotLearningRate(self.learning_rate_history, self.save_model_path, display=False)

            if early_stopping:
                # check if model accurary improved, and update counter if needed
                if self.val_acc_history[-1] > self.best_acc:
                    # save model checkpoint
                    print(' - Saving model checkpoint in {}'.format(self.save_model_path))
                    # save some extra parameters

                    stop = time.time()
                    self.training_time, _ = utilities_models_tf.tictoc(start, stop)
                    # self.training_time = 1234
                    self.training_epochs = epoch
                    self.best_acc = self.val_acc_history[-1]

                    # save model
                    self.save_model()

                    # reset counter
                    n_wait = 0
                else:
                    n_wait += 1
                # check max waiting is reached
                if n_wait == patience:
                    print(' -  Early stopping patient reached. Last model saved in {}'.format(self.save_model_path))
                    break

    def test(self, test_dataloader):
        '''
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
            test_gt = tf.concat([test_gt, utilities_models_tf.fix_labels(y.numpy(), self.unique_labels)], axis=0)
            test_logits = tf.concat([test_logits, self.model(x, training=False)], axis=0)

        test_stop = time.time()
        print('Overall model accuracy: {:.02f}'.format(utilities_models_tf.accuracy(test_gt, test_logits)))

        # return test predictions
        return test_gt, test_logits, test_stop-test_start

    def save_model(self):
        '''
        This saves the model along with its weights and additional information
        about its architecture and training performance
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
            'VALIDATION_ACC_HISTORY':self.val_acc_history
            }
        with open(os.path.join(self.save_model_path, 'model_summary_json.txt'), 'w') as outfile:
            json.dump(model_summary, outfile)

## Variational Auto-Encoder original implementation
class VAE_original(object):
    '''
    Implementation of a Variational Auto-Encoder model based on the keras
    implementation of Variational auto encoders.
    https://keras.io/examples/generative/vae/
    The encoder (and the relative decoder) is similar to the M2 architecture,
    having 3 conv layer in the encoder and 3 layers in the decoder.
    The model uses the compact representation generated by the encoder
    to both produce a label and generate back the original image (using the
    decoder). The loss to minimize is a sum of the reconstruction loss and
    the label-prediction loss.

    Steps and model architecture:
    1 - build a sampler: this will sample from the distribution of the
        compact representation of our data
    2 - build encoder: 3 layer conv with 32, 64 and 128 kernels, GroupNorm, ReLU (following original paper encoder structure)
    3 - build decoder: 3 layer traspose conv with 128, 64 and 32 kernels, ReLU

    Nice description of VAE here:https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73
    '''
    def __init__(self, number_of_input_channels,
                    num_classes,
                    input_size,
                    data_augmentation=True,
                    class_weights=None,
                    kernel_size=(5,5),
                    model_name='VAE',
                    vae_latent_dim=128,
                    debug=False):

        self.number_of_input_channels = number_of_input_channels
        self.num_classes = num_classes
        self.input_size=input_size
        self.debug = debug
        if class_weights is None:
            self.class_weights = np.ones([1, self.num_classes])
        else:
            self.class_weights = class_weights

        self.model_name = model_name
        self.vae_latent_dim = vae_latent_dim
        self.kernel_size=kernel_size

        # pre-processing steps
        inputs = Input(shape=[None, None, self.number_of_input_channels])

        # Create a data augmentation stage with normalization, horizontal flipping and rotations
        if data_augmentation is True:
            augmentor = tf.keras.Sequential(
                [
                    layers.experimental.preprocessing.Normalization(),
                    layers.experimental.preprocessing.RandomFlip("horizontal"),
                    layers.experimental.preprocessing.RandomRotation(0.1)
                ], name='NormalizationAugmentation'
            )
        else: # perform only normalization
            augmentor = tf.keras.Sequential([layers.experimental.preprocessing.Normalization()],name='Normalization')

        x = augmentor(inputs)

        augmented_norm = x

        # build sampler
        class Sampling(tf.keras.layers.Layer):
            ''' Uses (z_mean, z_log_var) to sample z, the vector encoding the image data'''

            def call(self, inputs):
                z_mean, z_log_var = inputs
                # get the dimentions of how many samples are needed
                batch = tf.shape(z_mean)[0]
                dim = tf.shape(z_mean)[1]
                # generate a normal random distribution
                epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
                # convert the random distribution to the z_mean, z_log_var distribution (reparametrization trick)
                return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        # build encoder with ResNet-like bloks
        # ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤   conv block 1
        y = Conv2D(filters=32,
                    kernel_size=self.kernel_size,
                    padding='same')(x)
        x = tfa.layers.GroupNormalization(groups=int(32/4))(y)
        x = tf.keras.layers.ReLU()(x)
        x = Conv2D(filters=32,
                    kernel_size=self.kernel_size,
                    padding='same')(x)
        # skip connection
        x = x + y
        # maxPool
        x = MaxPooling2D(pool_size=(2,2),
                    strides=2
                    )(x)

        # ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤   conv block 2
        y = x
        x = tfa.layers.GroupNormalization(groups=int(32/4))(y)
        x = tf.keras.layers.ReLU()(x)
        x = Conv2D(filters=32,
                    kernel_size=self.kernel_size,
                    padding='same')(x)
        x = tfa.layers.GroupNormalization(groups=int(32/4))(x)
        x = tf.keras.layers.ReLU()(x)
        x = Conv2D(filters=32,
                    kernel_size=self.kernel_size,
                    padding='same')(x)
        # skip connection
        x = x + y
        # maxPool
        x = MaxPooling2D(pool_size=(2,2),
                    strides=2
                    )(x)

        # ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤   conv block 3
        y = x
        x = tfa.layers.GroupNormalization(groups=int(32/4))(y)
        x = tf.keras.layers.ReLU()(x)
        x = Conv2D(filters=32,
                    kernel_size=self.kernel_size,
                    padding='same')(x)
        x = tfa.layers.GroupNormalization(groups=int(32/4))(x)
        x = tf.keras.layers.ReLU()(x)
        x = Conv2D(filters=32,
                    kernel_size=self.kernel_size,
                    padding='same')(x)
        # skip connection
        x = x + y
        # maxPool
        x = MaxPooling2D(pool_size=(2,2),
                    strides=2
                    )(x)

        # botle-neck
        x = tfa.layers.GroupNormalization(groups=int(32/4))(x)
        x = tf.keras.layers.ReLU()(x)
        y = Conv2D(filters=128,
                    kernel_size=self.kernel_size,
                    padding='same')(x)
        x = tfa.layers.GroupNormalization(groups=int(128/4))(y)
        x = tf.keras.layers.ReLU()(x)
        x = Conv2D(filters=128,
                    kernel_size=self.kernel_size,
                    padding='same')(x)
        # skip connection
        x = x + y

        # encoding vector
        encoding_vector = GlobalMaxPooling2D()(x)

        # FCN
        pred = Dropout(rate=0.2)(encoding_vector)
        pred = Dense(units=60, activation='relu')(pred)
        pred = Dense(units=self.num_classes, activation='softmax')(pred)

        # sampling
        z_mean = Dense(self.vae_latent_dim, name='z_mean')(encoding_vector)
        z_log_var = Dense(self.vae_latent_dim, name='z_log_var')(encoding_vector)
        z = Sampling()([z_mean, z_log_var])

        # build decoder
        aus_dim = [int(self.input_size[0] / 2**3), int(self.input_size[1] / 2**3)]
        x = Dense(aus_dim[0] * aus_dim[0] * self.vae_latent_dim, activation='relu')(z)
        x = tf.keras.layers.Reshape((aus_dim[0],aus_dim[0],128))(x)
        x = tf.keras.layers.Conv2DTranspose(128, 5, activation='relu', strides=2, padding='same')(x)
        x = tf.keras.layers.Conv2DTranspose(64, 5, activation='relu', strides=2, padding='same')(x)
        x = tf.keras.layers.Conv2DTranspose(32, 5, activation='relu', strides=2, padding='same')(x)

        decoder_outputs = tf.keras.layers.Conv2DTranspose(1,5,activation='tanh', padding='same')(x)
        self.model = Model(inputs=inputs, outputs=[z_mean, z_log_var, z, augmented_norm, pred, decoder_outputs], name=model_name)

        # save model paramenters
        self.num_filter_start = 32
        self.depth = 3
        self.num_filter_per_layer = [32, 32, 128]
        self.custom_model = False

    def train(self, training_dataloader,
                    validation_dataloader,
                    unique_labels=None,
                    loss=('cee'),
                    start_learning_rate = 0.001,
                    scheduler='polynomial',
                    power=0.1,
                    max_epochs=200,
                    early_stopping=True,
                    patience=20,
                    save_model_path=None,
                    save_model_architecture_figure=False,
                    vae_kl_weight=0.1,
                    vae_reconst_weight=0.1):
        '''
        The training loop goes as follows for one epoch
        1 - for all the batches of data in the dataloader
        2 - fix labels based on the unique labels specification
        3 - compute training logits
        4 - compute loss and accuracy
        5 - update weights using the optimizer

        When the training batches are finished run validation

        6 - compute validation logits
        7 - check for early stopping
        8 - same model if early stopping is reached or the max number of epochs
            int eh specified directory
        '''
        # define parameters useful for saving the model
        self.save_model_path = save_model_path

        # define parameters useful to store training and validation information
        self.train_loss_history, self.val_loss_history = [], []
        self.train_acc_history, self.val_acc_history = [], []
        self.initial_learning_rate = start_learning_rate
        self.scheduler = scheduler
        self.maxEpochs = max_epochs
        self.learning_rate_history = []
        self.loss = loss
        self.unique_labels = unique_labels
        self.num_validation_samples = 0
        self.num_training_samples = 0
        self.vae_kl_weight=vae_kl_weight
        self.vae_reconst_weight=vae_reconst_weight

        # save model architecture figure
        if save_model_architecture_figure is True:
            try:
                tf.keras.utils.plot_model(self.model, to_file=os.path.join(self.save_model_path, 'model_architecture.png'), show_shapes=True)
            except:
                print('Cannot save model architecture as figure. Printing instead.')
                self.model.summary()

        if early_stopping:
            self.best_acc = 0.0
            n_wait = 0

        start = time.time()
        # start looping through the epochs
        for epoch in range(self.maxEpochs):
            # initialize the variables
            epoch_train_loss, epoch_val_loss = [], []
            epoch_train_acc, epoch_val_acc = [], []

            running_loss = 0.0
            running_acc = 0.0

            # compute learning rate based on the scheduler
            if self.scheduler == 'linear':
                self.power = 1
            elif self.scheduler == 'polynomial':
                self.power = power
            else:
                raise TypeError('Invalid scheduler. given {} but expected linear or polynomial'.format(self.scheduler))

            lr = utilities_models_tf.leraningRateScheduler(self.initial_learning_rate, epoch, self.maxEpochs, power)
            self.learning_rate_history.append(lr)

            # set optimizer - using ADAM by default
            optimizer = Adam(lr=lr)

            # consume data from the dataloader
            step = 0
            for x, y in training_dataloader:
                step += 1

                # make data usable
                x = x.numpy()
                y = utilities_models_tf.fix_labels(y.numpy(), self.unique_labels)

                # save information about training patch size
                if epoch == 0 and step == 1:
                    self.batch_size = x.shape[0]

                # Open a GradientTape to record the operations run
                # during the forward pass, which enables autodifferentiation.
                with tf.GradientTape() as tape:
                    # Logits for this minibatch
                    z_mean, z_log_var, z, augmented_norm, train_logits, reconstruction = self.model(x, training=True)

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

                    # reconstruction loss
                    mse = tf.keras.losses.MeanSquaredError()
                    reconstruction_loss = mse(x, reconstruction) * 1000


                    #k-1 loss: Kulback-Leibler divergence that tries to make the latent space (z)
                    # of the encoded vector as regular as possible (N(0,1))
                    kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                    kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

                    # COMPUTE TOTAL LOSS
                    # it is very important for the loss to be a tensor! If not
                    # the gradient tape will not be able to use it during the
                    # backpropagation.
                    train_loss = classification_loss + self.vae_kl_weight*kl_loss + self.vae_reconst_weight*reconstruction_loss

                    # save metrics
                    epoch_train_loss.append(float(train_loss))
                    train_acc = utilities_models_tf.accuracy(y, train_logits)
                    epoch_train_acc.append(float(train_acc))

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                grads = tape.gradient(train_loss, self.model.trainable_weights)
                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

                # print values
                if epoch == 0:
                    print('\r', end='')
                    print('Epoch {:04d} training (counting training steps) -> {:04d} -> tr_loss:{:.4f}, tr_acc:{:.4f} \r'
                            .format(epoch+1, step, train_loss, train_acc),end='')
                else:
                    print('Epoch {:04d} training -> {:04d}/{:04d} -> tr_loss:{:.4f}, tr_acc:{:.4f} \r'
                            .format(epoch+1,
                                    step,
                                    self.num_training_samples//self.batch_size,
                                    train_loss,
                                    train_acc)
                            ,end='')

            # finisced all the training batches -> save training loss
            self.train_loss_history.append(np.mean(np.array(epoch_train_loss), axis=0))
            self.train_acc_history.append(np.mean(np.array(epoch_train_acc), axis=0))
            if epoch == 0:
                self.num_training_samples = self.batch_size*step

            # running validation loop
            step = 0
            for x, y in validation_dataloader:
                step += 1

                # make data usable
                x = x.numpy()
                y = utilities_models_tf.fix_labels(y.numpy(), self.unique_labels)

                # logits for this validation batch
                z_mean, z_log_var, z, augmented_norm, val_logits, reconstruction = self.model(x, training=False)  # Logits for this minibatch
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

                # reconstruction loss
                mse = tf.keras.losses.MeanSquaredError()
                reconstruction_loss = mse(x, reconstruction) * 1000


                #k-1 loss: Kulback-Leibler divergence that tries to make the latent space (z)
                # of the encoded vector as regular as possible (N(0,1))
                kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

                # COMPUTE TOTAL LOSS
                # it is very important for the loss to be a tensor! If not
                # the gradient tape will not be able to use it during the
                # backpropagation.
                val_loss = classification_loss + self.vae_kl_weight*kl_loss + self.vae_reconst_weight*reconstruction_loss

                epoch_val_loss.append(float(val_loss))
                val_acc = utilities_models_tf.accuracy(y, val_logits)
                epoch_val_acc.append(float(val_acc))

                # print values
                if epoch == 0:
                    print('\r', end='')
                    print('Epoch {:04d} validation (counting validation steps) -> {:04d} -> val_loss:{:.4f}, val_acc:{:.4f} \r'
                            .format(epoch+1, step, val_loss, val_acc),
                            end='')
                else:
                    print('Epoch {:04d} validation-> {:04d}/{:04d} -> val_loss:{:.4f}, val_acc:{:.4f} \r'
                            .format(epoch+1, step, self.num_validation_samples//self.batch_size, val_loss, val_acc),
                            end='')


            # finisced all the batches in the validation
            self.val_loss_history.append(np.mean(np.array(epoch_val_loss), axis=0))
            self.val_acc_history.append(np.mean(np.array(epoch_val_acc), axis=0))

            print('Epoch {:04d} -> tr_loss:{:.4f}, tr_acc:{:.4f}, val_loss:{:.4f}, val_acc:{:.4f}'.format(epoch+1,
                                self.train_loss_history[-1], self.train_acc_history[-1],
                                self.val_loss_history[-1], self.val_acc_history[-1]))
            if epoch == 0:
                self.num_validation_samples = self.batch_size*step

            if epoch % 2 == 0:
                utilities_models_tf.plotModelPerformance(self.train_loss_history,
                                     self.train_acc_history,
                                     self.val_loss_history,
                                     self.val_acc_history,
                                     self.save_model_path,
                                     display=False)

                utilities_models_tf.plotLearningRate(self.learning_rate_history, self.save_model_path, display=False)

                utilities_models_tf.plotVAEreconstruction(x, reconstruction, epoch, self.save_model_path)

            if early_stopping:
                # check if model accurary improved, and update counter if needed
                if self.val_acc_history[-1] > self.best_acc:
                    # save model checkpoint
                    print(' - Saving model checkpoint in {}'.format(self.save_model_path))
                    # save some extra parameters

                    stop = time.time()
                    self.training_time, _ = utilities_models_tf.tictoc(start, stop)
                    # self.training_time = 1234
                    self.training_epochs = epoch
                    self.best_acc = self.val_acc_history[-1]

                    # save model
                    self.save_model()

                    # reset counter
                    n_wait = 0
                else:
                    n_wait += 1
                # check max waiting is reached
                if n_wait == patience:
                    print(' -  Early stopping patient reached. Last model saved in {}'.format(self.save_model_path))
                    break

    def test(self, test_dataloader):
        '''
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
            test_gt = tf.concat([test_gt, utilities_models_tf.fix_labels(y.numpy(), self.unique_labels)], axis=0)
            _, _, z, _, aus_logits, _ = self.model(x, training=False)
            test_logits = tf.concat([test_logits, aus_logits], axis=0)

        test_stop = time.time()
        print('Overall model accuracy: {:.02f}'.format(utilities_models_tf.accuracy(test_gt, test_logits)))

        # return test predictions
        return test_gt, test_logits, test_stop-test_start

    def save_model(self):
        '''
        This saves the model along with its weights and additional information
        about its architecture and training performance
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
            'KL_LOSS_WEIGHT':self.vae_kl_weight,
            'RECONSTRUCTION_LOSS_WEIGHT':self.vae_reconst_weight,
            'VAE_LATENT_SPACE_DIM':self.vae_latent_dim
            }
        with open(os.path.join(self.save_model_path, 'model_summary_json.txt'), 'w') as outfile:
            json.dump(model_summary, outfile)

## Variational Auto-Encoder own implementation V1
class VAE1(object):
    '''
    Implementation of a Variational Auto-Encoder model based on the keras
    implementation of Variational auto encoders.
    https://keras.io/examples/generative/vae/
    The encoder (and the relative decoder) is similar to the M2 architecture,
    having 3 conv layer in the encoder and 3 layers in the decoder.
    The model uses the compact representation generated by the encoder
    to both produce a label and generate back the original image (using the
    decoder). The loss to minimize is a sum of the reconstruction loss and
    the label-prediction loss.

    Steps and model architecture:
    1 - build a sampler: this will sample from the distribution of the
        compact representation of our data
    2 - build encoder: 3 layer conv with 32, 64 and 128 kernels, GroupNorm, ReLU (following original paper encoder structure)
    3 - build decoder: 3 layer traspose conv with 128, 64 and 32 kernels, ReLU

    Nice description of VAE here:https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73
    '''
    def __init__(self, number_of_input_channels,
                    num_classes,
                    input_size,
                    data_augmentation=True,
                    class_weights=None,
                    kernel_size=(5,5),
                    model_name='VAE1',
                    vae_latent_dim=128,
                    debug=False):

        self.number_of_input_channels = number_of_input_channels
        self.num_classes = num_classes
        self.input_size=input_size
        self.debug = debug
        if class_weights is None:
            self.class_weights = np.ones([1, self.num_classes])
        else:
            self.class_weights = class_weights

        self.model_name = model_name
        self.vae_latent_dim = vae_latent_dim
        self.kernel_size=kernel_size

        # pre-processing steps
        inputs = Input(shape=[None, None, self.number_of_input_channels])

        # Create a data augmentation stage with normalization, horizontal flipping and rotations
        if data_augmentation is True:
            augmentor = tf.keras.Sequential(
                [
                    layers.experimental.preprocessing.Normalization(),
                    layers.experimental.preprocessing.RandomFlip("horizontal"),
                    layers.experimental.preprocessing.RandomRotation(0.1)
                ], name='NormalizationAugmentation'
            )
        else: # perform only normalization
            augmentor = tf.keras.Sequential([layers.experimental.preprocessing.Normalization()],name='Normalization')

        x = augmentor(inputs)

        augmented_norm = x

        # build sampler
        class Sampling(tf.keras.layers.Layer):
            ''' Uses (z_mean, z_log_var) to sample z, the vector encoding the image data'''

            def call(self, inputs):
                z_mean, z_log_var = inputs
                # get the dimentions of how many samples are needed
                batch = tf.shape(z_mean)[0]
                dim = tf.shape(z_mean)[1]
                # generate a normal random distribution
                epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
                # convert the random distribution to the z_mean, z_log_var distribution (reparametrization trick)
                return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        # CNN encoder
        n_filters = [32,64]
        for i in n_filters:
            x = Conv2D(filters=i,
                        kernel_size=kernel_size)(x)
            x = Conv2D(filters=i,
                        kernel_size=kernel_size)(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            x = MaxPooling2D(pool_size=(2,2),
                        strides=2
                        )(x)

        # bottle-neck
        x = Conv2D(filters=128, kernel_size=kernel_size)(x)
        x = Conv2D(filters=128, kernel_size=kernel_size)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # encoding vector
        encoding_vector = GlobalMaxPooling2D()(x)

        # FCN
        pred = Dropout(rate=0.2)(encoding_vector)
        pred = Dense(units=60, activation='relu')(pred)
        pred = Dense(units=self.num_classes, activation='softmax')(pred)

        # sampling
        z_mean = Dense(self.vae_latent_dim, name='z_mean')(encoding_vector)
        z_log_var = Dense(self.vae_latent_dim, name='z_log_var')(encoding_vector)
        z = Sampling()([z_mean, z_log_var])

        # build decoder
        aus_dim = [int(self.input_size[0] / 2**2), int(self.input_size[1] / 2**2)]
        x = Dense(aus_dim[0] * aus_dim[0] * self.vae_latent_dim, activation='relu')(z)
        x = tf.keras.layers.Reshape((aus_dim[0],aus_dim[0],self.vae_latent_dim))(x)
        for i in reversed(n_filters):
            x = tf.keras.layers.Conv2DTranspose(i, 3, strides=2, padding='same')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)

        decoder_outputs = tf.keras.layers.Conv2DTranspose(1, 3, activation='tanh', padding='same')(x)
        self.model = Model(inputs=inputs, outputs=[z_mean, z_log_var, z, augmented_norm, pred, decoder_outputs], name=model_name)

        # save model paramenters
        self.num_filter_start = 32
        self.depth = 2
        self.num_filter_per_layer = [32, 64, 128]
        self.custom_model = False

    def train(self, training_dataloader,
                    validation_dataloader,
                    unique_labels=None,
                    loss=('cee'),
                    start_learning_rate = 0.001,
                    scheduler='polynomial',
                    power=0.1,
                    max_epochs=200,
                    early_stopping=True,
                    patience=20,
                    save_model_path=None,
                    save_model_architecture_figure=False,
                    vae_kl_weight=0.1,
                    vae_reconst_weight=0.1):
        '''
        The training loop goes as follows for one epoch
        1 - for all the batches of data in the dataloader
        2 - fix labels based on the unique labels specification
        3 - compute training logits
        4 - compute loss and accuracy
        5 - update weights using the optimizer

        When the training batches are finished run validation

        6 - compute validation logits
        7 - check for early stopping
        8 - same model if early stopping is reached or the max number of epochs
            int eh specified directory
        '''
        # define parameters useful for saving the model
        self.save_model_path = save_model_path

        # define parameters useful to store training and validation information
        self.train_loss_history, self.val_loss_history = [], []
        self.train_acc_history, self.val_acc_history = [], []
        self.initial_learning_rate = start_learning_rate
        self.scheduler = scheduler
        self.maxEpochs = max_epochs
        self.learning_rate_history = []
        self.loss = loss
        self.unique_labels = unique_labels
        self.num_validation_samples = 0
        self.num_training_samples = 0
        self.vae_kl_weight=vae_kl_weight
        self.vae_reconst_weight=vae_reconst_weight

        # save model architecture figure
        if save_model_architecture_figure is True:
            try:
                tf.keras.utils.plot_model(self.model, to_file=os.path.join(self.save_model_path, 'model_architecture.png'), show_shapes=True)
            except:
                print('Cannot save model architecture as figure. Printing instead.')
                self.model.summary()

        if early_stopping:
            self.best_acc = 0.0
            n_wait = 0

        start = time.time()
        # start looping through the epochs
        for epoch in range(self.maxEpochs):
            # initialize the variables
            epoch_train_loss, epoch_val_loss = [], []
            epoch_train_acc, epoch_val_acc = [], []

            running_loss = 0.0
            running_acc = 0.0

            # compute learning rate based on the scheduler
            if self.scheduler == 'linear':
                self.power = 1
            elif self.scheduler == 'polynomial':
                self.power = power
            else:
                raise TypeError('Invalid scheduler. given {} but expected linear or polynomial'.format(self.scheduler))

            lr = utilities_models_tf.leraningRateScheduler(self.initial_learning_rate, epoch, self.maxEpochs, power)
            self.learning_rate_history.append(lr)

            # set optimizer - using ADAM by default
            optimizer = Adam(lr=lr)

            # consume data from the dataloader
            step = 0
            for x, y in training_dataloader:
                step += 1

                # make data usable
                x = x.numpy()
                y = utilities_models_tf.fix_labels(y.numpy(), self.unique_labels)

                # save information about training patch size
                if epoch == 0 and step == 1:
                    self.batch_size = x.shape[0]

                # Open a GradientTape to record the operations run
                # during the forward pass, which enables autodifferentiation.
                with tf.GradientTape() as tape:
                    # Logits for this minibatch
                    z_mean, z_log_var, z, augmented_norm, train_logits, reconstruction = self.model(x, training=True)

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

                    # reconstruction loss
                    mse = tf.keras.losses.MeanSquaredError()
                    reconstruction_loss = mse(x, reconstruction)
                    # print('Reconstruction loss \n', reconstruction_loss)


                    #k-1 loss: Kulback-Leibler divergence that tries to make the latent space (z)
                    # of the encoded vector as regular as possible (N(0,1))
                    kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                    kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
                    # print('KL loss \n',kl_loss)
                    # print('Classification loss \n',classification_loss)
                    # COMPUTE TOTAL LOSS
                    # it is very important for the loss to be a tensor! If not
                    # the gradient tape will not be able to use it during the
                    # backpropagation.
                    train_loss = classification_loss + self.vae_kl_weight*kl_loss + self.vae_reconst_weight*reconstruction_loss

                    # save metrics
                    epoch_train_loss.append(float(train_loss))
                    train_acc = utilities_models_tf.accuracy(y, train_logits)
                    epoch_train_acc.append(float(train_acc))

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                grads = tape.gradient(train_loss, self.model.trainable_weights)
                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

                # print values
                if epoch == 0:
                    print('\r', end='')
                    print('Epoch {:04d} training (counting training steps) -> {:04d} -> tr_loss:{:.4f}, tr_acc:{:.4f} \r'
                            .format(epoch+1, step, train_loss, train_acc),end='')
                else:
                    print('Epoch {:04d} training -> {:04d}/{:04d} -> tr_loss:{:.4f}, tr_acc:{:.4f} \r'
                            .format(epoch+1,
                                    step,
                                    self.num_training_samples//self.batch_size,
                                    train_loss,
                                    train_acc)
                            ,end='')

            # finisced all the training batches -> save training loss
            self.train_loss_history.append(np.mean(np.array(epoch_train_loss), axis=0))
            self.train_acc_history.append(np.mean(np.array(epoch_train_acc), axis=0))
            if epoch == 0:
                self.num_training_samples = self.batch_size*step

            # running validation loop
            step = 0
            for x, y in validation_dataloader:
                step += 1

                # make data usable
                x = x.numpy()
                y = utilities_models_tf.fix_labels(y.numpy(), self.unique_labels)

                # logits for this validation batch
                z_mean, z_log_var, z, augmented_norm, val_logits, reconstruction = self.model(x, training=False)  # Logits for this minibatch
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

                # reconstruction loss
                mse = tf.keras.losses.MeanSquaredError()
                reconstruction_loss = mse(x, reconstruction) * 1000


                #k-1 loss: Kulback-Leibler divergence that tries to make the latent space (z)
                # of the encoded vector as regular as possible (N(0,1))
                kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

                # COMPUTE TOTAL LOSS
                # it is very important for the loss to be a tensor! If not
                # the gradient tape will not be able to use it during the
                # backpropagation.
                val_loss = classification_loss + self.vae_kl_weight*kl_loss + self.vae_reconst_weight*reconstruction_loss

                epoch_val_loss.append(float(val_loss))
                val_acc = utilities_models_tf.accuracy(y, val_logits)
                epoch_val_acc.append(float(val_acc))

                # print values
                if epoch == 0:
                    print('\r', end='')
                    print('Epoch {:04d} validation (counting validation steps) -> {:04d} -> val_loss:{:.4f}, val_acc:{:.4f} \r'
                            .format(epoch+1, step, val_loss, val_acc),
                            end='')
                else:
                    print('Epoch {:04d} validation-> {:04d}/{:04d} -> val_loss:{:.4f}, val_acc:{:.4f} \r'
                            .format(epoch+1, step, self.num_validation_samples//self.batch_size, val_loss, val_acc),
                            end='')


            # finisced all the batches in the validation
            self.val_loss_history.append(np.mean(np.array(epoch_val_loss), axis=0))
            self.val_acc_history.append(np.mean(np.array(epoch_val_acc), axis=0))

            print('Epoch {:04d} -> tr_loss:{:.4f}, tr_acc:{:.4f}, val_loss:{:.4f}, val_acc:{:.4f}'.format(epoch+1,
                                self.train_loss_history[-1], self.train_acc_history[-1],
                                self.val_loss_history[-1], self.val_acc_history[-1]))
            if epoch == 0:
                self.num_validation_samples = self.batch_size*step

            if epoch % 2 == 0:
                utilities_models_tf.plotModelPerformance(self.train_loss_history,
                                     self.train_acc_history,
                                     self.val_loss_history,
                                     self.val_acc_history,
                                     self.save_model_path,
                                     display=False)

                utilities_models_tf.plotLearningRate(self.learning_rate_history, self.save_model_path, display=False)

                utilities_models_tf.plotVAEreconstruction(x, reconstruction, epoch, self.save_model_path)

            if early_stopping:
                # check if model accurary improved, and update counter if needed
                if self.val_acc_history[-1] > self.best_acc:
                    # save model checkpoint
                    print(' - Saving model checkpoint in {}'.format(self.save_model_path))
                    # save some extra parameters

                    stop = time.time()
                    self.training_time, _ = utilities_models_tf.tictoc(start, stop)
                    # self.training_time = 1234
                    self.training_epochs = epoch
                    self.best_acc = self.val_acc_history[-1]

                    # save model
                    self.save_model()

                    # reset counter
                    n_wait = 0
                else:
                    n_wait += 1
                # check max waiting is reached
                if n_wait == patience:
                    print(' -  Early stopping patient reached. Last model saved in {}'.format(self.save_model_path))
                    break

    def test(self, test_dataloader):
        '''
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
            test_gt = tf.concat([test_gt, utilities_models_tf.fix_labels(y.numpy(), self.unique_labels)], axis=0)
            _, _, z, _, aus_logits, _ = self.model(x, training=False)
            test_logits = tf.concat([test_logits, aus_logits], axis=0)

        test_stop = time.time()
        print('Overall model accuracy: {:.02f}'.format(utilities_models_tf.accuracy(test_gt, test_logits)))

        # return test predictions
        return test_gt, test_logits, test_stop-test_start

    def save_model(self):
        '''
        This saves the model along with its weights and additional information
        about its architecture and training performance
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
            'KL_LOSS_WEIGHT':self.vae_kl_weight,
            'RECONSTRUCTION_LOSS_WEIGHT':self.vae_reconst_weight,
            'VAE_LATENT_SPACE_DIM':self.vae_latent_dim
            }
        with open(os.path.join(self.save_model_path, 'model_summary_json.txt'), 'w') as outfile:
            json.dump(model_summary, outfile)






