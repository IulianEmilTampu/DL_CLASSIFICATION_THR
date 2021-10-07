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
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Conv2DTranspose, Dropout, LeakyReLU, Dense, GlobalMaxPooling2D, Flatten, Reshape, Softmax
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
                    norm_layer=None,
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

        inputs = Input(shape=[input_size[0], input_size[1], self.number_of_input_channels])

        # # Create a data augmentation stage with normalization, horizontal flipping and rotations
        # if data_augmentation is True:
        #     augmentor = tf.keras.Sequential([
        #             layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        #             layers.experimental.preprocessing.RandomRotation(0.02),
        #             layers.experimental.preprocessing.RandomCrop(input_size[0], input_size[0])],
        #             name='AugmentationAndCrop')
        # else: # perform only normalization
        #     augmentor = tf.keras.Sequential([
        #             layers.experimental.preprocessing.RandomCrop(input_size[0], input_size[0])],
        #             name='Crop')
        #
        # aug = augmentor(inputs)
        #
        # if norm_layer is not None:
        #     norm = norm_layer(aug)
        # else:
        #     norm = aug

        x = utilities_models_tf.augmentor(inputs)

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
        # x = GlobalMaxPooling2D()(x)
        x = Flatten()(x)
        # x = Dropout(rate=0.2)(x)
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
                    input_size,
                    data_augmentation=True,
                    norm_layer=None,
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

        # # Create a data augmentation stage with normalization, horizontal flipping and rotations
        # if data_augmentation is True:
        #     augmentor = tf.keras.Sequential([
        #             layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        #             layers.experimental.preprocessing.RandomRotation(0.02),
        #             layers.experimental.preprocessing.RandomCrop(input_size[0], input_size[0])],
        #             name='AugmentationAndCrop')
        # else: # perform only normalization
        #     augmentor = tf.keras.Sequential([
        #             layers.experimental.preprocessing.RandomCrop(input_size[0], input_size[0])],
        #             name='Crop')
        #
        # aug = augmentor(inputs)
        #
        # if norm_layer is not None:
        #     norm = norm_layer(aug)
        # else:
        #     norm = aug

        x = utilities_models_tf.augmentor(inputs)


        # build model
        n_filters = [32, 64, 128]
        x = norm
        for i in n_filters:
            x = Conv2D(filters=i,
                        kernel_size=self.kernel_size)(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            x = MaxPooling2D(pool_size=(2,2),
                        strides=2
                        )(x)

        # classifier
        # x = layers.SpatialDropout2D(0.2)(x)
        # x = Conv2D(self.num_classes, [5,5], padding='same', activation='relu')(x)
        # final = layers.GlobalMaxPooling2D()(x)
        x = GlobalMaxPooling2D()(x)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu')(x)
        # x = Dropout(0.2)(x)
        final = Dense(self.num_classes, activation='softmax')(x)

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
                    norm_layer=None,
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

        # # Create a data augmentation stage with normalization, horizontal flipping and rotations
        # if data_augmentation is True:
        #     augmentor = tf.keras.Sequential([
        #             layers.experimental.preprocessing.Normalization(),
        #             layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        #             layers.experimental.preprocessing.RandomRotation(0.02)],
        #             name='NormalizationAugmentation')
        # else: # perform only normalization
        #     augmentor = tf.keras.Sequential([
        #             layers.experimental.preprocessing.Normalization()],
        #             name='Normalization')
        #
        # x = augmentor(inputs)
        # aug = x

        x = utilities_models_tf.augmentor(inputs)

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

        # classifier
        # x = layers.SpatialDropout2D(0.2)(x)
        # x = Conv2D(self.num_classes, [5,5], padding='same', activation='relu')(x)
        # final = layers.GlobalMaxPooling2D()(x)
        x = GlobalMaxPooling2D()(x)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu')(x)
        # x = Dropout(0.2)(x)
        final = Dense(self.num_classes, activation='softmax')(x)

        # save model paramenters
        self.num_filter_start = n_filters[0]
        self.depth = len(n_filters)
        self.num_filter_per_layer = n_filters
        self.custom_model = False

        # finally make the model and return
        # self.model = Model(inputs=inputs, outputs=[final, aug], name=model_name)
        self.model = Model(inputs=inputs, outputs=final, name=model_name)

        # print model if needed
        if self.debug is True:
            print(self.model.summary())

## Using multiple convs at the same time
class M4(object):
    '''
   The encoder of the model uses dilated convolutions starting with with a 3x3
   kernel and with three dilations 0, 1 and 2 (-> simulate a 3x3, 5x5 and 7x7).
   The result is then concatenated along with the initial input and passed
   through an activation function.
    '''
    def __init__(self, number_of_input_channels,
                    num_classes,
                    normalization='BatchNorm',
                    dropout_rate=0.2,
                    data_augmentation=True,
                    class_weights=None,
                    kernel_size=(5,5),
                    model_name='M4',
                    debug=False):

        self.number_of_input_channels = number_of_input_channels
        self.num_classes = num_classes
        self.debug = debug
        if class_weights is None:
            self.class_weights = np.ones([1, self.num_classes])
        else:
            self.class_weights = class_weights

        self.model_name = model_name
        self.kernel_size=kernel_size

        inputs = Input(shape=[None, None, self.number_of_input_channels])

        # save augmented image to compute reconstruction
        if data_augmentation:
            x = utilities_models_tf.augmentor(inputs)
        else:
            x = inputs

        # build encoder with ResNet-like bloks
        def Encoder_conv_block(inputs, n_filters):
            '''
            Takes the input and processes it through three a 3x3 kernel with
            dilation of 0, 1 and 2 (simulating a 3x3, 5x5 and 7x7 convolution).
            The result is then concatenated along with the initial input,
            convolved through a 1x1 convolution and passed through an activation function.
            '''
            y = Conv2D(filters=n_filters,kernel_size=(1,1),padding='same', dilation_rate=1)(inputs)
            # perform conv with different kernel sizes
            conv3 = Conv2D(filters=n_filters,kernel_size=(3,3),padding='same', dilation_rate=1)(inputs)
            conv5 = Conv2D(filters=n_filters,kernel_size=(3,3),padding='same', dilation_rate=2)(inputs)
            conv7 = Conv2D(filters=n_filters,kernel_size=(3,3),padding='same', dilation_rate=3)(inputs)
            # concatenate
            x = tf.concat([conv3, conv5, conv7, y], axis=-1)
            # conbine the information af all filters together
            x = Conv2D(filters=n_filters,kernel_size=(1,1),padding='same')(x)
            # normalization
            if normalization == 'BatchNorm':
                x = BatchNormalization()(x)
            elif normalization == 'GroupNorm':
                x = tfa.layers.GroupNormalization(groups=int(n_filters/4))(x)
            else:
                raise ValueError(f'Not recognized normalization type. Expecting BatchNorm or GroupNorm but given {normalization}')
            # through the activation
            return tf.keras.layers.LeakyReLU()(x)

        # build encoder
        x = Encoder_conv_block(x, n_filters=16)
        x = MaxPooling2D(pool_size=(2,2),strides=2)(x)
        x = Encoder_conv_block(x, n_filters=32)
        x = MaxPooling2D(pool_size=(2,2),strides=2)(x)
        x = Encoder_conv_block(x, n_filters=64)
        x = MaxPooling2D(pool_size=(2,2),strides=2)(x)

        # bottle-neck
        x = Conv2D(filters=128, kernel_size=self.kernel_size, padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters=128, kernel_size=self.kernel_size, padding='same')(x)
        # x = tfa.layers.GroupNormalization(groups=int(128/4))(x)
        x = tf.keras.layers.LeakyReLU()(x)

        # encoding vector
        encoding_vector = GlobalMaxPooling2D()(x)

        # FCN
        pred = Dropout(rate=dropout_rate)(encoding_vector)
        pred = Dense(units=60, activation='relu')(pred)
        pred = Dropout(rate=dropout_rate)(pred)
        pred = Dense(units=self.num_classes)(pred)

        self.model = Model(inputs=inputs, outputs=pred, name=model_name)

        # save model paramenters
        self.num_filter_start = 16
        self.depth = 3
        self.num_filter_per_layer = [16, 32, 64]
        self.custom_model = False
        self.dropout_rate=dropout_rate
        self.normalization=normalization

        # print model if needed
        if self.debug is True:
            print(self.model.summary())

## ResNet50

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

        # # Create a data augmentation stage with normalization, horizontal flipping and rotations
        # if data_augmentation is True:
        #     augmentor = tf.keras.Sequential([
        #             layers.experimental.preprocessing.Normalization(),
        #             layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        #             layers.experimental.preprocessing.RandomRotation(0.02)],
        #             name='NormalizationAugmentation')
        # else: # perform only normalization
        #     augmentor = tf.keras.Sequential([
        #             layers.experimental.preprocessing.Normalization()],
        #             name='Normalization')

        x = utilities_models_tf.augmentor(inputs)

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
        self.num_filter_start = 'ResNet50'
        self.kernel_size = 'ResNet50'
        self.depth = 50
        self.num_filter_per_layer = 'ResNet50'
        self.custom_model = False

        # finally make the model and return
        self.model = Model(inputs=inputs, outputs=final, name=model_name)

        # print model if needed
        if self.debug is True:
            print(self.model.summary())


## Inception_v3

class InceptionV3(object):
    '''
    Imports the InceptionV3 architecture available in tensorflow.
    The FCN is made by a dense layer of 60 nodes with ReLU activation and dropout, and final softmax.
    '''
    def __init__(self, number_of_input_channels,
                    num_classes,
                    input_size,
                    data_augmentation=True,
                    class_weights=None,
                    model_name='InceptionV3',
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

        # # Create a data augmentation stage with normalization, horizontal flipping and rotations
        # if data_augmentation is True:
        #     augmentor = tf.keras.Sequential([
        #             layers.experimental.preprocessing.Normalization(),
        #             layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        #             layers.experimental.preprocessing.RandomRotation(0.02),
        #             layers.experimental.preprocessing.RandomCrop(input_size[0], input_size[0])],
        #             name='NormalizationAugmentationCropping')
        # else: # perform only normalization
        #     augmentor = tf.keras.Sequential([
        #             layers.experimental.preprocessing.Normalization(),
        #             layers.experimental.preprocessing.RandomCrop(input_size[0], input_size[0])],
        #             name='NormalizationCrop')
        #
        # x = augmentor(inputs)

        x = utilities_models_tf.augmentor(inputs)

        # import model
        inception = tf.keras.applications.inception_v3.InceptionV3(include_top=True,
                                classes=self.num_classes,
                                weights=None,
                                input_tensor=x,
                                input_shape=(input_size[0], input_size[0], self.number_of_input_channels))

        # save model paramenters
        self.num_filter_start = 'Inception'
        self.kernel_size = 'Inception'
        self.depth = 159
        self.num_filter_per_layer = 'Inception'
        self.custom_model = False

        # finally make the model and return
        self.model = Model(inputs=inputs, outputs=inception.output, name=model_name)

        # print model if needed
        if self.debug is True:
            print(self.model.summary())


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
                    norm_layer=None,
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

        # # Create a data augmentation stage with normalization, horizontal flipping and rotations
        # if data_augmentation is True:
        #     augmentor = tf.keras.Sequential([
        #             layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        #             layers.experimental.preprocessing.RandomRotation(0.02),
        #             layers.experimental.preprocessing.RandomCrop(input_size[0], input_size[0])],
        #             name='AugmentationAndCrop')
        # else: # perform only normalization
        #     augmentor = tf.keras.Sequential([
        #             layers.experimental.preprocessing.RandomCrop(input_size[0], input_size[0])],
        #             name='Crop')
        #
        # aug = augmentor(inputs)
        #
        # if norm_layer is not None:
        #     augmented_norm = norm_layer(aug)
        # else:
        #     augmented_norm = aug

        x = utilities_models_tf.augmentor(inputs)

        augmented_norm = tf.clip_by_value(x, clip_value_min=-1, clip_value_max=1)

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
                    padding='same')(augmented_norm)
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
        # x = Dense(aus_dim[0] * aus_dim[0] * self.vae_latent_dim, activation='relu')(z)
        # x = tf.keras.layers.Reshape((aus_dim[0],aus_dim[0],self.vae_latent_dim))(x)
        # x = tf.keras.layers.Conv2DTranspose(128, 3, activation='relu', strides=2, padding='same')(x)
        # x = tf.keras.layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
        # x = tf.keras.layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)
        #
        # decoder_outputs = tf.keras.layers.Conv2DTranspose(self.number_of_input_channels,5, activation='tanh', padding='same')(x)

        def resize_convolution(x, final_shape, filters):
            upsample = tf.image.resize(images=x,
                                            size=final_shape,
                                            method=tf.image.ResizeMethod.BILINEAR)
            up_conv = Conv2D(filters=filters,
                            kernel_size=3,
                            padding='same',
                            activation='relu')(upsample)
            return up_conv

        x = Dense(aus_dim[0] * aus_dim[0] * self.vae_latent_dim)(z)
        # x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Reshape((aus_dim[0],aus_dim[0],self.vae_latent_dim))(x)
        x = resize_convolution(x, final_shape=(aus_dim[0]*2, aus_dim[0]*2), filters=128)
        x = resize_convolution(x, final_shape=(aus_dim[0]*4, aus_dim[0]*4), filters=64)
        x = resize_convolution(x, final_shape=(aus_dim[0]*8, aus_dim[0]*8), filters=32)
        decoder_outputs = Conv2D(self.number_of_input_channels, 5, activation='tanh', padding='same')(x)

        self.model = Model(inputs=inputs, outputs=[pred, decoder_outputs, augmented_norm, z_mean, z_log_var, z], name=model_name)

        # save model paramenters
        self.num_filter_start = 32
        self.depth = 3
        self.num_filter_per_layer = [32, 32, 128]
        self.custom_model = False


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
                    norm_layer=None,
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

        # # Create a data augmentation stage with normalization, horizontal flipping and rotations
        # if data_augmentation is True:
        #     augmentor = tf.keras.Sequential([
        #             layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        #             layers.experimental.preprocessing.RandomRotation(0.02),
        #             layers.experimental.preprocessing.RandomCrop(input_size[0], input_size[0])],
        #             name='AugmentationAndCrop')
        # else: # perform only normalization
        #     augmentor = tf.keras.Sequential([
        #             layers.experimental.preprocessing.RandomCrop(input_size[0], input_size[0])],
        #             name='Crop')
        #
        # aug = augmentor(inputs)
        #
        # if norm_layer is not None:
        #     augmented_norm = norm_layer(aug)
        # else:
        #     augmented_norm = aug
        x = utilities_models_tf.augmentor(inputs)

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
        x = augmented_norm
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
        self.model = Model(inputs=inputs, outputs=[pred, decoder_outputs, augmented_norm, z_mean, z_log_var, z], name=model_name)

        # save model paramenters
        self.num_filter_start = 32
        self.depth = 2
        self.num_filter_per_layer = [32, 64, 128]
        self.custom_model = False

## Variational Auto-Encoder 2
class VAE2(object):
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

        # # Create a data augmentation stage with normalization, horizontal flipping and rotations
        # if data_augmentation is True:
        #     augmentor = tf.keras.Sequential(
        #         [
        #             layers.experimental.preprocessing.Normalization(),
        #             layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        #             layers.experimental.preprocessing.RandomRotation(0.02)],
        #             name='NormalizationAugmentation'
        #     )
        # else: # perform only normalization
        #     augmentor = tf.keras.Sequential([layers.experimental.preprocessing.Normalization()],name='Normalization')

        x = utilities_models_tf.augmentor(inputs)

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
        self.model = Model(inputs=inputs, outputs=[pred, decoder_outputs, augmented_norm, z_mean, z_log_var, z], name=model_name)

        # save model paramenters
        self.num_filter_start = 32
        self.depth = 3
        self.num_filter_per_layer = [32, 32, 128]
        self.custom_model = False

## Variational Auto-Encoder 3
class VAE3(object):
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

        # # Create a data augmentation stage with normalization, horizontal flipping and rotations
        # if data_augmentation is True:
        #     augmentor = tf.keras.Sequential(
        #         [
        #             layers.experimental.preprocessing.Normalization(),
        #             layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        #             layers.experimental.preprocessing.RandomRotation(0.02)],
        #             name='NormalizationAugmentation'
        #     )
        # else: # perform only normalization
        #     augmentor = tf.keras.Sequential([layers.experimental.preprocessing.Normalization()],name='Normalization')

        x = utilities_models_tf.augmentor(inputs)

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

        # CLASSIFIER BANCH
        c = Conv2D(filters=self.num_classes,
                    kernel_size=self.kernel_size,
                    padding='same')(x)
        c = GlobalMaxPooling2D()(c)
        pred = Softmax()(c)

        # VAE BRANCH
        # encoding vector
        encoding_vector = GlobalMaxPooling2D()(x)

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
        self.model = Model(inputs=inputs, outputs=[pred, decoder_outputs, augmented_norm, z_mean, z_log_var, z], name=model_name)

        # save model paramenters
        self.num_filter_start = 32
        self.depth = 3
        self.num_filter_per_layer = [32, 32, 128]
        self.custom_model = False

## VAE4 with M4 encoder
class VAE4(object):

    def __init__(self, number_of_input_channels,
                    num_classes,
                    input_size,
                    data_augmentation=True,
                    class_weights=None,
                    kernel_size=(5,5),
                    model_name='VAE4',
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

        # # Create a data augmentation stage with normalization, horizontal flipping and rotations
        # if data_augmentation is True:
        #     augmentor = tf.keras.Sequential(
        #         [
        #             layers.experimental.preprocessing.Normalization(),
        #             layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        #             layers.experimental.preprocessing.RandomRotation(0.02)],
        #             name='NormalizationAugmentation'
        #     )
        # else: # perform only normalization
        #     augmentor = tf.keras.Sequential([layers.experimental.preprocessing.Normalization()],name='Normalization')
        #
        # x = augmentor(inputs)

        x = utilities_models_tf.augmentor(inputs)

        # clip [-1,1] since using tanh as output of the decoder
        augmented_norm = tf.clip_by_value(x, clip_value_min=-1, clip_value_max=1)

        # build encoder with Inception-like bloks
        def Encoder_conv_block(inputs, n_filters):
            '''
            Takes the input and processes it through three a 3x3 kernel with
            dilation of 0, 1 and 2 (simulating a 3x3, 5x5 and 7x7 convolution).
            The result is then concatenated along with the initial input,
            convolved through a 1x1 convolution and passed through an activation function.
            '''
            y = Conv2D(filters=n_filters,kernel_size=(1,1),padding='same', dilation_rate=1)(inputs)
            # perform conv with different kernel sizes
            conv3 = Conv2D(filters=n_filters,kernel_size=(3,3),padding='same', dilation_rate=1)(inputs)
            conv5 = Conv2D(filters=n_filters,kernel_size=(3,3),padding='same', dilation_rate=2)(inputs)
            conv7 = Conv2D(filters=n_filters,kernel_size=(3,3),padding='same', dilation_rate=3)(inputs)
            # concatenate
            x = tf.concat([conv3, conv5, conv7, y], axis=-1)
            # conbine the information af all filters together
            x = Conv2D(filters=n_filters,kernel_size=(1,1),padding='same')(x)
            # normalization
            # x = tfa.layers.GroupNormalization(groups=int(n_filters/4))(x)
            x = BatchNormalization()(x)
            # through the activation
            return tf.keras.layers.LeakyReLU()(x)

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
        # build encoder
        x = Encoder_conv_block(x, n_filters=8)
        x = MaxPooling2D(pool_size=(2,2),strides=2)(x)
        x = Encoder_conv_block(x, n_filters=16)
        x = MaxPooling2D(pool_size=(2,2),strides=2)(x)
        x = Encoder_conv_block(x, n_filters=32)
        x = MaxPooling2D(pool_size=(2,2),strides=2)(x)

        # bottle-neck
        x = Conv2D(filters=64, kernel_size=kernel_size)(x)
        x = Conv2D(filters=128, kernel_size=kernel_size)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # encoding vector
        encoding_vector = GlobalMaxPooling2D()(x)

        # FCN
        pred = Dropout(rate=0.2)(encoding_vector)
        pred = Dense(units=60, activation='relu')(pred)
        pred = Dropout(rate=0.4)(pred)
        pred = Dense(units=self.num_classes)(pred)

        # sampling
        z_mean = Dense(self.vae_latent_dim, name='z_mean')(encoding_vector)
        z_log_var = Dense(self.vae_latent_dim, name='z_log_var')(encoding_vector)
        z = Sampling()([z_mean, z_log_var])

        # build decoder
        aus_dim = [int(self.input_size[0] / 2**3), int(self.input_size[1] / 2**3)]

        # x = Dense(aus_dim[0] * aus_dim[0] * self.vae_latent_dim, activation='relu')(z)
        # x = tf.keras.layers.Reshape((aus_dim[0],aus_dim[0],128))(x)
        # x = tf.keras.layers.Conv2DTranspose(128, 5, activation='relu', strides=2, padding='same')(x)
        # x = tf.keras.layers.Conv2DTranspose(64, 5, activation='relu', strides=2, padding='same')(x)
        # x = tf.keras.layers.Conv2DTranspose(32, 5, activation='relu', strides=2, padding='same')(x)
        #
        # decoder_outputs = tf.keras.layers.Conv2DTranspose(1,5,activation='tanh', padding='same')(x)

        def resize_convolution(x, final_shape, filters):
            upsample = tf.image.resize(images=x,
                                            size=final_shape,
                                            method=tf.image.ResizeMethod.BILINEAR)
            up_conv = Conv2D(filters=filters,
                            kernel_size=3,
                            padding='same',
                            activation='relu')(upsample)
            return up_conv

        x = Dense(aus_dim[0] * aus_dim[0] * self.vae_latent_dim)(z)
        # x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Reshape((aus_dim[0],aus_dim[0],self.vae_latent_dim))(x)
        x = resize_convolution(x, final_shape=(aus_dim[0]*2, aus_dim[0]*2), filters=128)
        x = resize_convolution(x, final_shape=(aus_dim[0]*4, aus_dim[0]*4), filters=64)
        x = resize_convolution(x, final_shape=(aus_dim[0]*8, aus_dim[0]*8), filters=32)
        decoder_outputs = tf.keras.layers.Conv2DTranspose(self.number_of_input_channels, 3, activation='tanh', padding='same')(x)

        self.model = Model(inputs=inputs, outputs=[pred, decoder_outputs, augmented_norm, z_mean, z_log_var, z], name=model_name)

        # save model paramenters
        self.num_filter_start = 16
        self.depth = 3
        self.num_filter_per_layer = [16, 32, 64]
        self.custom_model = False

## VAE5 with M4-like encoder and decoder
class VAE5(object):

    def __init__(self, number_of_input_channels,
                    num_classes,
                    input_size,
                    data_augmentation=True,
                    class_weights=None,
                    kernel_size=(5,5),
                    model_name='VAE5',
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

        x = utilities_models_tf.augmentor(inputs)

        # clip [-1,1] since using tanh as output of the decoder
        augmented_norm = tf.clip_by_value(x, clip_value_min=-1, clip_value_max=1)

        # build encoder with Inception-like bloks
        def Encoder_conv_block(inputs, n_filters):
            '''
            Takes the input and processes it through three a 3x3 kernel with
            dilation of 0, 1 and 2 (simulating a 3x3, 5x5 and 7x7 convolution).
            The result is then concatenated along with the initial input,
            convolved through a 1x1 convolution and passed through an activation function.
            '''
            y = Conv2D(filters=n_filters,kernel_size=(1,1),padding='same', dilation_rate=1)(inputs)
            # perform conv with different kernel sizes
            conv3 = Conv2D(filters=n_filters,kernel_size=(3,3),padding='same', dilation_rate=1)(inputs)
            conv5 = Conv2D(filters=n_filters,kernel_size=(3,3),padding='same', dilation_rate=2)(inputs)
            conv7 = Conv2D(filters=n_filters,kernel_size=(3,3),padding='same', dilation_rate=3)(inputs)
            # concatenate
            x = tf.concat([conv3, conv5, conv7, y], axis=-1)
            # conbine the information af all filters together
            x = Conv2D(filters=n_filters,kernel_size=(1,1),padding='same')(x)
            # normalization
            # x = tfa.layers.GroupNormalization(groups=int(n_filters/4))(x)
            x = BatchNormalization()(x)
            # through the activation
            return tf.keras.layers.LeakyReLU()(x)

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
        # build encoder
        x = Encoder_conv_block(x, n_filters=8)
        x = MaxPooling2D(pool_size=(2,2),strides=2)(x)
        x = Encoder_conv_block(x, n_filters=16)
        x = MaxPooling2D(pool_size=(2,2),strides=2)(x)
        x = Encoder_conv_block(x, n_filters=32)
        x = MaxPooling2D(pool_size=(2,2),strides=2)(x)

        # bottle-neck
        x = Conv2D(filters=64, kernel_size=kernel_size)(x)
        x = Conv2D(filters=128, kernel_size=kernel_size)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # encoding vector
        encoding_vector = GlobalMaxPooling2D()(x)

        # FCN
        pred = Dropout(rate=0.2)(encoding_vector)
        pred = Dense(units=60, activation='relu')(pred)
        pred = Dropout(rate=0.4)(pred)
        pred = Dense(units=self.num_classes)(pred)

        # sampling
        z_mean = Dense(self.vae_latent_dim, name='z_mean')(encoding_vector)
        z_log_var = Dense(self.vae_latent_dim, name='z_log_var')(encoding_vector)
        z = Sampling()([z_mean, z_log_var])

        # build decoder
        aus_dim = [int(self.input_size[0] / 2**3), int(self.input_size[1] / 2**3)]

        # x = Dense(aus_dim[0] * aus_dim[0] * self.vae_latent_dim, activation='relu')(z)
        # x = tf.keras.layers.Reshape((aus_dim[0],aus_dim[0],128))(x)
        # x = tf.keras.layers.Conv2DTranspose(128, 5, activation='relu', strides=2, padding='same')(x)
        # x = tf.keras.layers.Conv2DTranspose(64, 5, activation='relu', strides=2, padding='same')(x)
        # x = tf.keras.layers.Conv2DTranspose(32, 5, activation='relu', strides=2, padding='same')(x)
        #
        # decoder_outputs = tf.keras.layers.Conv2DTranspose(1,5,activation='tanh', padding='same')(x)

        # build encoder with Inception-like bloks
        def Decoder_conv_block(inputs, n_filters, final_shape):
            '''
            Takes the input and processes it through three a 3x3 kernel with
            dilation of 0, 1 and 2 (simulating a 3x3, 5x5 and 7x7 convolution).
            The result is then concatenated along with the initial input,
            convolved through a 1x1 convolution and passed through an activation function.
            '''

            # Utility function to perform resized convolution
            def resize_convolution(x, final_shape, filters, kernel_size=(3,3), dilation_rate=1):
                upsample = tf.image.resize(images=x,
                                                size=final_shape,
                                                method=tf.image.ResizeMethod.BILINEAR)
                up_conv = Conv2D(filters=filters,
                                kernel_size=kernel_size,
                                dilation_rate=dilation_rate,
                                padding='same',
                                activation='relu')(upsample)
                return up_conv

            y = resize_convolution(inputs, final_shape, n_filters, kernel_size=(1,1), dilation_rate=1)
            # perform resize_conv with different kernel sizes
            conv3 = resize_convolution(inputs, final_shape, n_filters, kernel_size=(3,3), dilation_rate=1)
            conv5 = resize_convolution(inputs, final_shape, n_filters, kernel_size=(3,3), dilation_rate=2)
            conv7 = resize_convolution(inputs, final_shape, n_filters, kernel_size=(3,3), dilation_rate=3)

            # concatenate
            x = tf.concat([conv3, conv5, conv7, y], axis=-1)
            # conbine the information af all filters together
            x = Conv2D(filters=n_filters,kernel_size=(1,1),padding='same')(x)
            # normalization
            x = BatchNormalization()(x)
            # through the activation
            return tf.keras.layers.LeakyReLU()(x)

        # reshape encoded vector and make ready for decoder step
        x = Dense(aus_dim[0] * aus_dim[0] * self.vae_latent_dim)(z)
        # x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Reshape((aus_dim[0],aus_dim[0],self.vae_latent_dim))(x)

        # decode
        x = resize_convolution(x, n_filters=128, final_shape=(aus_dim[0]*2, aus_dim[0]*2))
        x = resize_convolution(x, n_filters=64, final_shape=(aus_dim[0]*4, aus_dim[0]*4))
        x = resize_convolution(x, n_filters=32, final_shape=(aus_dim[0]*8, aus_dim[0]*8))
        # bring decoded image to original input number of channels
        decoder_outputs = Conv2D(filters=self.number_of_input_channels, kernel_size=(3,3), padding="same", activation="tanh")(x)

        # finally create model
        self.model = Model(inputs=inputs, outputs=[pred, decoder_outputs, augmented_norm, z_mean, z_log_var, z], name=model_name)

        # save model paramenters
        self.num_filter_start = 16
        self.depth = 3
        self.num_filter_per_layer = [16, 32, 64]
        self.custom_model = False
