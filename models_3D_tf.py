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

## Define compression layer

'''
What we want is to integrate infromation from all the b-scans in the volume
and use it for the classification as normal using the conventional models.
'''
class Sparse_volume_compressor(layers.Layer):
    def __init__(self, num_channels, filters):
        super(Sparse_volume_compressor, self).__init__()
        self.num_channels = num_channels
        self.filters = filters

    def call(self, sparse_volume):
        '''
        Specify the way the different b-scans are grouped together
        '''
        compressed_volume = tf.keras.layers.Conv3D(filters=self.filters,
                        kernel_size=(3,3,self.num_channels),
                        padding='same')(tf.expand_dims(sparse_volume, axis=-1))
        return compressed_volume

## LightOCT_3D

class LightOCT_3D(object):
    def __init__(self, num_classes,
                    num_channels=1,
                    input_size=(None, None, None)
                    data_augmentation=True,
                    normalizer=None,
                    class_weights=None,
                    kernel_size=(5,5),
                    model_name='LightOCT_3D',
                    debug=False):

        self.num_channels = num_channels
        self.num_classes = num_classes
        self.input_size=input_size
        self.debug = debug
        if class_weights is None:
            self.class_weights = np.ones([1, self.num_classes])
        else:
            self.class_weights = class_weights
        self.model_name = model_name
        self.kernel_size = kernel_size

        # inputs = Input(shape=self.input_size)
        inputs = Input(shape=[self.input_size[0],
                        self.input_size[1],
                        self.input_size[2],
                        self.num_channels])

        # [Conv3d-BatchNorm-LeakyRelu-MaxPool]
        x = layers.Conv3D(filters=8, kernel_size=(5,5,5), padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.MaxPool3D()(x)
        x = layers.SpatialDropout3D(rate=0.2)(x)

        x = layers.Conv3D(filters=32, kernel_size=(5,5,5), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.MaxPool3D()(x)


        # compress the 3D volume into a 2D image
        x = layers.Conv3D(filters=1, kernel_size=(1,1,1))(x)

        x = layers.Conv2D(filters=128, kernel_size=(1,1))(tf.squeeze(x, axis=-1))


        # flatten and then classifier (Flatten makes the model too large. Using
        # Global average pooling instead - bonus of using None as input size)
        # x = layers.Flatten()(x)
        x = layers.GlobalAveragePooling2D()(x)

        # classifier
        x = layers.Dense(units=64)(x)
        x = layers.Dropout(rate=0.2)(x)
        x = layers.Dense(units=32)(x)
        x = layers.Dropout(rate=0.2)(x)
        final = Dense(units=self.num_classes, activation='softmax')(x)

        # finally make the model and return
        self.model = Model(inputs=inputs, outputs=final, name=model_name)

        # save model paramenters
        self.num_filter_start = 8
        self.depth = 2
        self.num_filter_per_layer = [8, 32, 128]
        self.custom_model = False

        # print model if needed
        if self.debug is True:
            print(self.model.summary())

