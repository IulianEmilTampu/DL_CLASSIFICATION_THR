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
                    input_size=(None, None, None),
                    data_augmentation=True,
                    normalizer=None,
                    class_weights=None,
                    kernel_size=(5,5,5),
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
        x = layers.Conv3D(filters=8, kernel_size=self.kernel_size, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.MaxPool3D()(x)
        x = layers.SpatialDropout3D(rate=0.2)(x)

        x = layers.Conv3D(filters=32, kernel_size=self.kernel_size, padding='same')(x)
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

## ViT model working
class ViT_3D(object):
    def __init__(self, num_image_in_sequence,
                num_classes,
                input_size,
                patch_size,
                projection_dim=64,
                num_heads=4,
                mlp_head_units=(2048,1024),
                transformer_layers=8,
                transformer_units=None,
                data_augmentation=True,
                class_weights=None,
                model_name='ViT',
                debug=False):

        self.num_image_in_sequence = num_image_in_sequence
        self.num_classes = num_classes
        self.input_size = input_size
        self.debug = debug
        if class_weights is None:
            self.class_weights = np.ones([1, self.num_classes])
        else:
            self.class_weights = class_weights
        self.model_name = model_name
        self.custom_model = True

        # ViT parameters
        self.patch_size = patch_size
        self.num_patches = (self.input_size[0] // self.patch_size) * (self.input_size[1] // self.patch_size)
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.mlp_head_units = mlp_head_units
        self.transformer_layers = transformer_layers
        self.depth = self.transformer_layers
        if not transformer_units:
            self.transformer_units = [selfprojection_dim * 2, self.projection_dim]
        else:
            self.transformer_units = transformer_units

        # DEFINE BUILDING BLOCKS
        # ################ MLP (the classifier)
        def mlp(x, hidden_units, dropout_rate):
            for units in hidden_units:
                x = layers.Dense(units, activation=tf.nn.gelu)(x)
                x = layers.Dropout(dropout_rate)(x)
            return x

        # ################# PATCH EXTRACTION
        class PatchesFromSparseVolume(layers.Layer):
            def __init__(self, patch_size):
                super(PatchesFromSparseVolume, self).__init__()
                self.patch_size = patch_size

            def call(self, sparse_volume):
                batch_size = tf.shape(sparse_volume)[0]
                sequense_length = sparse_volume.shape[-1]
                patches = tf.image.extract_patches(
                    images=sparse_volume,
                    sizes=[1, self.patch_size, self.patch_size, 1],
                    strides=[1, self.patch_size, self.patch_size, 1],
                    rates=[1, 1, 1, 1],
                    padding="VALID",
                ) # [batch, n_patch_row, n_patch_col, patch_dim**2*n_img_in_sequence]
                flattened_patch_dims = self.patch_size**2
                n_patches = patches.shape[1]*patches.shape[2]
                # bring to # [batch, n_patch_per_img, patch_dim**2, n_img_in_sequence]
                patches_reshaped =  tf.reshape(patches, [batch_size, n_patches, flattened_patch_dims,sequense_length])
                # bring to shape [batch, n_patch_per_img, n_img_in_sequence, flatten_dim]
                return tf.transpose(patches_reshaped, perm=(0,1,3,2))

        # ################  PATCH ENCODING LAYER
        class SparceVolumePatchEncoder(layers.Layer):
            def __init__(self, num_patches_per_img, img_sequence_length, projection_dim):
                super(SparceVolumePatchEncoder, self).__init__()
                self.num_patches_per_img = num_patches_per_img
                self.img_sequence_length = img_sequence_length
                self.projection_dim = projection_dim
                self.projection = layers.Dense(units=projection_dim)
                self.img_embedding = layers.Embedding(
                    input_dim=self.num_patches_per_img, output_dim=projection_dim
                )
                self.sequence_embedding = layers.Embedding(
                    input_dim=self.img_sequence_length, output_dim=projection_dim
                )

            def call(self, patches):
                batch_size = tf.shape(patches)[0]
                '''
                Expecting input to have shape
                [batch, n_patches_per_img, n_img_sequense, project_dim]
                '''
                in_img_positions = tf.range(start=0, limit=self.num_patches_per_img, delta=1)
                in_sequence_positions = tf.range(start=0, limit=self.img_sequence_length, delta=1)

                '''
                embed image in sequence position
                [batch, n_patches_per_img, n_img_sequense, project_dim] + [n_img_sequense, project_dim]
                '''
                encoded = self.projection(patches) + self.sequence_embedding(in_sequence_positions)
                '''
                embed in image position
                [batch, n_img_sequense, n_patches_per_img, project_dim] + [n_img_sequense, project_dim]
                '''
                encoded = tf.transpose(encoded, perm=(0,2,1,3)) + self.img_embedding(in_img_positions)

                # flatten the sequence and return [batch, n_img_sequense * n_patches_per_img, project_dim]
                return tf.reshape(encoded, [batch_size, self.num_patches_per_img*self.img_sequence_length, self.projection_dim])


        # ACTUALLY BUILD THE MODEL
        inputs = layers.Input(shape=(self.input_size[0], self.input_size[1], self.num_image_in_sequence))

        # Augment data.
        if data_augmentation:
            augmented = utilities_models_tf.augmentor(inputs)
        else:
            augmented = inputs

        # Create patches.
        patches = PatchesFromSparseVolume(self.patch_size)(augmented)
        # Encode patches.
        encoded_patches = SparceVolumePatchEncoder(self.num_patches,
                            self.num_image_in_sequence,
                            self.projection_dim)(patches)

        # Create multiple layers of the Transformer block.
        for _ in range(self.transformer_layers):
            # Layer normalization 1.
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            # Create a multi-head attention layer.
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=self.projection_dim, dropout=0.1
            )(x1, x1)
            # Skip connection 1.
            x2 = layers.Add()([attention_output, encoded_patches])
            # Layer normalization 2.
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            # MLP.
            x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
            # Skip connection 2.
            encoded_patches = layers.Add()([x3, x2])

        # Create a [batch_size, projection_dim] tensor.
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.5)(representation)
        # Add MLP.
        features = mlp(representation, hidden_units=self.mlp_head_units, dropout_rate=0.5)
        # Classify outputs.
        logits = layers.Dense(self.num_classes, activation='softmax')(features)
        # Create the Keras model.
        self.model = Model(inputs=inputs, outputs=logits)

## M4 like model

class M4_3D(object):
    def __init__(self, num_classes,
                    num_channels=1,
                    input_size=(None, None, None),
                    data_augmentation=True,
                    normalizer=None,
                    class_weights=None,
                    kernel_size=(5,5,5),
                    model_name='M4_3D',
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

        # save augmented image to compute reconstruction
        if data_augmentation:
            x = utilities_models_tf.augmentor(inputs)
        else:
            x = inputs

        # build encoder with ResNet-like bloks
        def Encoder_conv_block(inputs, n_filters, kernel_size=3):
            '''
            Takes the input and processes it through three a 3x3 kernel with
            dilation of 0, 1 and 2 (simulating a 3x3, 5x5 and 7x7 convolution).
            The result is then concatenated along with the initial input,
            convolved through a 1x1 convolution and passed through an activation function.
            '''
            y = layers.Conv3D(filters=n_filters,kernel_size=(1,1,1),padding='same', dilation_rate=1)(inputs)
            # perform conv with different kernel sizes
            conv3 = layers.Conv3D(filters=n_filters,kernel_size=kernel_size,padding='same', dilation_rate=1)(inputs)
            conv5 = layers.Conv3D(filters=n_filters,kernel_size=kernel_size,padding='same', dilation_rate=2)(inputs)
            conv7 = layers.Conv3D(filters=n_filters,kernel_size=kernel_size,padding='same', dilation_rate=3)(inputs)

            # perform depth wise  separable convolution to mix the different channels
            x = tf.concat([y, conv3, conv5, conv7],axis=-1)
            x = layers.Conv3D(filters=n_filters, kernel_size=(1,1,1), padding='same')(x)

            # normalization
            x = BatchNormalization()(x)

            # through the activation
            return tf.keras.layers.LeakyReLU()(x)

        # build encoder
        x = Encoder_conv_block(x, n_filters=32, kernel_size=self.kernel_size)
        x = layers.MaxPool3D()(x)
        x = Encoder_conv_block(x, n_filters=64, kernel_size=self.kernel_size)
        x = layers.MaxPool3D()(x)
        x = Encoder_conv_block(x, n_filters=128, kernel_size=self.kernel_size)
        x = layers.MaxPool3D()(x)

        # bottle-neck
        x = layers.Conv3D(filters=256, kernel_size=self.kernel_size, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = layers.Conv3D(filters=256, kernel_size=self.kernel_size, padding='same')(x)
        # x = tfa.layers.GroupNormalization(groups=int(128/4))(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)

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



























