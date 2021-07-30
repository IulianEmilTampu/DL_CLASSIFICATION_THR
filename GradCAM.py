'''
Grad-CAM implementation [1] as described in post available at [2].

[1] Selvaraju RR, Cogswell M, Das A, Vedantam R, Parikh D, Batra D. Grad-cam:
    Visual explanations from deep networks via gradient-based localization.
    InProceedings of the IEEE international conference on computer vision 2017
    (pp. 618-626).

[2] https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/

'''

from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2

class gradCAM:
    def __init__(self, model, classIdx, layerName=None, use_image_prediction=True, debug=False):
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












