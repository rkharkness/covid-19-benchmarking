import albumentations as A

import sys 
sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab

import argparse
import cv2
import random
import sys
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

from tensorflow.keras.models import load_model
from scipy import interp

import os
try:
    import lime
except:
    sys.path.append(os.path.join('..', '..')) # add the current directory
    import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from matplotlib import pyplot as plt

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from mpl_toolkits.axes_grid1 import ImageGrid
from torch.nn import ReLU

import matplotlib.cm as cm


from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2

import seaborn as sns

from tqdm import tqdm

import sys
sys.path.append('../')


#tf.disable_eager_execution() 
#from tf_explain.core.grad_cam import GradCAM
# from tf_explain.core.gradients_inputs import GradientsInputs
#from tf_explain.core.activations import ExtractActivations
#from tf_explain.core.occlusion_sensitivity import OcclusionSensitivity
#from tf_explain.core.integrated_gradients import IntegratedGradients
#from tf_explain.core.smoothgrad import SmoothGrad
#from tf_explain.core.vanilla_gradients import VanillaGradients

# from models.res_attn import AttentionResNetModified
# from models.ecovnet import ECovNet
# from models.coronet_tfl import CoroNet
from scripts.deep_viz_keras.saliency import GradientSaliency
from scripts.deep_viz_keras.integrated_gradients import IntegratedGradients
from scripts.deep_viz_keras.guided_backprop import GuidedBackprop

# from dataloaders import make_generators

from scripts.models.ssl_am2 import SSL_AM
from scripts.dataloaders import make_generators

#import tensorflow.compat.v1 as tf1 

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import Model

from tensorflow.python.framework import ops 
import math
import copy

import torch
import torch.nn as nn

from functools import partial
from PIL import Image

import torch
import torch.nn.functional as F

from PIL import Image

import os
import json
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import torchvision
from torchvision import models
from torchvision import transforms

from captum.attr import Saliency
#from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

def get_custom_bce(epsilon = 1e-7):
  def custom_bce(y_true, y_pred):
    return -tf.math.reduce_mean(y_true * tf.math.log(tf.math.maximum(y_pred, tf.constant(epsilon))) + (1. - y_true) * tf.math.log(tf.math.maximum(1. - y_pred, tf.constant(epsilon))))
  return custom_bce

def deprocess_image(x):
    """Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    """
    # normalize tensor: center on 0., ensure std is 0.25
    x = x.copy()
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def show_image(image, grayscale = True, ax=None, title=''):
    image = np.squeeze(image)
    if ax is None:
        plt.figure()
    plt.axis('off')
    if len(image.shape) == 2 or grayscale == True:
        if len(image.shape) == 3:
            image = np.sum(np.abs(image), axis=2)
            
        vmax = np.percentile(image, 99)
        vmin = np.min(image)
        image = deprocess_image(image)
        plt.imshow(image, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
       # plt.imshow(mask)
        plt.title(title)
    else:
        image = deprocess_image(image)
        image = image + 127.5
        image = image.astype('uint8')      
        plt.imshow(image)
        #plt.imshow(mask)
        plt.title(title)

# Keras Gradient*Input methods
class KerasGradient():
    def __init__(self, data, model, test, k, weights, threshold_data=None, type=['vanilla', 'integrated', 'backprop']):
        super(KerasGradient, self).__init__()

        self.model = model.build_model()
        self.model = self.load_weights(self.model, k, weights)
        self.network = self.model['model']
        self.model['loss_fn'] = get_custom_bce()
        self.k = k
        self.test = test
        self.data = data

        if threshold_data != None:
            with open(threshold_data) as f:
                thresholds = json.load(f)
                thresholds = thresholds['All']
                self.best_threshold = thresholds['best_threshold'][k-1]
                print(f'loading threshold: {self.best_threshold}')
         #else:
          #      self.best_threshold = 0.5
        else:
            self.best_threshold = None

        self.type = type
        self.mapping = ['Negative', 'Positive']

        for i in self.type:
            assert i in ['integrated', 'vanilla', 'backprop']


    def load_weights(self, model, k, weights):
        print(f"loading weights: {weights} for {model['model_name']}...")
        
        def convert_path(weights, k, suffix):
            weights_path = f"{weights}_{k}{suffix}"
            return weights_path

        if model['model_type'] == 'keras':
            suffix = '.h5'
            weights = convert_path(weights, k, suffix)
            print(f"loading weights: {weights} for {model['model_name']}...")
            model['model'].load_weights(weights)
        #    model['model'].built = True
        #    model['model'].trainable = False
        
        elif model['model_type'] == 'pytorch':
            if model['model_name'] == 'mag_sd':
                model['model'].resume_model_from_path(weights, k)
                model['model'].set_eval()
            else:
                suffix = '.pth'
                weights = convert_path(weights, k, suffix)
                print(f"loading weights: {weights} for {model['model_name']}...")
                model['model'].load_state_dict(torch.load(weights,map_location=torch.device('cpu')))
                model['model'].eval()
        
        return model


    def vanilla(self):
        img, label = self.data
        img = tf.expand_dims(img, 0)
        predictions = self.model['model'](img)

        if predictions.shape[0]>1:
            binary = [1*(p.numpy() >= self.best_threshold) for p in predictions]
        else:
            binary = 1*(predictions.numpy() >= self.best_threshold)

        pred_label = self.mapping[binary[0][0]]

        vanilla = GradientSaliency(self.model)
        mask = vanilla.get_mask(img, label) #binary[0][0])
        img = img.numpy()

        show_image(img, ax=plt.subplot(1,3,1), title='cxr')  
        show_image(mask[0], ax=plt.subplot(1,3,2), title='vanilla gradient')  

        mask = vanilla.get_smoothed_mask(img, label) #binary[0][0])
        show_image(mask[0], ax=plt.subplot(1,3,3), title='smoothed vanilla gradient')
        plt.savefig(f"/MULTIX/DATA/HOME/covid-19-benchmarking/results/{self.model['model_name']}/{self.test}/vanilla_{self.model['model_name']}_gt_{label}_{pred_label}_{predictions.numpy()}.png")            

    def integrated(self):
        img, label = self.data
        img = tf.expand_dims(img, 0)
        predictions = self.model['model'](img)

        if predictions.shape[0]>1:
            binary = [1*(p.numpy() >= self.best_threshold) for p in predictions]
        else:
            binary = 1*(predictions.numpy() >= self.best_threshold)

        pred_label = self.mapping[binary[0][0]]
        inter_grad = IntegratedGradients(self.model)

        show_image(img, ax=plt.subplot(1,3,1), title='cxr')

        mask = inter_grad.get_mask(img, label)
        show_image(mask[0], ax=plt.subplot(1,3,2), title='integrated grad')

        mask = inter_grad.get_smoothed_mask(img, label)
        show_image(mask[0], ax=plt.subplot(1,3,3), title='smoothed integrated grad')

        plt.savefig(f"/MULTIX/DATA/HOME/covid-19-benchmarking/results/{self.model['model_name']}/{self.test}/inter_grad_{self.model['model_name']}_gt_{label}_{pred_label}_{predictions.numpy()}.png")            

    def backprop(self):
        img, label = self.data
        img = tf.expand_dims(img, 0)
        predictions = self.model['model'](img)
        if predictions.shape[0]>1:
            binary = [1*(p.numpy() >= self.best_threshold) for p in predictions]
        else:
            binary = 1*(predictions.numpy() >= self.best_threshold)

        pred_label = self.mapping[binary[0][0]]

        guided_bprop = GuidedBackprop(self.model)

        show_image(img, ax=plt.subplot(1,3,1), title='cxr')

        mask = guided_bprop.get_mask(img, label)
        show_image(mask[0], ax=plt.subplot(1,3,2), title='guided backprop')

        mask = guided_bprop.get_smoothed_mask(img, label)
        show_image(mask[0], ax=plt.subplot(1,3,3), title='smoothed guided backprop')
        plt.savefig(f"/MULTIX/DATA/nccid/guided_bp_{self.model['model_name']}_'gt'{label}_{pred_label}.png")
        plt.show()

    def gradient_visuals(self):
            if 'vanilla' in self.type:
                self.vanilla()
            if 'integrated' in self.type:
                self.integrated()
            if 'backprop' in self.type:
                self.backprop()


class KerasExtras():
    def __init__(self, data, model, threshold=0.5):
        super(KerasExtras, self).__init__()
        self.data = data
        self.model = model
        self.threshold = threshold  
        self.mapping = ['Negative', 'Positive']

    def lime(self):
        img, label = self.data
        predictions = self.model['model'](img)

        if predictions.shape[0]>1:
            binary = [1*(p.numpy() >= self.threshold) for p in predictions]
        else:
            binary = 1*(predictions.numpy() >= self.threshold)
        plt.figure()
        pred_label = self.mapping[binary[0][0]]
        label = self.mapping[label[0][0]]
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(np.array(img[0]).astype('double'), self.model['model'], top_labels=1, hide_color=0, num_samples=100)
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=True)
        plt.imshow(mark_boundaries(temp, mask))

        plt.savefig(f"/MULTIX/DATA/nccid/lime_{self.model['model_name']}_'gt'{label}_{pred_label}.png")
        plt.show()
    
    def occlusion(self, occluding_size=100, occluding_stride=5, occluding_pixel=0):
        img, label = self.data
        out = self.model['model'](img)
        out = out[0]
        if out.shape[0]>1:
            binary = [1*(p.numpy() >= self.threshold) for p in out]
           # softmax_like = [[p, 1-p] for p in out]
        else:
            binary = 1*(out.numpy() >= self.threshold)
            #softmax_like = [out, 1-out] 
            
        # Getting the index of the winning class:
        print(binary[0])
        pred_label = self.mapping[binary[0]]

        #index_object = [i for i, j in enumerate(out) if j == m]
        img = img[0]
        height, width, _ = img.shape
        output_height = int(math.ceil((height-occluding_size) / occluding_stride + 1))
        output_width = int(math.ceil((width-occluding_size) / occluding_stride + 1))
        heatmap = np.zeros((output_height, output_width))
        
        for h in range(output_height):
            for w in range(output_width):
                # Occluder region:
                h_start = h * occluding_stride
                w_start = w * occluding_stride
                h_end = min(height, h_start + occluding_size)
                w_end = min(width, w_start + occluding_size)

                # Getting the image copy, applying the occluding window and classifying it again:
                input_image = copy.copy(img.numpy())
                input_image[h_start:h_end, w_start:w_end,:] =  occluding_pixel            

                im = input_image #.transpose((2,0,1))
                im = np.expand_dims(im, axis=0)
                out = self.model['model'](im)
                out = out[0]
                print('scanning position (%s, %s)'%(h,w))
                # It's possible to evaluate the VGG-16 sensitivity to a specific object.
                # To do so, you have to change the variable "index_object" by the index of
                # the class of interest. The VGG-16 output indices can be found here:
                # https://github.com/HoldenCaulfieldRye/caffe/blob/master/data/ilsvrc12/synset_words.txt
                prob = (out[0]) 
                heatmap[h,w] = prob

        f = pylab.figure()
        f.add_subplot(1, 2, 1)  # this line outputs images side-by-side    
        ax = sns.heatmap(heatmap,xticklabels=False, yticklabels=False)
        f.add_subplot(1, 2, 2)
        plt.imshow(img)
        plt.savefig(f"/MULTIX/DATA/nccid/occlusion_{self.model['model_name']}_'gt'{label}_{pred_label}.png")

        plt.show()
        return heatmap

    def all(self):
        _ = self.lime()
     #   _ = self.occlusion()
    

@tf.custom_gradient
def guidedRelu(x):
    def grad(dy):
        return tf.cast(dy>0,"float32") * tf.cast(x>0, "float32") * dy
    return tf.nn.relu(x), grad

# Keras GradCAM methods
class KerasGradCAM():
    def __init__(self, data, model, threshold=0.5, layer_name=None):
        super(KerasGradCAM, self).__init__()
        self.data = data
        self.model = model
        self.threshold = threshold
        self.mapping = ['Negative', 'Positive']
        self.layer_name = layer_name
        if self.layer_name == None:
            self.layer_name = self.find_target_layer()

        self.guided_model = self.build_guided_model()


    def find_target_layer(self):
        for layer in reversed(self.model['model'].layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM")

    def compute_heatmap(self, image, upsample_size=(480,480), eps=1e-5):
        image, label = self.data
        gradModel = Model(
            inputs=[self.model['model'].inputs],
            outputs=[self.model['model'].get_layer(self.layer_name).output, self.model['model'].output]
        )
        # record operations for automatic differentiation

        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convOuts, preds) = gradModel(inputs)  # preds after softmax
            loss = preds[0]

        # compute gradients with automatic differentiation
        grads = tape.gradient(loss, convOuts)

        # discard batch
        convOuts = convOuts[0]
        grads = grads[0]
        norm_grads = tf.divide(grads, tf.reduce_mean(tf.square(grads)) + tf.constant(eps))
        # compute weights
        weights = tf.reduce_mean(norm_grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOuts), axis=-1)

        cam = np.array(cam)
        numer = cam - np.min(cam)
        denom = (cam.max() - cam.min()) + eps
        cam = numer / denom
        # Apply reLU
        cam = np.maximum(cam, 0)

        cam = cam / np.max(cam)
        cam = cv2.resize(cam, upsample_size,cv2.INTER_LINEAR)

        # convert to 3D
        cam3 = np.expand_dims(cam, axis=2)
        cam3 = np.tile(cam3, [1, 1, 3])

        return cam3

    
    def gradcam_plus(self):
        eps = 1e-5
        img_tensor, label = self.data
        conv_layer = self.model['model'].get_layer(self.layer_name)
        heatmap_model = Model([self.model['model'].inputs], [conv_layer.output, self.model['model'].output])

        with tf.GradientTape() as gtape1:
            with tf.GradientTape() as gtape2:
                with tf.GradientTape() as gtape3:
                    conv_output, predictions = heatmap_model(img_tensor)
                    category_id = 1*(predictions[0].numpy()>=self.threshold)
                    if self.mapping:
                        print(self.mapping[category_id[0]])
                    output = predictions[:, 0]
                    conv_first_grad = gtape3.gradient(output, conv_output)
                conv_second_grad = gtape2.gradient(conv_first_grad, conv_output)
            conv_third_grad = gtape1.gradient(conv_second_grad, conv_output)
        
        global_sum = np.sum(conv_output, axis=(0, 1, 2))

        alpha_num = conv_second_grad[0]
        alpha_denom = conv_second_grad[0]*2.0 + conv_third_grad[0]*global_sum
        alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, 1e-10)
        
        alphas = alpha_num/alpha_denom
        alpha_normalization_constant = np.sum(alphas, axis=(0,1))
        alphas /= alpha_normalization_constant

        weights = np.maximum(conv_first_grad[0], 0.0)

        deep_linearization_weights = np.sum(weights*alphas, axis=(0,1))
        cam = np.sum(deep_linearization_weights*conv_output[0], axis=2)

        cam = np.array(cam)
        numer = cam - np.min(cam)
        denom = (cam.max() - cam.min()) + eps
        cam = numer / denom
        # Apply reLU
        cam = np.maximum(cam, 0)

        cam = cam / np.max(cam)
        cam = cv2.resize(cam, (480,480),cv2.INTER_LINEAR)

        # convert to 3D
        cam3 = np.expand_dims(cam, axis=2)
        cam3 = np.tile(cam3, [1, 1, 3])

        return cam3


    def overlay_gradCAM(self,img, cam3):
        cam3 = np.uint8(255 * cam3)
        img = 255.0 * img

        cam3 = cv2.applyColorMap(cam3, cv2.COLORMAP_JET)

        new_img = 0.3 * cam3 + 0.8 * np.array(img)

        return (new_img * 255.0 / new_img.max()).astype("uint8")

    def build_guided_model(self):
        gbModel = Model(
            inputs = [self.model['model'].inputs],
            outputs = [self.model['model'].get_layer(self.layer_name).output]
        )
        layer_dict = [layer for layer in gbModel.layers[1:] if hasattr(layer,"activation")]
        for layer in layer_dict:
            if layer.activation == tf.keras.activations.relu:
                layer.activation = guidedRelu
        
        return gbModel

    def guided_backprop(self, images, upsample_size):
        """Guided Backpropagation method for visualizing input saliency."""
        with tf.GradientTape() as tape:
            inputs = tf.cast(images, tf.float32)
            tape.watch(inputs)
            outputs = self.guided_model(inputs)

        grads = tape.gradient(outputs, inputs)[0]

        saliency = cv2.resize(np.asarray(grads), upsample_size)

        return saliency

    def showCAMs(self, upsample_size=(480,480)):
        img, label = self.data
        cam3 = self.compute_heatmap(image=img, upsample_size=upsample_size)
        gradcam = self.overlay_gradCAM(img, cam3)
        gradcam = cv2.cvtColor(gradcam[0], cv2.COLOR_BGR2RGB)
        # Guided backprop
        gb = self.guided_backprop(img, upsample_size)
        gb_im = deprocess_image(gb)
        gb_im = cv2.cvtColor(gb_im, cv2.COLOR_BGR2RGB)
        # Guided GradCAM
        guided_gradcam = deprocess_image(gb*cam3)
        guided_gradcam = cv2.cvtColor(guided_gradcam, cv2.COLOR_BGR2RGB)

        cam_plus = self.gradcam_plus()
        gradcam_plus = self.overlay_gradCAM(img, cam_plus)
        gradcam_plus = cv2.cvtColor(gradcam_plus[0], cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(15, 10))
        plt.subplot(141)
        plt.title('GradCAM')
        plt.axis('off')
        plt.imshow(gradcam)
        # plt.imshow(gradcam, cmap='jet', alpha=0.5)

        plt.subplot(142)
        plt.title('Guided Backprop')
        plt.axis('off')
        plt.imshow(gb_im)
        
        plt.subplot(143)
        plt.title('Guided GradCAM')
        plt.axis('off')
        plt.imshow(guided_gradcam)
        plt.show()

        plt.subplot(144)
        plt.title('GradCAM++')
        plt.axis('off')
        plt.imshow(gradcam_plus)

        plt.savefig(f"/MULTIX/DATA/nccid/{self.model['model_name']}_gc.png")
        plt.show()


    # def find_target_layer(self):
    #     for layer in reversed(self.model['model'].layers):
    #         if len(layer.output_shape) == 4:
    #             return layer.name
    #     raise ValueError("Could not find 4D layer. Cannot apply GradCAM")
    
    # def deprocess_image(self, x):
    #     x = np.array(x).copy()
    #     if np.ndim(x) > 3:
    #         x = np.squeeze(x)
    #     x -= x.mean()
    #     x /= (x.std() + 1e-5)
    #     x *= 0.1
    #     x += 0.5
    #     x = np.clip(x, 0, 1)
    #     x *= 255 #to convert into RGB
    #     x = np.clip(x, 0, 255).astype('uint8')
    #     return x

    # def grad_cam(self):


    # def gradcam(self, layer_name):
    #     # First, we create a model that maps the input image to the activations
    #     # of the last conv layer as well as the output predictions
    #     img, label = self.data
    #     grad_model = tf.keras.models.Model(
    #         [self.model['model'].inputs], [self.model['model'].get_layer(layer_name).output, self.model['model'].output]
    #     )

    #     # Then, we compute the gradient of the top predicted class for our input image
    #     # with respect to the activations of the last conv layer
    #     with tf.GradientTape() as tape:
    #         last_conv_layer_output, preds = grad_model(img)
    #         #if pred_index is None:
    #         category_id = 1*(preds[0].numpy()>=self.threshold)
    #         class_channel = preds[:, 0]
    #         #class_channel = preds[0] 
    #         # This is the gradient of the output neuron (top predicted or chosen)
    #         # with regard to the output feature map of the last conv layer
    #         grads = tape.gradient(class_channel, last_conv_layer_output)

    #         # This is a vector where each entry is the mean intensity of the gradient
    #         # over a specific feature map channel
    #         pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    #         # We multiply each channel in the feature map array
    #         # by "how important this channel is" with regard to the top predicted class
    #         # then sum all the channels to obtain the heatmap class activation
    #         last_conv_layer_output = last_conv_layer_output[0]
    #         heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    #        # heatmap = tf.squeeze(heatmap)
    #         #heatmap = tf.expand_dims(heatmap, -1)
    #         # For visualization purpose, we will also normalize the heatmap between 0 & 1
    #         heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    #         heatmap = cv2.resize(np.array(heatmap), (480,480), cv2.INTER_LINEAR)
    #         heatmap = np.expand_dims(heatmap, -1)

    #         numer = heatmap - np.min(heatmap)
    #         eps = 1e-10
    #         denom = (heatmap.max() - heatmap.min()) + eps
    #         heatmap = numer / denom
    #         heatmap = (heatmap * 255).astype("uint8")
            
    #         colormap = cv2.COLORMAP_JET

    #         heatmap = cv2.applyColorMap(heatmap, colormap)
    #         heatmap = np.expand_dims(heatmap, 0)

    #         alpha = 0.5
    #         print(img.shape, heatmap.shape)
    #         output = cv2.addWeighted(np.array(img, dtype=float), alpha, np.array(heatmap, dtype=float), 1 - alpha, 0)

    #         return output
    #     # weights = np.mean(grads_val, axis=(0, 1))
    #     # cam = np.dot(output, weights)

    #     # cam = cv2.resize(cam, (480,480), cv2.INTER_LINEAR)
    #     # cam = np.maximum(cam, 0)
    #     # cam = cam / cam.max()
    #     # return cam

    # def build_guided_model(self):
    #     print('building guided_model')
    #     try:
    #         @tf.RegisterGradient('GuidedRelu')
    #         def _guided_backprop(op, grad):
    #             dtype = op.outputs[0].dtype
    #             gate_g = tf.cast(grad > 0., dtype)
    #             gate_y = tf.cast(op.outputs[0] > 0, dtype)
    #             return gate_y * gate_g * grad
    #     except KeyError: #KeyError is raised if 'GuidedRelu' has already been registered as a gradient
    #         pass
    #     from tensorflow.python.keras.activations import linear

    #     cfg = self.model['model'].get_config()
    #     g = tf1.get_default_graph()
    #     # Compiling the model within this loop implements Guided Backprop
    #     with g.gradient_override_map({'Relu': 'GuidedRelu'}):
    #         # Copying model using it's config
    #         guid_model = Model.from_config(cfg)
    #         return guid_model

    # #Guided Backpropagation method
    # def guided_backprop(self):
    #    # for layer in self.guided_model.layers:
    #     #    print(layer) 
    #     print(self.guided_model)
    #     img, label = self.data
    #     grad_model = tf.keras.models.Model(
    #         [self.guided_model.inputs], [self.guided_model.get_layer(self.layer_name).output]
    #     )
    #     with tf.GradientTape() as tape:
    #         inputs = tf.cast(img, tf.float32)
    #         tape.watch(inputs)
    #         outputs = grad_model(inputs)

    #     grads = tape.gradient(outputs, inputs)[0]

    #     saliency = cv2.resize(np.asarray(grads), (480,480))
    #     return saliency
    
    # def gradcam_plus(self):
    #     eps = 1e-10
    #     img_tensor, label = self.data
    #     conv_layer = self.model['model'].get_layer(self.layer_name)
    #     heatmap_model = Model([self.model['model'].inputs], [conv_layer.output, self.model['model'].output])

    #     with tf.GradientTape() as gtape1:
    #         with tf.GradientTape() as gtape2:
    #             with tf.GradientTape() as gtape3:
    #                 conv_output, predictions = heatmap_model(img_tensor)
    #                 category_id = 1*(predictions[0].numpy()>=self.threshold)
    #                 if self.mapping:
    #                     print(self.mapping[category_id[0]])
    #                 output = predictions[:, 0]
    #                 conv_first_grad = gtape3.gradient(output, conv_output)
    #             conv_second_grad = gtape2.gradient(conv_first_grad, conv_output)
    #         conv_third_grad = gtape1.gradient(conv_second_grad, conv_output)
        
    #     global_sum = np.sum(conv_output, axis=(0, 1, 2))

    #     alpha_num = conv_second_grad[0]
    #     alpha_denom = conv_second_grad[0]*2.0 + conv_third_grad[0]*global_sum
    #     alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, 1e-10)
        
    #     alphas = alpha_num/alpha_denom
    #     alpha_normalization_constant = np.sum(alphas, axis=(0,1))
    #     alphas /= alpha_normalization_constant

    #     weights = np.maximum(conv_first_grad[0], 0.0)

    #     deep_linearization_weights = np.sum(weights*alphas, axis=(0,1))
    #     grad_CAM_map = np.sum(deep_linearization_weights*conv_output[0], axis=2)

    #     # heatmap = np.maximum(grad_CAM_map, 0)
    #     # max_heat = np.max(heatmap)
    #     # if max_heat == 0:
    #     #     max_heat = 1e-10
    #     # heatmap /= max_heat

    #     heatmap = cv2.resize(np.array(grad_CAM_map), (480,480), cv2.INTER_LINEAR)
    #     heatmap = np.expand_dims(heatmap, -1)

    #     numer = heatmap - np.min(heatmap)
    #     denom = (heatmap.max() - heatmap.min()) + eps
    #     heatmap = numer / denom
    #     heatmap = (heatmap * 255).astype("uint8")

    #     alpha = 0.5

    #     colormap = cv2.COLORMAP_JET
    #     heatmap = cv2.applyColorMap(heatmap, colormap)
    #     heatmap = np.expand_dims(heatmap, 0)

    #     output = cv2.addWeighted(np.array(img_tensor, dtype=float), alpha, np.array(heatmap, dtype=float), 1 - alpha, 0)
    #     # return the resulting heatmap to the calling function
    #     return output



    #     """ This funtions computers Guided Grad-CAM
    #     as well as visualises all 3 approaches"""
    # def guided_gradcam_plus_fn(self, cls=-1, visualize=True, save=True):
    #     img, label = self.data
    #     predictions = self.model['model'](img)

    #     if predictions.shape[0]>1:
    #         binary = [1*(p.numpy() >= self.threshold) for p in predictions]
    #         #softmax_like = [[p, 1-p] for p in predictions]

    #     else:
    #         binary = 1*(predictions.numpy() >= self.threshold)
    #        # softmax_like = [predictions, 1-predictions]

    #     binary = binary[0][0]
    #     pred_label = self.mapping[binary]
    #     print('Model prediction: '+ pred_label)
    #     print()
    #   #  print('Probabilities:')

    #  #   for i in range(2): # express single value bce pred as class-wise probabilities
    #    #     print('\t{}. {}\t{:.3f}'.format(i, self.mapping[i], softmax_like[0,i]))
    
    #     if cls == -1:
    #         cls = binary

    #     print()
    #     print("Explanation for '{}':".format(self.mapping[cls]))
    #     print()
    #     gradcam = self.gradcam(self.layer_name)
    #    # gradcam_up = cv2.resize(np.array(gradcam), (480,480), cv2.INTER_LINEAR)
    #     gb = self.guided_backprop()
    #     guided_gradcam = gb * gradcam #_up[..., np.newaxis]

    #     gradcam_plus = self.gradcam_plus()

    #     if save:
    #         jetcam = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
    #         jetcam = cv2.resize(jetcam, (480,480), cv2.INTER_LINEAR)

    #         jetcam = (np.float32(jetcam) + self.deprocess_image(img[0])) / 2
    #         cv2.imwrite('gradcam.jpg', np.uint8(jetcam))
    #         cv2.imwrite('guided_backprop.jpg', self.deprocess_image(gb))
    #         cv2.imwrite('guided_gradcam.jpg', self.deprocess_image(guided_gradcam))
        
        # if visualize:
        #     plt.figure(figsize=(15, 10))
        #     plt.subplot(141)
        #     plt.title('GradCAM')
        #     plt.axis('off')
        #    # plt.imshow(self.deprocess_image(gradcam))
        #     plt.imshow(gradcam[0], cmap='jet', alpha=0.5)

        #     plt.subplot(142)
        #     plt.title('Guided Backprop')
        #     plt.axis('off')
        #     plt.imshow(np.flip(self.deprocess_image(gb), -1))
            
        #     plt.subplot(143)
        #     plt.title('Guided GradCAM')
        #     plt.axis('off')
        #     plt.imshow(np.flip(self.deprocess_image(guided_gradcam), -1))
        #     plt.show()

        #     plt.subplot(144)
        #     plt.title('GradCAM++')
        #     plt.axis('off')
        #     plt.imshow(np.flip(self.deprocess_image(gradcam_plus[0]), -1))

        #     plt.savefig(f"/MULTIX/DATA/nccid/{self.model['model_name']}_gc.png")
        #     plt.show()

        # return gradcam, gb, guided_gradcam, gradcam_plus
    


class TorchVisuals():
    def __init__(self, data, model, threshold=0.5, layer_name=None, visualize=True):
        super(TorchVisuals, self).__init__()
        self.data = data
        self.model = model
        self.threshold = self.load_threshold(threshold_data)
        self.mapping = ['Negative', 'Positive']

        self.visualize = visualize

      #  self.model['model'].eval()

    def attribute_image_features(self, pred, algorithm, **kwargs):
        img, label = self.data
        self.model['model'].zero_grad()
        tensor_attributions = algorithm.attribute(img,
                                                target=pred,
                                                **kwargs
                                                )
        return tensor_attributions

    def captum_visuals(self):
        img, label = self.img
        output = self.model['model'](self.img)
        out = output[0]
        if out.shape[0] > 1:
            binary = [1*(p >= self.threshold) for p in out]
            softmax_like = [[p, 1-p] for p in out]
        else:
            binary = 1*(out >= self.threshold)
            softmax_like = [out, 1-out] 

        pred_label = self.mapping[binary[0][0]]

        saliency = Saliency(self.model['model'])
        grads = saliency.attribute(self.img, target=binary)
        grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))

        print('Predicted:', pred_label, '(', output.squeeze().item(), ')')

        integrated_gradients = IntegratedGradients(model)
        attributions_ig = integrated_gradients.attribute(input, target=binary, n_steps=200)
        default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                    [(0, '#ffffff'),
                                                    (0.25, '#000000'),
                                                    (1, '#000000')], N=256)

        ig_viz = viz.visualize_image_attr(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1,2,0)),
                                    np.transpose(self.img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                    method='heat_map',
                                    cmap=default_cmap,
                                    show_colorbar=True,
                                    sign='positive',
                                    outlier_perc=1)


        noise_tunnel = NoiseTunnel(integrated_gradients)

        attributions_ig_nt = noise_tunnel.attribute(input, nt_samples=10, nt_type='smoothgrad_sq', target=pred_label_idx)
        ig_nt_viz = viz.visualize_image_attr_multiple(np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1,2,0)),
                                            np.transpose(self.img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                            ["original_image", "heat_map"],
                                            ["all", "positive"],
                                            cmap=default_cmap,
                                            show_colorbar=True)
        torch.manual_seed(0)
        np.random.seed(0)

        gradient_shap = GradientShap(self.model['model'])

        # Defining baseline distribution of images
        rand_img_dist = torch.cat([input * 0, input * 1])

        attributions_gs = gradient_shap.attribute(input,
                                                n_samples=50,
                                                stdevs=0.0001,
                                                baselines=rand_img_dist,
                                                target=binary)
        gs_viz = viz.visualize_image_attr_multiple(np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1,2,0)),
                                            np.transpose(self.img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                            ["original_image", "heat_map"],
                                            ["all", "absolute_value"],
                                            cmap=default_cmap,
                                            show_colorbar=True)
        
        occlusion = Occlusion(self.model['model'])
        attributions_occ = occlusion.attribute(input,
                                            strides = (3, 8, 8),
                                            target=binary,
                                            sliding_window_shapes=(3,15, 15),
                                            baselines=0)

        occ_viz = viz.visualize_image_attr_multiple(np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1,2,0)),
                                            np.transpose(self.img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                            ["original_image", "heat_map"],
                                            ["all", "positive"],
                                            show_colorbar=True,
                                            outlier_perc=2,
                                            )

        if visualize:
            plt.figure(figsize=(15, 10))
            plt.subplot(141)
            plt.title('Original Image')
            plt.axis('off')
            plt.imshow(self.deprocess_image(self.img))
            plt.imshow(gradcam, cmap='jet', alpha=0.5)

            plt.figure(figsize=(15, 10))
            plt.subplot(141)
            plt.title('Integrated Gradients')
            plt.axis('off')
            plt.imshow(self.deprocess_image(ig_viz))
            plt.imshow(gradcam, cmap='jet', alpha=0.5)

            plt.subplot(142)
            plt.title('Noise Tunnel')
            plt.axis('off')
            plt.imshow(np.flip(self.deprocess_image(ig_nt_viz), -1))
            
            plt.subplot(143)
            plt.title('Gradient Shap')
            plt.axis('off')
            plt.imshow(np.flip(self.deprocess_image(gs_viz), -1))
            plt.show()

            plt.subplot(144)
            plt.title('Occlusion')
            plt.axis('off')
            plt.imshow(np.flip(self.deprocess_image(occ_viz), -1))

            plt.savefig(f"/MULTIX/DATA/nccid/{model['model']}_captum_gt_{label}_{pred_label}.png")

    def lime(self):
        img, label = self.data
        predictions = self.model['model'](img)

        if predictions.shape[0]>1:
            binary = [1*(p >= self.threshold) for p in predictions]
        else:
            binary = 1*(predictions >= self.threshold)

        pred_label = self.mapping[binary[0][0]]
        label = self.mapping[label[0][0]]
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(np.array(img[0]).astype('double'), model['model'], top_labels=1, hide_color=0, num_samples=100)
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=True)
        plt.imshow(mark_boundaries(temp, mask))
        plt.savefig(f"/MULTIX/DATA/nccid/lime_{self.model['model_name']}_'gt'{label}_{pred_label}.png")

    def all(self):
        self.lime()
        self.captum_visuals()

        

if __name__ == "__main__":
   ## args parse for data, threshold etc.
    parser = argparse.ArgumentParser(description='Model Training Script')
    parser.add_argument('--data_csv', default='/MULTIX/DATA/HOME/covid-19-benchmarking/data/nccid_preprocessed14_21.csv', type=str, help='Path to data file')
    parser.add_argument('--weights', default=None)
    parser.add_argument('--test', default=None, type=str, help='Choose on of: [ltht, nccid_test, nccid_val, nccid_leeds, chexpert, custom]')
    parser.add_argument('--threshold_data', type=str, help='Path to threshold data, generated by nccid_test i.e. root/model_name_performance_nccid_test_df.csv')
    parser.add_argument('--subpop_analysis', default=False, type=bool, help='Perform subpop analysis or not, populations dependent on selected test dataset')
    parser.add_argument('--prevalence_analysis', default=False, type=bool, help='Vary prevalence of COVID in test population')

    args = parser.parse_args()
    
    tf.config.run_functions_eagerly(True)

    df = pd.read_csv(args.data_csv)
    ltht = False


    if args.test == 'ltht':
        df['xray_status'] = df['FinalPCR']
        df['cxr_path'] = df['path']
        df['kfold_1'] = 'test'
        ltht = True

        if args.subpop_analysis == True:
            data_dict = create_subpopulations(df, args.test)
        else:
            data_dict = {'All': df}
            data_dict = {'All': data_dict}
            
    elif args.test == 'nccid_val':
            data_dict = {'All': df}
            data_dict = {'All': data_dict} 
        
    elif args.test == 'nccid_test':
        df = df[df['xray_status']!=np.nan]
        df = df.dropna(subset=['xray_status'])
        df = df[df['kfold_1'] == 'test']

        if args.subpop_analysis == True:
            pass #data_dict = create_subpopulations(df, args.test)
        else:
            data_dict = {'All': df}
            data_dict = {'All': data_dict}
    ## load data

   ##  init model
    

    params = {'batchsize':16, "num_workers":4, "k":1}
    model = SSL_AM_Seg(pretrained=True, supervised=True)
    df = data_dict['All']['All']
    _, _, test_loader = make_generators(model.build_model(), df, df, df, params)       
    
    data_batch = next(iter(test_loader))
    print(data_batch[1].shape)
    for data in zip(data_batch[0], data_batch[1]):
        feature_explainer = KerasGradient(data, model, test=args.test, weights=args.weights, k=1, threshold_data=args.threshold_data, type=['vanilla', 'integrated', 'backprop'])
        feature_explainer.gradient_visuals()

## for tf or torch
   # feature_explainer = TorchVisuals()
# captum https://github.com/probayes/Covid19Xray/blob/master/pycovid19xray/explainer.py
    # guidedBP = GuidedBackprop(model=model)
    # gb_cam = guidedBP.guided_backprop(np.expand_dims(img_1,axis=0),(480,480))
    # guided_gradcam = deprocess_image(gb_cam)
    # plt.imshow(guided_gradcam)
    # plt.savefig('/content/guided_gradcam.png')
