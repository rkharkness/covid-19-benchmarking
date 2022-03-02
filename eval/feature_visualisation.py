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


from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2

import seaborn as sns

from tqdm import tqdm

import sys
sys.path.append('../')
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


import tensorflow.compat.v1 as tf1 

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

def show_image(image, grayscale = True, ax=None, title=''):
    if ax is None:
        plt.figure()
    plt.axis('off')
    if len(image.shape) == 2 or grayscale == True:
        if len(image.shape) == 3:
            image = np.sum(np.abs(image), axis=2)
            
        vmax = np.percentile(image, 99)
        vmin = np.min(image)

        plt.imshow(image, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
        plt.title(title)
    else:
        image = image + 127.5
        image = image.astype('uint8')
        
        plt.imshow(image)
        plt.title(title)

# Keras Gradient*Input methods
class KerasGradient():
    def __init__(self, data, model, threshold=0.5, type=['vanilla', 'integrated', 'backprop']):
        super(KerasGradient, self).__init__()
        self.data = data
        self.model = model
        self.threshold = threshold
        self.type = type
        self.mapping = ['Negative', 'Positive']

        for i in self.type:
            assert i in ['integrated', 'vanilla', 'smooth']

    def vanilla(self):
        img, label = self.data
        predictions = self.model['model'](img)

        if predictions.shape[0]>1:
            binary = [1*(p.numpy() >= self.threshold) for p in predictions]
        else:
            binary = 1*(predictions.numpy() >= self.threshold)

        pred_label = self.mapping[binary[0][0]]

        vanilla = GradientSaliency(self.model)
        mask = vanilla.get_mask(img, np.expand_dims(label.numpy(), 0))
        show_image(mask[0], ax=plt.subplot(1,2,1), title='vanilla gradient')  

        mask = vanilla.get_smoothed_mask(img, np.expand_dims(label.numpy(), 0))
        show_image(mask[0], ax=plt.subplot(1,2,2), title='smoothed vanilla gradient')
        plt.savefig(f"/MULTIX/DATA/nccid/smooth_vanilla_{self.model['model_name']}_'gt'{label}_{pred_label}.png")            

    def integrated(self):
        img, label = self.data
        predictions = self.model['model'](img)

        if predictions.shape[0]>1:
            binary = [1*(p.numpy() >= self.threshold) for p in predictions]
        else:
            binary = 1*(predictions.numpy() >= self.threshold)

        pred_label = self.mapping[binary[0][0]]
        inter_grad = IntegratedGradients(self.model)
        mask = inter_grad.get_mask(img)
        show_image(mask, ax=plt.subplot('121'), title='integrated grad')

        mask = inter_grad.get_smoothed_mask(img)
        show_image(mask, ax=plt.subplot('122'), title='smoothed integrated grad')
        plt.savefig(f"/MULTIX/DATA/nccid/inter_grad_{self.model['model_name']}_'gt'{label}_{pred_label}.png")            

    def backprop(self):
        img, label = self.data
        predictions = self.model['model'](img)

        if predictions.shape[0]>1:
            binary = [1*(p.numpy() >= self.threshold) for p in predictions]
        else:
            binary = 1*(predictions.numpy() >= self.threshold)

        pred_label = self.mapping[binary[0][0]]

        guided_bprop = GuidedBackprop(self.model)
        mask = guided_bprop.get_mask(img)
        show_image(mask, ax=plt.subplot('121'), title='guided backprop')

        mask = guided_bprop.get_smoothed_mask(img)
        show_image(mask, ax=plt.subplot('122'), title='smoothed guided backprop')
        plt.savefig(f"/MULTIX/DATA/nccid/guided_bp_{self.model['model_name']}_'gt'{label}_{pred_label}.png")

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

        pred_label = self.mapping[binary]
        label = self.mapping[label]
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(np.array(img[0]).astype('double'), model['model'], top_labels=1, hide_color=0, num_samples=100)
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=True)
        plt.imshow(mark_boundaries(temp, mask))

        plt.savefig(f"/MULTIX/DATA/nccid/lime_{self.model['model_name']}_'gt'{label}_{pred_label}.png")
    
    def occlusion(self, occluding_size=100, occluding_stride=5, occluding_pixel=0):
        img, label = self.data
        out = model['model'](img)
        out = out[0]
        if out.shape[0]>1:
            binary = [1*(p.numpy() >= self.threshold) for p in out]
            softmax_like = [[p, 1-p] for p in out]
        else:
            binary = 1*(out.numpy() >= self.threshold)
            softmax_like = [out, 1-out] 
            
        # Getting the index of the winning class:
        m = max(softmax_like)
        pred_label = self.mapping[m]

        index_object = [i for i, j in enumerate(out) if j == m]
        height, width, _ = image.shape
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
                input_image = copy.copy(image)
                input_image[h_start:h_end, w_start:w_end,:] =  occluding_pixel            

                im = input_image.transpose((2,0,1))
                im = np.expand_dims(im, axis=0)
                out = model['model'](im)
                out = out[0]
                print('scanning position (%s, %s)'%(h,w))
                # It's possible to evaluate the VGG-16 sensitivity to a specific object.
                # To do so, you have to change the variable "index_object" by the index of
                # the class of interest. The VGG-16 output indices can be found here:
                # https://github.com/HoldenCaulfieldRye/caffe/blob/master/data/ilsvrc12/synset_words.txt
                prob = (out[index_object]) 
                heatmap[h,w] = prob

        f = pylab.figure()
        f.add_subplot(1, 2, 0)  # this line outputs images side-by-side    
        ax = sns.heatmap(heatmap,xticklabels=False, yticklabels=False)
        f.add_subplot(1, 2, 1)
        plt.imshow(image)
        plt.savefig(f"/MULTIX/DATA/nccid/occlusion_{self.model['model_name']}_'gt'{label}_{pred_label}.png")

        plt.show()
        print ('Object index is %s'%index_object) 
        return heatmap

    def all(self):
        _ = self.lime()
        _ = self.occlusion()
    


# Keras GradCAM methods
class KerasGradCAM():
    def __init__(self, data, model, threshold=0.5, layer_name=None):
        super(KerasGradCAM, self).__init__()
        self.img, self.label = data
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
    
    def deprocess_image(self, x):
        x = np.array(x).copy()
        if np.ndim(x) > 3:
            x = np.squeeze(x)
        x -= x.mean()
        x /= (x.std() + 1e-5)
        x *= 0.1
        x += 0.5
        x = np.clip(x, 0, 1)
        x *= 255 #to convert into RGB
        x = np.clip(x, 0, 255).astype('uint8')
        return x

    def gradcam(self, layer_name):
        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        grad_model = tf.keras.models.Model(
            [self.model['model'].inputs], [self.model['model'].get_layer(layer_name).output, self.model['model'].output]
        )

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(self.img)
            #if pred_index is None:
            category_id = 1*(preds[0].numpy()>=self.threshold)
            class_channel = preds[:, category_id[0]]
            #class_channel = preds[0] 
            # This is the gradient of the output neuron (top predicted or chosen)
            # with regard to the output feature map of the last conv layer
            grads = tape.gradient(class_channel, last_conv_layer_output)

            # This is a vector where each entry is the mean intensity of the gradient
            # over a specific feature map channel
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            # We multiply each channel in the feature map array
            # by "how important this channel is" with regard to the top predicted class
            # then sum all the channels to obtain the heatmap class activation
            last_conv_layer_output = last_conv_layer_output[0]
            heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)

            # For visualization purpose, we will also normalize the heatmap between 0 & 1
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            
            return heatmap
        # weights = np.mean(grads_val, axis=(0, 1))
        # cam = np.dot(output, weights)

        # cam = cv2.resize(cam, (480,480), cv2.INTER_LINEAR)
        # cam = np.maximum(cam, 0)
        # cam = cam / cam.max()
        # return cam

    def build_guided_model(self):
        try:
            @tf.RegisterGradient('GuidedRelu')
            def _guided_backprop(op, grad):
                dtype = op.outputs[0].dtype
                gate_g = tf.cast(grad > 0., dtype)
                gate_y = tf.cast(op.outputs[0] > 0, dtype)
                return gate_y * gate_g * grad
        except KeyError: #KeyError is raised if 'GuidedRelu' has already been registered as a gradient
            pass
        from tensorflow.python.keras.activations import linear

        cfg = self.model['model'].get_config()
        g = tf1.get_default_graph()
        # Compiling the model within this loop implements Guided Backprop
        with g.gradient_override_map({'Relu': 'GuidedRelu'}):
            # Copying model using it's config
            guid_model = Model.from_config(cfg)
            return guid_model

    #Guided Backpropagation method
    def guided_backprop(self):
       # for layer in self.guided_model.layers:
        #    print(layer) 
        grad_model = tf.keras.models.Model(
            [self.guided_model.inputs], [self.guided_model.get_layer(self.layer_name).output]
        )
        with tf.GradientTape() as tape:
            inputs = tf.cast(self.img, tf.float32)
            tape.watch(inputs)
            outputs = grad_model(inputs)

        grads = tape.gradient(outputs, inputs)[0]

        saliency = cv2.resize(np.asarray(grads), (480,480))
        return saliency
    
    def gradcam_plus(self):
        img_tensor = self.img
        conv_layer = self.model['model'].get_layer(self.layer_name)
        heatmap_model = Model([self.model['model'].inputs], [conv_layer.output, self.model['model'].output])

        with tf.GradientTape() as gtape1:
            with tf.GradientTape() as gtape2:
                with tf.GradientTape() as gtape3:
                    conv_output, predictions = heatmap_model(img_tensor)
                    category_id = 1*(predictions[0].numpy()>=self.threshold)
                    if self.mapping:
                        print(self.mapping[category_id[0]])
                    output = predictions[:, category_id[0]]
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
        grad_CAM_map = np.sum(deep_linearization_weights*conv_output[0], axis=2)

        heatmap = np.maximum(grad_CAM_map, 0)
        max_heat = np.max(heatmap)
        if max_heat == 0:
            max_heat = 1e-10
        heatmap /= max_heat
        return heatmap

        """ This funtions computers Guided Grad-CAM
        as well as visualises all 3 approaches"""
    def guided_gradcam_plus_fn(self, cls=-1, visualize=True, save=True):
       # print(self.img.shape)
        predictions = self.model['model'](self.img)

        if predictions.shape[0]>1:
            binary = [1*(p.numpy() >= self.threshold) for p in predictions]
            #softmax_like = [[p, 1-p] for p in predictions]

        else:
            binary = 1*(predictions.numpy() >= self.threshold)
           # softmax_like = [predictions, 1-predictions]

        binary = binary[0][0]
        pred_label = self.mapping[binary]
        print('Model prediction: '+ pred_label)
        print()
      #  print('Probabilities:')

     #   for i in range(2): # express single value bce pred as class-wise probabilities
       #     print('\t{}. {}\t{:.3f}'.format(i, self.mapping[i], softmax_like[0,i]))
    
        if cls == -1:
            cls = binary

        print()
        print("Explanation for '{}':".format(self.mapping[cls]))
        print()
        gradcam = self.gradcam(self.layer_name)
        gradcam_up = cv2.resize(np.array(gradcam), (480,480), cv2.INTER_LINEAR)
        gb = self.guided_backprop()
        print(gb.shape)
        guided_gradcam = gb * gradcam_up[..., np.newaxis]

        gradcam_plus = self.gradcam_plus()

        if save:
            jetcam = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
            jetcam = cv2.resize(jetcam, (480,480), cv2.INTER_LINEAR)

            jetcam = (np.float32(jetcam) + self.deprocess_image(self.img)) / 2
            cv2.imwrite('gradcam.jpg', np.uint8(jetcam))
            cv2.imwrite('guided_backprop.jpg', self.deprocess_image(gb))
            cv2.imwrite('guided_gradcam.jpg', self.deprocess_image(guided_gradcam))
        
        if visualize:
            plt.figure(figsize=(15, 10))
            plt.subplot(141)
            plt.title('GradCAM')
            plt.axis('off')
            plt.imshow(self.deprocess_image(self.img))
            plt.imshow(gradcam, cmap='jet', alpha=0.5)

            plt.subplot(142)
            plt.title('Guided Backprop')
            plt.axis('off')
            plt.imshow(np.flip(self.deprocess_image(gb), -1))
            
            plt.subplot(143)
            plt.title('Guided GradCAM')
            plt.axis('off')
            plt.imshow(np.flip(self.deprocess_image(guided_gradcam), -1))
            plt.show()

            plt.subplot(144)
            plt.title('GradCAM++')
            plt.axis('off')
            plt.imshow(np.flip(self.deprocess_image(gradcam_plus), -1))
  
        return gradcam, gb, guided_gradcam, gradcam_plus
    


class TorchVisuals():
    def __init__(self, data, model, threshold=0.5, layer_name=None):
        super(TorchVisuals, self).__init__()
        self.label = data[0]
        self.img = data[1]
        self.model = model
        self.threshold = threshold
        self.mapping = ['Negative', 'Positive']

        self.model['model'].eval()

    def attribute_image_features(self, pred, algorithm, **kwargs):
        self.model['model'].zero_grad()
        tensor_attributions = algorithm.attribute(self.img,
                                                target=pred,
                                                **kwargs
                                                )
        return tensor_attributions

    def captum_visuals(self):
        output = self.model['model'](self.img)
        out = out[0]
        if out.shape[0] > 1:
            binary = [1*(p >= self.threshold) for p in out]
            softmax_like = [[p, 1-p] for p in out]
        else:
            binary = 1*(out >= self.threshold)
            softmax_like = [out, 1-out] 

        pred_label = self.mapping[binary]

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

    def lime(self):
        img, label = self.data
        predictions = self.model['model'](img)

        if predictions.shape[0]>1:
            binary = [1*(p >= self.threshold) for p in predictions]
        else:
            binary = 1*(predictions >= self.threshold)

        pred_label = self.mapping[binary]
        label = self.mapping[label]
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(np.array(img[0]).astype('double'), model['model'], top_labels=1, hide_color=0, num_samples=100)
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=True)
        plt.imshow(mark_boundaries(temp, mask))
        plt.savefig(f"/MULTIX/DATA/nccid/lime_{self.model['model_name']}_'gt'{label}_{pred_label}.png")
        


# captum https://github.com/probayes/Covid19Xray/blob/master/pycovid19xray/explainer.py
    # guidedBP = GuidedBackprop(model=model)
    # gb_cam = guidedBP.guided_backprop(np.expand_dims(img_1,axis=0),(480,480))
    # guided_gradcam = deprocess_image(gb_cam)
    # plt.imshow(guided_gradcam)
    # plt.savefig('/content/guided_gradcam.png')
