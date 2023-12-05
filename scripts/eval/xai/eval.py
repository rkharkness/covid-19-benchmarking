import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

from skimage.segmentation import mark_boundaries

#from tensorflow.keras.backend import eager_learning_phase_scope
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score, auc, precision_recall_curve, confusion_matrix, classification_report, roc_curve

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from matplotlib import pyplot as plt

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from mpl_toolkits.axes_grid1 import ImageGrid

try:
    import lime
except:
    sys.path.append(os.path.join('..', '..')) # add the current directory
    import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries

from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2

from tqdm import tqdm

from tf_explain.core.grad_cam import GradCAM
# from tf_explain.core.gradients_inputs import GradientsInputs
from tf_explain.core.activations import ExtractActivations
from tf_explain.core.occlusion_sensitivity import OcclusionSensitivity
from tf_explain.core.integrated_gradients import IntegratedGradients
from tf_explain.core.smoothgrad import SmoothGrad
from tf_explain.core.vanilla_gradients import VanillaGradients

from models.res_attn import AttentionResNetModified
from models.ecovnet import ECovNet
from models.coronet_tfl import CoroNet
from deep_viz_keras.saliency import GradientSaliency
#from deep_viz_keras.integrated_gradients import IntegratedGradients
from deep_viz_keras.visual_backprop import VisualBackprop


from dataloaders import make_generators

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


#from art.estimators.classification import TensorFlowV2Classifier
#from art.attacks.evasion import FastGradientMethod
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

@tf.custom_gradient
def guidedRelu(x):
    def grad(dy):
        return tf.cast(dy>0,"float32") * tf.cast(x>0, "float32") * dy
    return tf.nn.relu(x), grad

# Reference: https://github.com/nguyenhoa93/cnn-visualization-keras-tf2
class GuidedBackprop(object):
    def __init__(self,model, layerName=None):
        self.model = model['model']
        self.layerName = layerName
        if self.layerName == None:
            self.layerName = self.find_target_layer()

        self.gbModel = self.build_guided_model()
        
    def find_target_layer(self): #base_pretrained_model.layers[0].compute_output_shape(input_shape = t_x.shape[1:])
        for node_index, layer in enumerate(reversed(self.model.layers[:-1])):
            if len(layer.compute_output_shape(input_shape = (None,480,480,3))) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM")

    def build_guided_model(self):
        gbModel = Model(
            inputs = [self.model.inputs],
            outputs = [self.model.get_layer(self.layerName).output]
        )

        layer_dict = [layer for layer in gbModel.layers[1:] if hasattr(layer,"activation")]
        for layer in layer_dict:
            if layer.activation == tf.keras.activations.relu:
                layer.activation = guidedRelu
        return gbModel
    
    def guided_backprop(self, images, upsample_size=(480,480)):
        """Guided Backpropagation method for visualizing input saliency."""
        with tf.GradientTape() as tape:
            inputs = tf.cast(images, tf.float32)
            inputs = tf.expand_dims(inputs, 0)
            tape.watch(inputs)
            outputs = self.gbModel(inputs)

        grads = tape.gradient(outputs, inputs)[0]

        saliency = cv2.resize(np.asarray(grads), upsample_size)

        return saliency


# class DropoutPrediction(object):
#   """Sourced from: https://stackoverflow.com/questions/43529931/how-to-calculate-prediction-uncertainty-using-keras.
#   Converts dropout layer into a training mode for prediction."""
#   def __init__(self, model, visualise):
#     self.model = model
#     self.model.built = True
#     self.visualise = visualise

#     self.gradcam = GradCAM(self.model)
#     self.guided_backprop = GuidedBackprop(self.model)
#    # print(model.layers[0])
#    # self.f = K.function([self.model.layers[0].input], [K.learning_phase()],
#      #           [self.model.layers[0].output])
#     if self.visualise == True:
#         self.visual_fn = [self.gradcam.compute_heatmap, self.guided_backprop.guided_backprop]
#     else:
#         self.visual_fn = None

#   def predict_with_uncertainty(self, x, no_classes=2, n_iter=10):
#       x = tf.expand_dims(x[0], 0)
#       result = np.zeros((n_iter,) + (x.shape[0], no_classes) )
#       print(x.shape)
#       img_list = []
#       for i in range(n_iter):
#           result[i,:, :] = self.model(x)[0]

      #     if self.visualise == True:
      #         print('x',x.shape)
      #         out = self.gradcam.compute_heatmap(x)
      #         fig = plt.figure()
      #         print('o',out[0].shape)
      #         plt.imshow(out[0])
      #         plt.savefig('/content/drive/MyDrive/mcdo_sample.png')
      #         img_list.append(out)

      # fig = plt.figure()
      # grid = ImageGrid(fig, 111,  # similar to subplot(111)
      #                   nrows_ncols=(n_iter,2),  # creates 2x2 grid of axes
      #                   axes_pad=0.1,  # pad between axes in inch.
      #                   )

      # img_list = [y for x in img_list for y in x]
      # for ax, im in zip(grid, [img_list[i] for i in range(len(img_list))]):
      #     print('im', im.shape)
      #     ax.imshow(im)
      
      # plt.savefig('/content/drive/MyDrive/mcdo_vis.png')
    #  
    #  prediction = result.mean(axis=0)
    #  uncertainty = result.std(axis=0)
    #  
    #  return prediction, uncertainty, img_list

def lime(img, label, model):
    model = model['model']
    pred = model(img)
    '''https://www.kaggle.com/yohanb/explaining-keras-model-with-lime'''

    incorrect_pred = [int(x > 0.5) != y for x, y in zip(pred,label)]
    incorrect_pred = [int(i) for i in incorrect_pred]

    incorrect_pred = np.array(incorrect_pred)
    explainer = lime_image.LimeImageExplainer(random_state=42)

    fig, ax = plt.subplots(5, 6, sharex='col', sharey='row')
    fig.set_figwidth(20)
    fig.set_figheight(16)
    print(range(sum(incorrect_pred)))
    indices = random.sample(range(sum(incorrect_pred)), 3)
    print(indices)
    for j in range(3):
        print(img.shape)
        print(img[incorrect_pred])
        explanation = explainer.explain_instance(img[incorrect_pred][indices[j]], 
                                                model.call, 
                                                top_labels=5, hide_color=0, num_samples=1000, 
                                                random_seed=42)
        ax[j,0].imshow(img[incorrect_pred][indices[j]])
        ax[j,0].set_title(label[incorrect_pred][indices[j]])

        for i in range(3):
            temp, mask = explanation.get_image_and_mask(i, positive_only=True, 
                                                        num_features=5, hide_rest=True)
            ax[j,i+1].imshow(mark_boundaries(temp / 2 + 0.5, mask))
            ax[j,i+1].set_title('p({}) = {:.4f}'.format(label, pred[incorrect_pred][indices[j]][i]))  
            plt.savefig(f'/content/lime_{label}_{pred[incorrect_pred][indices[j]][i]}.png')  

def lime(img, model):
    '''https://www.kaggle.com/yohanb/explaining-keras-model-with-lime'''
 #   img = tf.expand_dims(img, 0)
    print(img.shape)
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(np.array(img[0]).astype('double'), model['model'], top_labels=1, hide_color=0, num_samples=100)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
    plt.imshow(mark_boundaries(temp, mask))
    plt.savefig(f"/MULTIX/DATA/lime_{model['model_name']}.png")
    
@tf.custom_gradient
def guidedRelu(x):
    def grad(dy):
        return tf.cast(dy>0,"float32") * tf.cast(x>0, "float32") * dy
    return tf.nn.relu(x), grad

# Reference: https://github.com/eclique/keras-gradcam with adaption to tensorflow 2.0  
class GuidedBackprop:
    def __init__(self,model, layerName=None):
        self.model = model
        self.layerName = layerName
        if self.layerName == None:
            self.layerName = self.find_target_layer()

        self.gbModel = self.build_guided_model()
        
    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply Guided Backpropagation")

    def build_guided_model(self):
        gbModel = Model(
            inputs = [self.model.inputs],
            outputs = [self.model.get_layer(self.layerName).output]
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
            outputs = self.gbModel(inputs)

        grads = tape.gradient(outputs, inputs)[0]

        saliency = cv2.resize(np.asarray(grads), upsample_size)

        return saliency

    
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


class GradCAMPlusPlus:
    def __init__(self, model, layerName=None):
        self.model = model
        self.layerName = layerName
        if self.layerName == None:
            self.layerName = self.find_target_layer()

        self.gc_plus = self.build_plus_model()
        
    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply Guided Backpropagation")

    def build_plus_model(self):
        PlusModel = Model(
            inputs = [self.model.inputs],
            outputs = [self.model.get_layer(self.layerName).output, self.model.output]
        )
        return PlusModel

    def grad_cam_plus(self, img, label_name=None,
                        category_id=None):
        """Get a heatmap by Grad-CAM++.
        Args:
            model: A model object, build from tf.keras 2.X.
            img: An image ndarray.
            layer_name: A string, layer name in model.
            label_name: A list or None,
                show the label name by assign this argument,
                it should be a list of all label names.
            category_id: An integer, index of the class.
                Default is the category with the highest score in the prediction.
        Return:
            A heatmap ndarray(without color).
        """
        img_tensor = np.expand_dims(img, axis=0)
        print(self.gc_plus.layers)
        with tf.GradientTape() as gtape1:
            with tf.GradientTape() as gtape2:
                with tf.GradientTape() as gtape3:
                    conv_output, predictions = self.gc_plus(img_tensor)
                    if category_id==None:
                        category_id = 0 #int(predictions[0]>0.5)
                    if label_name:
                        print(label_name[category_id])
                    output = predictions[:, category_id]
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

        cam = cv2.resize(heatmap, (480,480),cv2.INTER_LINEAR)
        # convert to 3D
        cam3 = np.expand_dims(cam, axis=2)
        cam3 = np.tile(cam3, [1, 1, 3])

        return cam3

class Gradcam:
    # Adapted with some modification from https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/
    def __init__(self, model, layerName=None):
        """
        model: pre-softmax layer (logit layer)
        """
        self.model = model
        self.layerName = layerName

        if self.layerName == None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM")

    def compute_heatmap(self, image, classIdx, upsample_size, eps=1e-5):
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output, self.model.output]
        )
        # record operations for automatic differentiation
        image = tf.expand_dims(image, 0)
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convOuts, preds) = gradModel(inputs)  # preds after softmax
            # loss = preds[:,:, classIdx]
            loss = preds

        # compute gradients with automatic differentiation
        grads = tape.gradient(loss, convOuts)
        # discard batch
        convOuts = convOuts[0]
        grads = grads[0]
        norm_grads = tf.divide(grads, tf.reduce_mean(tf.square(grads)) + tf.constant(eps))

        # compute weights
        weights = tf.reduce_mean(norm_grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOuts), axis=-1)

        # Apply reLU
        cam = np.maximum(cam, 0)
        cam = cam / np.max(cam)
        cam = cv2.resize(cam, (upsample_size),cv2.INTER_LINEAR)

        # convert to 3D
        cam3 = np.expand_dims(cam, axis=2)
        cam3 = np.tile(cam3, [1, 1, 3])

        return cam3


def overlay_gradCAM(img, cam3):
    cam3 = np.uint8(255 * cam3)
    cam3 = cv2.applyColorMap(cam3, cv2.COLORMAP_JET)

    new_img = 0.3 * cam3 + 0.5 * img

    return (new_img * 255.0 / new_img.max()).astype("uint8")

def show_imgwithheat(img, heatmap, alpha=0.3):
    """Show the image with heatmap.
    Args:
        img_path: string.
        heatmap:  image array, get it by calling grad_cam().
        alpha:    float, transparency of heatmap.
        return_array: bool, return a superimposed image array or not.
    Return:
        None or image array.
    """
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap, (224, 224))
    img = cv2.resize(img, (224,224))

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    new_img = alpha * heatmap + 0.5 * img

#        new_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    return (new_img * 255.0 / new_img.max()).astype("uint8")



def main(model, df):

    tprs = []
    aucs = []
    thresh_list = []
    mean_fpr = np.linspace(0, 2, 100)
    plt.figure(figsize=(10,10))

    for k in range(1,6):
        model['model'].load_weights(f'/MULTIX/DATA/nccid/weights/coronet_tfl/coronet_tfl_{k}.h5')
        model['model'].built=True
        model['model'].trainable=False

        train_df = df[df[f'kfold_{k}'] == "train"]
        val_df = df[df[f'kfold_{k}'] == "val"]
        test_df = df[df[f'kfold_{k}'] == 'test']

        params = {'batchsize':4, "num_workers":1, "k":k}

        _, _, test_loader = make_generators(model['model_type'], train_df, val_df, test_df, params)

        gt_list = []
        pred_list = []
        output_list = []
        for img, label in test_loader:
            pred = model['model'](img)
            pred_list.append(pred)
            gt_list.append(label)


        fpr, tpr, thresholds = roc_curve(np.array(gt_list).flatten(), np.array(pred_list).flatten(),drop_intermediate=False)

        thresh_list.append(thresholds)
        
        roc_auc = auc(fpr, tpr)    

        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                label='ROC fold %d (AUC = %0.2f)' % (k, roc_auc))

        lw=2
        plt.plot(fpr, tpr, lw=lw, label=f'(area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f"{model['model_name']}")
        plt.legend(loc="lower right")
        plt.savefig(f"/MULTIX/DATA/nccid/{model['model_name']}_roc_{k}.png")

        J = tpr - fpr
        ix = np.argmax(J)
        best_thresh = thresholds[ix]

        gt_list = np.array(gt_list).flatten()
        pred_list = np.array(pred_list).flatten()
        output_list = np.array([np.int(i>best_thresh) for i in pred_list])

        print(classification_report(gt_list, output_list, target_names=['Negative','Positive'])) # performance metrics
        print(confusion_matrix(gt_list, output_list))
        plt.savefig(f"/MULTIX/DATA/nccid/{model['model_name']}_cm_{k}.png")

        # visualisations
        if k == 1:
            img, label = next(iter(test_loader))

            img_1 = img[0]
            label_1 = label[0]

            img_1 = np.expand_dims(img_1.numpy(), 0)
            vanilla = GradientSaliency(model)
            mask = vanilla.get_mask(img_1, np.expand_dims(label_1.numpy(), 0))
            show_image(mask[0], ax=plt.subplot(1,2,1), title='vanilla gradient')

            mask = vanilla.get_smoothed_mask(img, np.expand_dims(label_1.numpy(), 0))
            show_image(mask[0], ax=plt.subplot(1,2,2), title='smoothed vanilla gradient')
            plt.savefig(f"/MULTIX/DATA/nccid/{model['model_name']}_smooth_van.png")
            
#            plt.figure()
 #           visual_bprop = VisualBackprop(model)
  #          mask = visual_bprop.get_mask(img_1)
   #         show_image(mask[0], ax=plt.subplot('121'), title='visual backprop')
  #          print('vprop')
   #         mask = visual_bprop.get_smoothed_mask(img_1)
   #         show_image(mask[0], ax=plt.subplot('122'), title='smoothed visual backprop')
    #        print('save')
    #        plt.savefig(f"/MULTIX/DATA/nccid/{model['model_name']}_grad_back.png")

            gc = Gradcam(model=model['model'])
            cam3 = gc.compute_heatmap(img_1[0], 0, (480,480))

            gradcam = overlay_gradCAM(img_1, cam3)
            gradcam = cv2.cvtColor(gradcam, cv2.COLOR_BGR2RGB)
            
            # # Guided backprop
            guidedBP = GuidedBackprop(model=model['model'])
            gb = guidedBP.guided_backprop(img_1,(480,480))
            #gb_im = deprocess_image(gb)
           # gb_im = cv2.cvtColor(gb_im, cv2.COLOR_BGR2RGB)
                # Guided GradCAM

            guided_gradcam = deprocess_image(gb*cam3)
            guided_gradcam = gb*cam3

            guided_gradcam = cv2.cvtColor(guided_gradcam, cv2.COLOR_BGR2RGB)
            plt.figure()
            plt.imshow(guided_gradcam)
            plt.savefig(f"/MULTIX/DATA/nccid/{model['model_name']}_guided_gradcam.png")

            gcpp = GradCAMPlusPlus(model['model'])
            sal = gcpp.grad_cam_plus(img_1[0])
            lime_img = tf.cast(img_1, tf.float32)
            lime(img, model)

          #  sal = show_imgwithheat(img_1, sal)
           # plt.figure()
           # plt.imshow(sal)
           # plt.savefig(f"/MULTIX/DATA/nccid/{model['model_name']}_gc_plus_plus.png")

            data = (np.array(img), None)
            explainer = GradCAM()
            gc_grid = explainer.explain(data, model['model'], class_index=0)
            explainer.save(gc_grid, "/MULTIX/DATA/nccid", f"{model['model_name']}_grad_cam.png")

            # """## Activation Maps"""
            # target_layers = ["block7b_project_conv"]
            # explainer = ExtractActivations()
            # grid = explainer.explain(data, model['model'], target_layers)    
            # explainer.save(grid, '.', "activation.png")

            """## Occlusion Sensitivity"""
            explainer = OcclusionSensitivity()
            grid = explainer.explain(data, model['model'], 0, patch_size=10)
            explainer.save(grid, "/MULTIX/DATA/nccid",f"{model['model_name']}_occlusion_analysis10.png")

            # explainer = OcclusionSensitivity()
            # grid = explainer.explain(data, model['model'], 0, patch_size=10)
            # explainer.save(grid, ".", "occlusion_analysis10.png")

            """## Integrated Gradients"""
            class_index = 0
            explainer = IntegratedGradients()

            grid = explainer.explain(data, model['model'], class_index, n_steps=15)
            explainer.save(grid,  "/MULTIX/DATA/nccid", f"{model['model_name']}_integrated_gradients.png")

            # """## Smooth Grad"""
            # class_index = 0
            # explainer = SmoothGrad()

            # grid = explainer.explain(data, model['model'], class_index, num_samples=20, noise=1.0)
            # explainer.save(grid, ".", "smooth_gradients.png")

            # """## Vanilla Gradients"""
            # class_index = 0
            # explainer = IntegratedGradients()

            # grid = explainer.explain(data, model['model'], class_index)
            # explainer.save(grid, ".", "vanilla_gradients.png")
    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate',fontsize=10)
    plt.ylabel('True Positive Rate',fontsize=10)
    plt.title(f"Cross-Validation ROC of {model['model_name']}",fontsize=14)
    plt.legend(loc="lower right", prop={'size': 8})
    plt.savefig(f"/MULTIX/DATA/nccid/{model['model_name']}_roc.png")
    plt.show()


if __name__ == "__main__":   
    parser = argparse.ArgumentParser(description='Model Training Script')
    parser.add_argument('--data_csv', default='/MULTIX/DATA/nccid_preprocessed.csv', type=str, help='Path to data file')
    args = parser.parse_args()

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    assert tf.test.is_gpu_available()
    assert tf.test.is_built_with_cuda()
    
    df = pd.read_csv(args.data_csv)
    mapping = {'negative':0, 'positive':1}
    
    df = df[df['xray_status']!=np.nan]
    df = df.dropna(subset=['xray_status'])
    
    df['xray_status'] = df['xray_status'].map(mapping)

    ecovnet = CoroNet(dropout_act=False)
    model = ecovnet.build_model()
    
    main(model, df)



    # guidedBP = GuidedBackprop(model=model)
    # gb_cam = guidedBP.guided_backprop(np.expand_dims(img_1,axis=0),(480,480))
    # guided_gradcam = deprocess_image(gb_cam)
    # plt.imshow(guided_gradcam)
    # plt.savefig('/content/guided_gradcam.png')
