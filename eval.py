import albumentations as A

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import argparse
import cv2
import random
import sys
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.python.keras import backend as K

from tensorflow.keras.models import load_model

import os

from skimage.segmentation import mark_boundaries

from tensorflow.python.keras.backend import eager_learning_phase_scope
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from matplotlib import pyplot as plt

#from models.res_attn import AttentionResNetModified
# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()

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

from residual_attn.res_attn import AttentionResNetModified

from dataloaders import make_generators

from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import FastGradientMethod


@tf.custom_gradient
def guidedRelu(x):
    def grad(dy):
        return tf.cast(dy>0,"float32") * tf.cast(x>0, "float32") * dy
    return tf.nn.relu(x), grad

# Reference: https://github.com/nguyenhoa93/cnn-visualization-keras-tf2
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

# # Reference: https://github.com/nguyenhoa93/cnn-visualization-keras-tf2
class GradCAM:
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

    def compute_heatmap(self, image, upsample_size, classIdx=None, eps=1e-5):
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output, self.model.output]
        )
        # record operations for automatic differentiation
            
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convOuts, preds) = gradModel(inputs)  # preds after softmax
            if classIdx is None:
                classIdx = np.argmax(preds)
            loss = preds[:, classIdx]

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
        cam = cv2.resize(cam, upsample_size,cv2.INTER_LINEAR)

        # convert to 3D
        cam3 = np.expand_dims(cam, axis=2)
        cam3 = np.tile(cam3, [1, 1, 3])

        return cam3



class DropoutPrediction(object):
  """Sourced from: https://stackoverflow.com/questions/43529931/how-to-calculate-prediction-uncertainty-using-keras.
  Converts dropout layer into a training mode for prediction."""
  def __init__(self,model, visualise=False):
    self.model = model
    self.f = K.function([self.model.layers[0].input],      
    
                              [self.model.output]) # function to apply dropout
    if visualise == True:
        self.visual_fn = [GradCAM(), lime, GuidedBackprop()]
    else:
        self.visual_fn = None
    
  def predict_with_uncertainty(self, x, n_iter=10):
    result = np.zeros((n_iter,) + x.shape)

    for iter in range(n_iter):
        result[iter] = self.f(x, 1)

        if self.visual_fn == True:
            out = [fn(x) for fn in self.visualise_fn]

    prediction = result.mean(axis=0)
    uncertainty = result.var(axis=0)
    return prediction, uncertainty, out

def lime(img, label, model):
    assert model.model_type == 'keras'
    pred = model(img)
    '''https://www.kaggle.com/yohanb/explaining-keras-model-with-lime'''
    print(pred)
    print(label)

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

def lime(img, label, model):
    assert model.model_type == 'keras'

    plt.imshow(img)
    img = np.expand_dims(img, axis=0)
    pred = model(img)

    pred = np.asarray(pred)

    img = img.astype(np.float64)
    '''https://www.kaggle.com/yohanb/explaining-keras-model-with-lime'''
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(img[0], model.call, top_labels=1, hide_color=0, num_samples=10)
    temp, mask = explanation.get_image_and_mask(0, positive_only=False, num_features=10, hide_rest=False)
    plt.imshow(mark_boundaries(temp, mask))

## Adversarial attack
#pip install adversarial-robustness-toolbox
## https://github.com/Trusted-AI/adversarial-robustness-toolbox
def art(batch_x, batch_y, model):

  x = batch_x[0]
  x = np.expand_dims(x, axis=0)

  preds = model(x).numpy()

  # Create ART classifier for TensorFlow2.x and eager execution
  loss_object = tf.keras.losses.BinaryCrossentropy()
  nb_classes = 2
  # Define preprocessing using the mean values defined in from tensorflow.keras.applications.resnet50.preprocess_input
  # Expecting input images in BGR format
  preprocessing = (np.asarray([103.939, 116.779, 123.68]), 1)
  clip_values = (0, 255)

  # Create ART classifier
  classifier = TensorFlowV2Classifier(model, nb_classes=nb_classes, loss_object=loss_object,
                                      #preprocessing=preprocessing,
                                      clip_values=clip_values, input_shape=(3,480,480))

  # Evaluate ART classifier
  x = batch_x[0]
  x = image.img_to_array(x)
  x = np.expand_dims(x, axis=0)

  # Convert RGB to BGR
  x = x[..., ::-1]

  preds = classifier.predict(x)
  
  # Create FGSM attack and create adversarial example for tractor
  attack = FastGradientMethod(classifier=classifier, norm=np.inf, eps=8, eps_step=8, targeted=False,
                              num_random_init=0, batch_size=1, minimal=False)

  x_adv = attack.generate(x)

  perturbation = np.mean(np.abs((x_adv - x)))
 # print('Accuracy on adversarial test data: {:4.2f}%'.format(accuracy_test * 100))
  print('Average perturbation: {:4.2f}'.format(perturbation))

  # ResNet50 model now predicts `plow` with probability 0.39
  preds_adv = model(x_adv).numpy()
 # print('Predicted by modl on adversarial example:', decode_predictions(preds_adv, top=3)[0])

  # Plot adversarial example
  plt.imshow(x_adv[0][..., ::-1] / 255)

  # Calculate L_inf of perturbation, should be equal to eps
  l_inf = np.amax(x_adv[0, :, :, 0] - x[0, :, :, 0])
  print('Perturbation l_inf norm: %d' % l_inf)


if __name__ == "__main__":
    model = AttentionResNetModified()
    model.built = True
    model.load_weights('/content/drive/MyDrive/covid-19-benchmarking/res_attn.h5')

    # df = pd.read_csv()
    # mapping = {'negative':0, 'positive':1}
    
    # df = df[df['xray_status']!=np.nan]
    # df = df.dropna(subset=['xray_status'])
    
    # df['xray_status'] = df['xray_status'].map(mapping)

    params = {'batchsize':8, 'num_workers':1}

    df = pd.read_csv('/content/drive/MyDrive/new_covidx_data.csv')

    root = '/content/new_data'

    fullpath = []
    for idx, row in df.iterrows():
      full = root +'/' + row['split'] +'/' + row['finding'] + '/' + row['img']
      fullpath.append(full)

    df['paths'] = fullpath

    mapping = {'COVID-19':1.0, 'pneumonia':1.0, 'normal':0.0}

    # mapping = {'cats':0.0, 'dogs':1.0}
    df['outcome'] = df['finding'].map(mapping)

    df['paths'] = fullpath
    print(df['paths'].values)

    mapping = {'COVID-19':1.0, 'pneumonia':1.0, 'normal':0.0}

    # mapping = {'cats':0.0, 'dogs':1.0}
    df['outcome'] = df['finding'].map(mapping)

    #df = df.sample(frac=0.2)
    train_df = df[df['split']=='train']
    val_sample = train_df.sample(frac=0.2)
    train_df = train_df.drop(val_sample.index)
    val_sample['split'] = 'val'

    df = pd.concat([train_df, val_sample])
    print(df.head())

    print(df['split'].value_counts())

    #make generators
    params = {'batchsize':36, "num_workers":1, "k":1}

    train_df = df[df[f'split'] == "train"]
    val_df = df[df[f'split'] == "val"]
    test_df = df[df[f'split'] == 'test']
    
    train_loader, val_loader, test_loader = make_generators(model.model_type, train_df, val_df, test_df, params)

    for batch in tqdm(val_loader):
        if len(batch) > 1:
            batch_x, batch_y = batch # batch_y can be paired image 
            with tf.GradientTape() as tape:
                pred = model(batch_x)
              #  lime(batch_x[0], batch_y[0], model)
                art(batch_x, batch_y, model)






