import albumentations as A

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import argparse
import cv2
import random

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Multiply
from tensorflow.keras.layers import Lambda

# from keras import optimizers

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from tensorflow.keras.regularizers import l2
from tensorflow.python.keras import backend as K

import tensorflow.keras as keras

from tensorflow.keras.models import load_model

import os


from keras.models import load_model

from skimage.segmentation import mark_boundaries

from deep_viz_keras import utils

from tensorflow.python.keras.backend import eager_learning_phase_scope
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

try:
    import lime
except:
    sys.path.append(os.path.join('..', '..')) # add the current directory
    import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries

class KerasDropoutPrediction(object):
  """Sourced from: https://stackoverflow.com/questions/43529931/how-to-calculate-prediction-uncertainty-using-keras.
  Puts dropout layer in training mode."""
  def __init__(self,model):
    self.model = model
    self.f = K.function([self.model.layers[0].input],      
                              [self.model.output])
  def predict(self,x, n_iter=10):
    with eager_learning_phase_scope(value=1): # 0=test, 1=train
      Yt_hat = np.array([self.f((x))[0] for _ in range(n_iter)])
    
    # result = []
    # for _ in range(n_iter):
    #   result.append(self.f([x , 1]))

    # result = np.array(result).reshape(n_iter,len(x)).T
    return Yt_hat

def lime(img, label, model):
    if model.get_model_type() == 'keras':
        pred = model.call(img)
    else:
        model.eval()

        pass # do torch
    '''https://www.kaggle.com/yohanb/explaining-keras-model-with-lime'''

    incorrect_pred = pred.argmax(axis=1) != label

    explainer = lime_image.LimeImageExplainer(random_state=42)

    fig, ax = plt.subplots(5, 6, sharex='col', sharey='row')
    fig.set_figwidth(20)
    fig.set_figheight(16)
    indices = random.sample(range(sum(incorrect_pred)), 5)
    for j in range(5):
        explanation = explainer.explain_instance(img[incorrect_pred][indices[j]], 
                                                model.call, 
                                                top_labels=5, hide_color=0, num_samples=1000, 
                                                random_seed=42)
        ax[j,0].imshow(img[incorrect_pred][indices[j]])
        ax[j,0].set_title(label[incorrect_pred][indices[j]])
        for i in range(5):
            temp, mask = explanation.get_image_and_mask(i, positive_only=True, 
                                                        num_features=5, hide_rest=True)
            ax[j,i+1].imshow(mark_boundaries(temp / 2 + 0.5, mask))
            ax[j,i+1].set_title('p({}) = {:.4f}'.format(label, pred[incorrect_pred][indices[j]][i]))  
            plt.savefig(f'/MULTIX/DATA/HOME/lime_{label}_{pred[incorrect_pred][indices[j]][i]}.png')  


def guided_backprop(img, model):
    guided_bprop = utils.GuidedBackprop(model)
    mask = guided_bprop.get_mask(img) # compute the gradients
    show_image(mask)
    plt.savefig("/MULTIX/INPUT/DATA/guided_bprop.png")