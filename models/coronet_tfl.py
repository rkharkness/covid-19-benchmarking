from numpy.random import seed
seed(8) #1

import tensorflow
tensorflow.random.set_seed(7)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
import numpy as np

import keras
import tensorflow as tf

from tensorflow.keras import layers

from tensorflow.keras import optimizers

from tensorflow.keras.applications import Xception

class WeightedBCE(keras.losses.Loss):
  def __init__(self, trial=None):
    """adapted from: https://stackoverflow.com/questions/46009619/keras-weighted-binary-crossentropy"""            
    super().__init__()
    self.trial = trial
    self.weights = {'0':1.32571275, '1':0.80276873}

  def call(self, y_true, y_pred):
        # Original binary crossentropy (see losses.py):
        # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Calculate the binary crossentropy
        b_ce = K.binary_crossentropy(y_true, y_pred)

        # Apply the weights
        weight_vector = y_true * self.weights['1'] + (1. - y_true) * self.weights['0']
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return K.mean(weighted_b_ce)
    
class CoroNet(Model):
    def __init__(self, dropout_act=True):
        super(CoroNet, self).__init__()

        self.optimizer =  optimizers.Adam()
        self.loss_fn = WeightedBCE()
        self.lr = 1e-4
        self.model_name = 'coronet_tfl'
        self.supervised = True
        self.dropout_act = dropout_act
        self.model_type = 'keras'

        self.model = self.build_model()
    
    def build_model(self):
        conv_base = Xception(weights='imagenet',
                        include_top=False,
                        input_shape=(480,480, 3))
        conv_base.trainable = True
        conv_base = Model(conv_base.input, conv_base.layers[-37].output)


        x = conv_base.output
        x = layers.AveragePooling2D(12)(x)

        x = layers.Flatten()(x)
        x = layers.Dropout(0.5)(x, training=self.dropout_act)
        x = layers.Dense(256, activation='relu')(x)
        prediction = layers.Dense(1, activation='sigmoid')(x)

        model = Model(inputs=conv_base.input, outputs=prediction)
        
        model = {'model':model, 'optimizer':self.optimizer, 'loss_fn':self.loss_fn, 'lr':self.lr,
            'model_name':self.model_name, 'model_type':self.model_type,'supervised':self.supervised}
        return model
    
    def call(self, inputs):
      output = self.model['model'](inputs)
      return output

if __name__ == "__main__":
  coronet = CoroNet()
  model = coronet.build_model()
  model = model['model']
  print(model.summary())
  for layer in model.layers:
      print(layer.name)
