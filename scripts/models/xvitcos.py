import tensorflow as tf
#import tensorflow_addons as tfa
from vit_keras import vit, utils, visualize
from math import ceil
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense


class xVitCOS():
    def __init__(self, dropout_act=True, pretrained=None):
        super(xVitCOS, self).__init__()
        self.pretrained = pretrained
        if self.pretrained != None:
            self.num_classes = 1
        else:
            self.num_classes = 5

        self.model_name = 'xvitcos'
        self.model_type = 'keras'
        self.optimizer = Adam()
        self.lr = 1e-4
        self.loss_fn = keras.losses.BinaryCrossentropy(
            from_logits=False,
         )
        self.supervised = True
        self.dropout_act = dropout_act
        print(self.num_classes)
        
    def build_model(self):
        base_model = vit.vit_b16(
            image_size=480,
            activation='softmax', # sigmoid?
            pretrained=True,
            include_top=False,
            pretrained_top=False
        )
        x = base_model.output
        x = tf.keras.layers.Dropout(0.3)(x, training=self.dropout_act)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        y = tf.keras.layers.Dense(5, activation='sigmoid')(x)
        model = Model(base_model.input, y)
        if self.pretrained != None:
          if self.pretrained !=True:
            model.load_weights(self.pretrained)
          x = Dense(1, activation='sigmoid')(model.layers[-2].output)
          model = Model(model.input, x)
        return {'model':model, 'optimizer':self.optimizer, 'loss_fn':self.loss_fn, 'lr':self.lr,
            'model_name':self.model_name, 'pretrained':self.pretrained, 'model_type':self.model_type,'supervised':self.supervised}


    def call(self, inputs):
      output = self.model['model'](inputs)
      return output

if __name__ == "__main__":
  xvitcos = xVitCOS()
  model = xvitcos.build_model()
  model = model['model']
  print(model.summary())
  for layer in model.layers:
      print(layer.name)