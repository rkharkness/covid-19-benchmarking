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

    
class CoroNet_Tfl_Seg(Model):
    def __init__(self, dropout_act=True):
        super(CoroNet_Tfl_Seg, self).__init__()

        self.optimizer =  optimizers.SGD()
        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=False) #, label_smoothing=0.3) # WeightedBCE()
        self.lr = 1e-4
        self.model_name = 'coronet_tfl_seg'
        self.supervised = True
        self.dropout_act = dropout_act
        self.model_type = 'keras'

        self.model = self.build_model()
    
    def build_model(self):
        conv_base = Xception(weights='imagenet',
                        include_top=False,
                        input_shape=(480,480, 3))
        conv_base.trainable = True
        conv_base = Model(conv_base.input, conv_base.output)


        x = conv_base.output
        x = layers.GlobalAveragePooling2D()(x)

        x = layers.Flatten()(x)
        x = layers.Dropout(0.3)(x, training=self.dropout_act)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x, training=self.dropout_act)
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
