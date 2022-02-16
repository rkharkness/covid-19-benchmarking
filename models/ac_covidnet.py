from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, RepeatVector, Lambda, Multiply, Conv2D, Reshape
from tensorflow.keras.layers import BatchNormalization, Concatenate, AveragePooling2D, Flatten, Conv2DTranspose, Average, add
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Flatten, MaxPooling2D, DepthwiseConv2D

from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute, Activation, Input, add, multiply
from tensorflow.keras.layers import concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
import tensorflow.keras.backend as K
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow as tf

import tensorflow_addons as tfa
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

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
        y_pred = tf.cast(y_pred+1e-10, tf.float32)

        # Calculate the binary crossentropy
        b_ce = K.binary_crossentropy(y_true, y_pred)

        # Apply the weights
        weight_vector = y_true * self.weights['1'] + (1. - y_true) * self.weights['0']
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return K.mean(weighted_b_ce)

class SupervisedContrastiveLoss(keras.losses.Loss):
    def __init__(self, temperature=1, name=None):
        super(SupervisedContrastiveLoss, self).__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)

def attention_block_2d(x, g, data_format='channels_first'):
    inter_channel = g.get_shape().as_list()[1]
    theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(x)
    phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(g)
    f = Activation('relu')(add([theta_x, phi_g]))
    psi_f = Conv2D(1, [1, 1], strides=[1, 1], data_format=data_format)(f)

    rate = Activation('sigmoid')(psi_f)
    att_x = multiply([x, rate])

    return att_x

def Pepx(filters, kernel=(1, 1)):
    def inside(x):
        x = Conv2D(filters, kernel, padding='same', activation = 'relu') (x)
        x = Conv2D(filters, kernel, padding='same', activation = 'relu') (x)
        x = DepthwiseConv2D((3, 3), padding='same') (x)
        x = Conv2D(filters, kernel, padding='same', activation = 'relu') (x)
        x = Conv2D(filters, kernel, padding='same', activation = 'relu') (x)
        return x
    return inside

# usage:
# x = Pepx(params) (x)

class ACCovidNet(Model):
    optimizer = Adam()
    model_name = 'ac_covidnet'
    model_type = 'keras'

    def __init__(self, shape=(480,480, 3), supcon=True, dropout_act=True):
      super(ACCovidNet, self).__init__()

      self.supcon = supcon # true - means pretrain encoder, false means train classifier

      self.shape = shape
      self.learning_rate = 0.001
      self.hidden_units = 512
      self.projection_units = 128
      self.dropout_rate = 0.5
      self.dropout_act = True

      self.temperature = 0.05

      self.supervised = True
      
      if self.supcon == True:
          self.loss_fn = SupervisedContrastiveLoss(self.temperature)
      else:
          self.loss_fn = WeightedBCE()
      
      self.num_classes = 1

      self.model = self.build_model()

    def build_model(self):
      i = Input(shape=self.shape)
      ip = MaxPooling2D(pool_size=(2, 2), padding='same') (i)
      c1 = Conv2D(kernel_size=(7, 7), filters=56, activation='relu', padding='same') (ip)
      c1p = MaxPooling2D(pool_size=(2, 2), padding='same') (c1)

      ###### PEPX BLOCKS + Conv1x1 ######
      p1_1 = Pepx(56) (c1p)
      cr1 = Conv2D(kernel_size=(1, 1), filters=56, activation='relu', padding='same') (c1p)
      concat_cr1_p12 = add([p1_1, cr1])
      p1_2 = Pepx(56) (concat_cr1_p12)
      concat_cr1_p13 = add([p1_2, cr1, p1_1])
      p1_3 = Pepx(56) (concat_cr1_p13)
      concat_p1_cr2 = add([p1_1, p1_2, p1_3, cr1])
      cr1p = MaxPooling2D(pool_size=(2, 2), padding='same') (concat_p1_cr2)

      a1 = attention_block_2d(p1_3, Average()([p1_1, p1_2]))
      concat_cr1_p21 = add([cr1, a1])
      p1_3p = MaxPooling2D(pool_size=(2, 2), padding='same') (concat_cr1_p21)

      p2_1 = Pepx(112) (p1_3p)
      cr2 = Conv2D(kernel_size=(1, 1), filters=112, activation='relu', padding='same') (cr1p)
      concat_cr2_p22 = add([p2_1, cr2])
      p2_2 = Pepx(112) (concat_cr2_p22)
      concat_cr2_p23 = add([p2_2, cr2, p2_1])
      p2_3 = Pepx(112) (concat_cr2_p23)
      concat_cr2_p24 = add([p2_3, cr2, p2_1, p2_2])
      p2_4 = Pepx(112) (concat_cr2_p24)
      concat_p2_cr3 = add([p2_1, p2_2, p2_3, p2_4, cr2])
      cr2p = MaxPooling2D(pool_size=(2, 2), padding='same') (concat_p2_cr3)

      a2 = attention_block_2d(p2_4, Average()([p2_1, p2_2, p2_3]))
      concat_cr2_p31 = add([cr2, a2])
      p2_4p = MaxPooling2D(pool_size=(2, 2), padding='same') (concat_cr2_p31)

      p3_1 = Pepx(216) (p2_4p)
      cr3 = Conv2D(kernel_size=(1, 1), filters=216, activation='relu', padding='same') (cr2p)
      concat_cr3_p32 = add([p3_1, cr3])
      p3_2 = Pepx(216) (concat_cr3_p32)
      concat_cr3_p33 = add([p3_2, cr3, p3_1])
      p3_3 = Pepx(216) (concat_cr3_p33)
      concat_cr3_p34 = add([p3_3, cr3, p3_1, p3_2])
      p3_4 = Pepx(216) (concat_cr3_p34)
      concat_cr3_p35 = add([p3_4, cr3, p3_1, p3_2, p3_3])
      p3_5 = Pepx(216) (concat_cr3_p35)
      concat_cr3_p36 = add([p3_5, cr3, p3_1, p3_2, p3_3, p3_4])
      p3_6 = Pepx(216) (concat_cr3_p36)
      concat_p3_cr4 = add([p3_1, p3_2, p3_3, p3_4, p3_5, p3_6, cr3])
      cr3p = MaxPooling2D(pool_size=(2, 2), padding='same') (concat_p3_cr4)

      a3 = attention_block_2d(p3_6, Average()([p3_1, p3_2, p3_3, p3_4, p3_5]))
      concat_cr3_p41 = add([cr3, a3])
      p3_6p = MaxPooling2D(pool_size=(2, 2), padding='same') (concat_cr3_p41)

      p4_1 = Pepx(424) (p3_6p)
      cr4 = Conv2D(kernel_size=(1, 1), filters=424, activation='relu', padding='same') (cr3p)
      concat_cr4_p4_2 = add([p4_1, cr4])
      p4_2 = Pepx(424) (concat_cr4_p4_2)
      concat_cr4_p4_3 = add([p4_2, cr4, p4_1])

      p4_3 = Pepx(424) (p4_2)
      #########################################

      af = attention_block_2d(cr4, Average()([p4_1, p4_2, p4_3]))
      f = Flatten() (af)
      fc1 = Dense(1024, activation = 'relu')(f)

      fc1 = Dropout(self.dropout_rate)(fc1, training=self.dropout_act) # added dropout layer

      fc2 = Dense(256, activation = 'relu')(fc1)

      fc3 = Dense(self.num_classes, activation='sigmoid')(fc2)
      self.model = Model(inputs = i, outputs = fc3)

      if self.supcon == True:
        encoder = self.create_encoder()
        model = self.add_projection_head(encoder)
      else:
        encoder = self.create_encoder()
        model = self.create_classifier(encoder)

      model = {'model': model, 'optimizer':self.optimizer, 'loss_fn':self.loss_fn,
              'model_name':self.model_name, 'model_type':self.model_type, 
              'supervised':self.supervised}

      return model      
    
    def create_encoder(self):
        accovidnet = self.model
        model = Model(inputs = accovidnet.input, outputs = accovidnet.layers[-3].output)
        return model

    def add_projection_head(self, encoder):
        features = encoder.output
        outputs = layers.Dense(self.projection_units, activation="relu")(features)
        model = keras.Model(
            inputs=encoder.input, outputs=outputs, name="cifar-encoder_with_projection-head"
        )
        return model

    def create_classifier(self, encoder):
        # Adam()
        for layer in encoder.layers:
            layer.trainable = self.supcon # freeze encoder weights if training classifier

        accovidnet = self.model
        model = keras.Model(inputs = encoder.input, outputs = accovidnet.output)
        return model

    def call(self, inputs):
      output = self.model['model'](inputs)
      return output
  

if __name__ == '__main__':
    ac_covidnet = ACCovidNet(supcon=True)
    model = ac_covidnet.build_model()
    model = model['model']
    model.summary()
    for layer in model.layers:
        print(layer.name)