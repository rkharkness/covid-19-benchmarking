
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 15 19:20:46 2021
@author: nextgen
"""


import os
import keras

import tensorflow as tf

import re

import glob

import tarfile
from tensorflow.keras.optimizers import Adam
import os
import warnings
import pandas as pd
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import random
from PIL import Image
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageOps
import math
import os
import random
import copy
import numpy as np 
import os
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

from tensorflow.keras import backend as keras
from keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.losses import Loss, MeanSquaredError
from tensorflow.keras import losses
import os
import shutil

class configs:
    model = "Vnet"
    suffix = "genesis_chest_ct"
    exp_name = model + "-" + suffix

    train_fold = [0, 1, 2, 3, 4]
    valid_fold = [5, 6]
    test_fold = [7, 8, 9]
    hu_min = -1000.0
    hu_max = 1000.0
    scale = 32
    input_rows = 64
    input_cols = 64
    input_deps = 32
    nb_class = 1

    # model pre-training
    verbose = 1
    weights = None
    batch_size = 6
    optimizer = "sgd"
    workers = 10
    max_queue_size = workers * 4
    save_samples = "png"
    nb_epoch = 10000
    patience = 50
    lr = 1e1 # lr scheduler

    # image deformation
    nonlinear_rate = 0.9
    paint_rate = 0.9
    outpaint_rate = 0.8
    inpaint_rate = 1.0 - outpaint_rate
    local_rate = 0.5
    flip_rate = 0.4


try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb

def loss_fn():
    def ssim_loss_minusone(y_true, y_pred):
        return 1-tf.reduce_mean(tf.image.ssim(tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred), 2.0))
    return ssim_loss_minusone

def ssim_loss(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred), 2.0))

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """
    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

 
def bezier_curve(points, nTimes=1000):

    """
       Given a set of control points, return the
       bezier curve defined by the control points.
 
       Control points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000
 
        See http://processingjs.nihongoresources.com/bezierinfo/
    """
    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])
    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


def data_augmentation(x, y, prob=0.5):
    # augmentation by flipping
    cnt = 3
    while random.random() < prob and cnt > 0:
        degree = random.choice([0, 1, 2])
        x = np.flip(x, axis=degree)
        y = np.flip(y, axis=degree)
        cnt = cnt - 1

    return x, y

def nonlinear_transformation(x, prob=0.5):
    if random.random() >= prob:
        return x

    points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
    xpoints = [p[0] for p in points]
    ypoints = [p[1] for p in points]
    xvals, yvals = bezier_curve(points, nTimes=100000)

    if random.random() < 0.5:
        # Half change to get flip
        xvals = np.sort(xvals)

    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)

    nonlinear_x = np.interp(x, xvals, yvals)
    return nonlinear_x

 

def local_pixel_shuffling(x, prob=0.5):
    if random.random() >= prob:
        return x

    image_temp = copy.deepcopy(x)
    orig_image = copy.deepcopy(x)
    img_rows, img_cols, img_deps = x.shape

    num_block = 500
    for _ in range(num_block):
        block_noise_size_x = random.randint(1, img_rows//10)
        block_noise_size_y = random.randint(1, img_cols//10)

        noise_x = random.randint(0, img_rows-block_noise_size_x)
        noise_y = random.randint(0, img_cols-block_noise_size_y)

        if img_deps >3 :
            filters = 3

            window = orig_image[noise_x:noise_x+block_noise_size_x, 
                               noise_y:noise_y+block_noise_size_y, 
                                :,
                           ]

            window = window.flatten()
            np.random.shuffle(window)
            window = window.reshape((block_noise_size_x, 
                                 block_noise_size_y, 
                                 img_deps))
                                 #block_noise_size_z))

            image_temp[noise_x:noise_x+block_noise_size_x, 
                      noise_y:noise_y+block_noise_size_y, 
                      :] = window

        else :
            window = orig_image[noise_x:noise_x+block_noise_size_x, 
                               noise_y:noise_y+block_noise_size_y, 
                                :,
                           ]

            window = window.flatten()
            np.random.shuffle(window)
            window = window.reshape((block_noise_size_x, 
                                 block_noise_size_y, 
                                 img_deps))

            image_temp[noise_x:noise_x+block_noise_size_x, 
                      noise_y:noise_y+block_noise_size_y, 
                       :] = window

    local_shuffling_x = image_temp
    return local_shuffling_x

 

def image_in_painting(x):
    img_rows, img_cols, img_deps = x.shape

    cnt = 5
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = random.randint(img_rows//6, img_rows//3)
        block_noise_size_y = random.randint(img_cols//6, img_cols//3)
        block_noise_size_z = random.randint(0,3)
        noise_x = random.randint(3, img_rows-block_noise_size_x-3)
        noise_y = random.randint(3, img_cols-block_noise_size_y-3)
        noise_z = random.randint(0, img_deps-block_noise_size_z)
        x_point = random.randint(noise_x, noise_x+block_noise_size_x)
        y_point = random.randint(noise_y, noise_y+block_noise_size_y)
        inpainting = np.zeros((block_noise_size_x,block_noise_size_y,3))
        
        for row in range(0,inpainting.shape[0]-1):
            for col in range(0,inpainting.shape[1]-1):
                inpainting[row,col] = x[x_point,y_point]
        x[
          noise_x:noise_x+block_noise_size_x, 
          noise_y:noise_y+block_noise_size_y, 
          :] = inpainting *1.0

        cnt -= 1
    return x

def image_out_painting(x):
    img_rows, img_cols, img_deps = x.shape
    image_temp = copy.deepcopy(x)
    
    x = np.ones((img_rows, img_cols, img_deps) ) * 1.0
    
    x_point = random.randint(0, img_rows-1)
    y_point = random.randint(0, img_cols-1)
    
    for row in range(0,x.shape[0]):
        for col in range(0,x.shape[1]):
            x[row,col] = image_temp[x_point,y_point]
    
    cnt = 4
    while cnt>0:
        block_noise_size_x = img_rows - random.randint(3*img_rows//7, 4*img_rows//7)
        block_noise_size_y = img_cols - random.randint(3*img_cols//7, 4*img_cols//7)
        block_noise_size_z = img_deps - random.randint(3*img_deps//7, 4*img_deps//7)
        noise_x = random.randint(3, img_rows-block_noise_size_x-3)
        noise_y = random.randint(3, img_cols-block_noise_size_y-3)
        noise_z = random.randint(0, img_deps-block_noise_size_z)

        x[
          noise_x:noise_x+block_noise_size_x, 
          noise_y:noise_y+block_noise_size_y, 
          :] = image_temp[ noise_x:noise_x+block_noise_size_x, 
          noise_y:noise_y+block_noise_size_y, :] * 1.0
        cnt -= 1
    return x


def generate_pair(img):
    img = np.array(img)
    img_rows, img_cols, img_deps = img.shape[0], img.shape[1], img.shape[2]

    y = img
#    print('y', y.shape)
    x = copy.deepcopy(y)   
 #   print('deep', x.shape)
    # Local Shuffle Pixel
    x = local_pixel_shuffling(x, prob=configs.local_rate)

    # Apply non-Linear transformation with an assigned probability
    x = nonlinear_transformation(x, configs.nonlinear_rate)

    # Inpainting & Outpainting
    if random.random() < configs.paint_rate:
        if random.random() < configs.inpaint_rate:
            # Inpainting
            x = image_in_painting(x)
        else:
            # Outpainting
            x = image_out_painting(x)
  #  print('ret', x.shape)
    return x #img, np.array(x)
    
@tf.function
def tf_generate_pair(input):
  y = tf.numpy_function(generate_pair, [input], tf.float32)
  return y

def CBAM_attention(x, ratio=8, kernel_size=7, dr_ratio=0.2, activ_regularization=0.0001, kernel_initializer = tf.keras.initializers.VarianceScaling()):
      channel = x.get_shape()[-1]

      ##channel attention##
      avg_pool = tf.reduce_mean(x, axis=[1,2], keepdims=True)
      avg_pool = Dense(units = channel//ratio ,activation='relu', activity_regularizer=tf.keras.regularizers.l1(activ_regularization),kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5), use_bias=True,bias_initializer='zeros',trainable=True)(avg_pool)
      avg_pool = Dense(channel, activation = 'relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5),activity_regularizer=tf.keras.regularizers.l1(activ_regularization), use_bias=True, bias_initializer='zeros',trainable=True)(avg_pool)

      max_pool = tf.reduce_max(x, axis=[1,2], keepdims=True)
      max_pool = Dense(units = channel//ratio, activation='relu', activity_regularizer=tf.keras.regularizers.l1(activ_regularization),kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5), use_bias=True,bias_initializer='zeros',trainable=True)(max_pool)
      max_pool = Dense(channel, activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5), activity_regularizer=tf.keras.regularizers.l1(activ_regularization),use_bias=True, bias_initializer='zeros',trainable=True)(max_pool)
      f = Add()([avg_pool, max_pool])
      f = Activation('sigmoid')(f)

      after_channel_att = multiply([x, f])

      ##spatial attention##
      kernel_size = kernel_size
      avg_pool_2 = tf.reduce_mean(x, axis=[1,2], keepdims=True)
      max_pool_2 = tf.reduce_max(x, axis=[1,2], keepdims=True)
      concat = tf.concat([avg_pool,max_pool],3)
      concat = Conv2D(filters=1, kernel_size=[kernel_size,kernel_size], strides=[1,1], padding='same', kernel_initializer=kernel_initializer,use_bias=False)(concat)
      concat = Activation('sigmoid')(concat)
      ##final_cbam##
      attention_feature = multiply([x,concat])
      return attention_feature


class SSL_AM(Model):
    def __init__(self, pretrained=None, supervised=False):
        super(SSL_AM, self).__init__()
        self.pretrained = pretrained
        self.model_name = 'ssl_am'
        self.model_type = 'keras'
        self.supervised = supervised
        self.input_size = (480,480,3)
        self.optimizer = Adam() #Adam(clipvalue=2, clipnorm=1)
        if self.supervised:
            self.loss_fn = losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
            assert self.pretrained
            self.lr = 1e-5
        else:
            self.loss_fn = loss_fn() #MeanSquaredError()
            self.lr =1e-4
        
    def unet_cbam(self, x):
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        conv1 = CBAM_attention(conv1)

        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        conv2 = CBAM_attention(conv2)

        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        conv3 = CBAM_attention(conv3)

        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        conv4 = CBAM_attention(conv4)

        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5, name='dropout_1')(conv5)
        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))

        merge6 = concatenate([drop4, up6], axis=3)

        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        conv6 = CBAM_attention(conv6)
        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))

        merge7 = concatenate([conv3, up7], axis=3)

        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        conv7 = CBAM_attention(conv7)
        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))

        merge8 = concatenate([conv2, up8], axis=3)

        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))

        merge9 = concatenate([conv1, up9], axis=3)

        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation='relu')(conv9)

        final = UpSampling3D(size=(1, 1, 3))(conv10)
        model = Model(inputs=x, outputs=final)

        if (self.pretrained and type(self.pretrained)!=bool):
            print('loading weights ...')
            model.load_weights(self.pretrained)

        return model
    
    def classifier(self, activ_regularization=1e-5):
        input = Input(self.input_size)

        unet_cbam_model = self.unet_cbam(input)
        unet_cbam_classification = Model(inputs = unet_cbam_model.input, outputs= unet_cbam_model.get_layer('dropout_1').output)

        clf = tf.keras.Sequential()
        clf.add(unet_cbam_classification)
        clf.add(Conv2D(512,(2,2), activation = 'relu' ,padding = 'same')) # ,  kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)))
        clf.add(Conv2D(256,(2,2), activation = 'relu' , padding = 'same')) #,  kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)))
        clf.add(BatchNormalization())
        clf.add(Conv2D(128,(2,2), activation = 'relu' , padding = 'same')) #,  kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)))
        clf.add(Conv2D(64,(2,2), activation = 'relu' , padding = 'same')) # ,  kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)))
        clf.add(BatchNormalization())
        #clf.add(MaxPooling2D(pool_size=(2,2)))
        clf.add(GlobalAveragePooling2D())
      #  clf.add(BatchNormalization())

        clf.add(Flatten())
        #clf.add(Dense(512, activation = 'relu')) #, activity_regularizer=tf.keras.regularizers.l2(activ_regularization))) # ,kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)))
        clf.add(Dense(256, activation = 'relu')) #, activity_regularizer=tf.keras.regularizers.l2(activ_regularization))) #,kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)))
        clf.add(Dense(128, activation = 'relu')) #, activity_regularizer=tf.keras.regularizers.l2(activ_regularization))) # ,kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)))
        clf.add(Dropout(0.3))
        clf.add(Dense(64,  activation='relu')) #,activity_regularizer=tf.keras.regularizers.l2(activ_regularization))) # ,  kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)))
        #clf.add(Dense(32, activation='relu')) # ,activity_regularizer=tf.keras.regularizers.l2(activ_regularization)))
        clf.add(Dropout(0.3))
        clf.add(Dense(32, activation='relu' ))
        #clf.add(Dense(8, activation = 'relu'))
        clf.add(Dense(1, activation='sigmoid'))

#        clf.layers[0].trainable = False            
            
        return clf

    def build_model(self):
        input = Input(self.input_size)
        if self.supervised:
            assert self.pretrained
            model = self.classifier()
        else:
            model = self.unet_cbam(input)
            
        return {'model':model, 'optimizer':self.optimizer, 'loss_fn':self.loss_fn, 'lr':self.lr,
        'model_name':self.model_name, 'model_type':self.model_type, 'supervised':self.supervised}
        
if __name__ == "__main__":
    model = SSL_AM(supervised=True, pretrained="/MULTIX/DATA/nccid/ssl_am_chexpert_unsupervised_1.h5").build_model()
    print(model['model'].summary())
