import skimage.io as io
import skimage.transform as trans

import numpy as np
import copy

from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras

from modelsgenesis_pytorch import models_genesis_config

import os
import shutil
import random

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_memory_growth(gpus[1], True)
    except RuntimeError as e:
        print(e)


def unet(pretrained_weights=None, input_size=(480, 480, 3)):
    #mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
    #with mirrored_strategy.scope():
        inputs = tf.keras.Input(input_size)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

        conv10 = Conv2D(1, 1, activation='relu')(conv9)
        final = UpSampling3D(size=(1, 1, 3))(conv10)

        model = Model(inputs=inputs, outputs=final)

        # model.summary()

        if (pretrained_weights):
            model.load_weights(pretrained_weights)

        return model

def unet_cbam(pretrained_weights=None, input_size=(480, 480, 3), kernel_size=3, ratio=3, activ_regularization=0.01):
    inputs = tf.keras.Input(input_size)

    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)

    ##channel attention##
    avg_pool = GlobalAveragePooling2D()(conv1)

    avg_pool = Dense(avg_pool.shape[-1], activation='relu',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization),
                     kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros', trainable=True)(avg_pool)

    avg_pool = Dense(avg_pool.shape[-1], kernel_initializer='he_normal',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization), use_bias=True,
                     bias_initializer='zeros', trainable=True)(avg_pool)

    max_pool = GlobalMaxPooling2D()(conv1)

    max_pool = Dense(max_pool.shape[-1], activation='relu',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization),
                     kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros', trainable=True)(max_pool)

    max_pool = Dense(max_pool.shape[-1], activation='relu', kernel_initializer='he_normal',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization), use_bias=True,
                     bias_initializer='zeros', trainable=True)(max_pool)

    f = Add()([avg_pool, max_pool])

    f = Activation('relu')(f)

    after_channel_att = multiply([conv1, f])

    ##spatial attention##
    kernel_size = kernel_size

    avg_pool_2 = Lambda(lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True))(after_channel_att)

    max_pool_2 = Lambda(lambda x: tf.keras.backend.max(x, axis=3, keepdims=True))(after_channel_att)

    concat = Concatenate(axis=3)([avg_pool_2, max_pool_2])

    cbam_feature = Conv2D(1, (kernel_size, kernel_size), strides=(1, 1), padding='same', activation='relu',
                          activity_regularizer=tf.keras.regularizers.l2(activ_regularization), use_bias=True,
                          kernel_initializer='he_normal', bias_initializer='zeros', trainable=True)(concat)

    cbam_feature = Dropout(0.2)(cbam_feature)

    ##final_cbam##

    after_spatial_att = multiply([conv1, cbam_feature])

    pool1 = MaxPooling2D(pool_size=(2, 2))(after_spatial_att)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)

    ##channel attention##

    avg_pool = GlobalAveragePooling2D()(conv2)

    avg_pool = Dense(avg_pool.shape[-1], activation='relu',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization),
                     kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros', trainable=True)(avg_pool)

    avg_pool = Dense(avg_pool.shape[-1], kernel_initializer='he_normal',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization), use_bias=True,
                     bias_initializer='zeros', trainable=True)(avg_pool)

    max_pool = GlobalMaxPooling2D()(conv2)

    max_pool = Dense(max_pool.shape[-1], activation='relu',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization),
                     kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros', trainable=True)(max_pool)

    max_pool = Dense(max_pool.shape[-1], activation='relu', kernel_initializer='he_normal',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization), use_bias=True,
                     bias_initializer='zeros', trainable=True)(max_pool)

    f = Add()([avg_pool, max_pool])

    f = Activation('relu')(f)

    after_channel_att = multiply([conv2, f])

    ##spatial attention##
    kernel_size = kernel_size
    avg_pool_2 = Lambda(lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True))(after_channel_att)
    max_pool_2 = Lambda(lambda x: tf.keras.backend.max(x, axis=3, keepdims=True))(after_channel_att)

    concat = Concatenate(axis=3)([avg_pool_2, max_pool_2])
    cbam_feature = Conv2D(1, (kernel_size, kernel_size), strides=(1, 1), padding='same', activation='relu',
                          activity_regularizer=tf.keras.regularizers.l2(activ_regularization), use_bias=True,
                          kernel_initializer='he_normal', bias_initializer='zeros', trainable=True)(concat)

    cbam_feature = Dropout(0.2)(cbam_feature)

    ##final_cbam##
    after_spatial_att_2 = multiply([conv2, cbam_feature])

    pool2 = MaxPooling2D(pool_size=(2, 2))(after_spatial_att_2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)

    ##channel attention##
    avg_pool = GlobalAveragePooling2D()(conv3)

    avg_pool = Dense(avg_pool.shape[-1], activation='relu',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization),
                     kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros', trainable=True)(avg_pool)

    avg_pool = Dense(avg_pool.shape[-1], kernel_initializer='he_normal',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization), use_bias=True,
                     bias_initializer='zeros', trainable=True)(avg_pool)

    max_pool = GlobalMaxPooling2D()(conv3)

    max_pool = Dense(max_pool.shape[-1], activation='relu',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization),
                     kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros', trainable=True)(max_pool)

    max_pool = Dense(max_pool.shape[-1], activation='relu', kernel_initializer='he_normal',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization), use_bias=True,
                     bias_initializer='zeros', trainable=True)(max_pool)

    f = Add()([avg_pool, max_pool])

    f = Activation('relu')(f)

    after_channel_att = multiply([conv3, f])

    ##spatial attention##
    kernel_size = kernel_size

    avg_pool_2 = Lambda(lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True))(after_channel_att)

    max_pool_2 = Lambda(lambda x: tf.keras.backend.max(x, axis=3, keepdims=True))(after_channel_att)

    concat = Concatenate(axis=3)([avg_pool_2, max_pool_2])

    cbam_feature = Conv2D(1, (kernel_size, kernel_size), strides=(1, 1), padding='same', activation='relu',
                          activity_regularizer=tf.keras.regularizers.l2(activ_regularization), use_bias=True,
                          kernel_initializer='he_normal', bias_initializer='zeros', trainable=True)(concat)

    cbam_feature = Dropout(0.2)(cbam_feature)

    ##final_cbam##
    after_spatial_att_3 = multiply([conv3, cbam_feature])

    pool3 = MaxPooling2D(pool_size=(2, 2))(after_spatial_att_3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)

    ##channel attention##
    avg_pool = GlobalAveragePooling2D()(conv4)

    avg_pool = Dense(avg_pool.shape[-1], activation='relu',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization),
                     kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros', trainable=True)(avg_pool)

    avg_pool = Dense(avg_pool.shape[-1], kernel_initializer='he_normal',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization), use_bias=True,
                     bias_initializer='zeros', trainable=True)(avg_pool)

    max_pool = GlobalMaxPooling2D()(conv4)

    max_pool = Dense(max_pool.shape[-1], activation='relu',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization),
                     kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros', trainable=True)(max_pool)

    max_pool = Dense(max_pool.shape[-1], activation='relu', kernel_initializer='he_normal',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization), use_bias=True,
                     bias_initializer='zeros', trainable=True)(max_pool)

    f = Add()([avg_pool, max_pool])

    f = Activation('relu')(f)

    after_channel_att = multiply([conv4, f])

    ##spatial attention##
    kernel_size = kernel_size

    avg_pool_2 = Lambda(lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True))(after_channel_att)

    max_pool_2 = Lambda(lambda x: tf.keras.backend.max(x, axis=3, keepdims=True))(after_channel_att)

    concat = Concatenate(axis=3)([avg_pool_2, max_pool_2])

    cbam_feature = Conv2D(1, (kernel_size, kernel_size), strides=(1, 1), padding='same', activation='relu',
                          activity_regularizer=tf.keras.regularizers.l2(activ_regularization), use_bias=True,
                          kernel_initializer='he_normal', bias_initializer='zeros', trainable=True)(concat)

    cbam_feature = Dropout(0.2)(cbam_feature)

    ##final_cbam##
    after_spatial_att_4 = multiply([conv4, cbam_feature])

    drop4 = Dropout(0.5)(after_spatial_att_4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    ##channel attention##
    avg_pool = GlobalAveragePooling2D()(conv5)

    avg_pool = Dense(avg_pool.shape[-1], activation='relu',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization),
                     kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros', trainable=True)(avg_pool)
    avg_pool = Dense(avg_pool.shape[-1], kernel_initializer='he_normal',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization), use_bias=True,
                     bias_initializer='zeros', trainable=True)(avg_pool)

    max_pool = GlobalMaxPooling2D()(conv5)
    max_pool = Dense(max_pool.shape[-1], activation='relu',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization),
                     kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros', trainable=True)(max_pool)

    max_pool = Dense(max_pool.shape[-1], activation='relu', kernel_initializer='he_normal',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization), use_bias=True,
                     bias_initializer='zeros', trainable=True)(max_pool)

    f = Add()([avg_pool, max_pool])

    f = Activation('relu')(f)

    after_channel_att = multiply([conv5, f])

    ##spatial attention##
    kernel_size = kernel_size

    avg_pool_2 = Lambda(lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True))(after_channel_att)

    max_pool_2 = Lambda(lambda x: tf.keras.backend.max(x, axis=3, keepdims=True))(after_channel_att)

    concat = Concatenate(axis=3)([avg_pool_2, max_pool_2])

    cbam_feature = Conv2D(1, (kernel_size, kernel_size), strides=(1, 1), padding='same', activation='relu',
                          activity_regularizer=tf.keras.regularizers.l2(activ_regularization), use_bias=True,
                          kernel_initializer='he_normal', bias_initializer='zeros', trainable=True)(concat)

    cbam_feature = Dropout(0.2)(cbam_feature)

    ##final_cbam##
    after_spatial_att_5 = multiply([conv5, cbam_feature])

    drop5 = Dropout(0.5)(after_spatial_att_5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))

    merge6 = concatenate([drop4, up6], axis=3)

    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)

    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    ##channel attention##
    avg_pool = GlobalAveragePooling2D()(conv6)

    avg_pool = Dense(avg_pool.shape[-1], activation='relu',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization),
                     kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros', trainable=True)(avg_pool)

    avg_pool = Dense(avg_pool.shape[-1], kernel_initializer='he_normal',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization), use_bias=True,
                     bias_initializer='zeros', trainable=True)(avg_pool)

    max_pool = GlobalMaxPooling2D()(conv6)

    max_pool = Dense(max_pool.shape[-1], activation='relu',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization),
                     kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros', trainable=True)(max_pool)

    max_pool = Dense(max_pool.shape[-1], activation='relu', kernel_initializer='he_normal',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization), use_bias=True,
                     bias_initializer='zeros', trainable=True)(max_pool)

    f = Add()([avg_pool, max_pool])

    f = Activation('relu')(f)

    after_channel_att = multiply([conv6, f])

    ##spatial attention##
    kernel_size = kernel_size

    avg_pool_2 = Lambda(lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True))(after_channel_att)

    max_pool_2 = Lambda(lambda x: tf.keras.backend.max(x, axis=3, keepdims=True))(after_channel_att)

    concat = Concatenate(axis=3)([avg_pool_2, max_pool_2])

    cbam_feature = Conv2D(1, (kernel_size, kernel_size), strides=(1, 1), padding='same', activation='relu',
                          activity_regularizer=tf.keras.regularizers.l2(activ_regularization), use_bias=True,
                          kernel_initializer='he_normal', bias_initializer='zeros', trainable=True)(concat)

    cbam_feature = Dropout(0.2)(cbam_feature)

    ##final_cbam##
    after_spatial_att_6 = multiply([conv6, cbam_feature])

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(after_spatial_att_6))

    merge7 = concatenate([conv3, up7], axis=3)

    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)

    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

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

    model = Model(inputs=inputs, outputs=final)

    # model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def CBAM_attention(inputs,ratio,kernel_size,dr_ratio,activ_regularization):
    x = inputs
    channel = x.get_shape()[-1]

    ##channel attention##
    avg_pool = tf.reduce_mean(x, axis=[1,2], keepdims=True)
    avg_pool = Dense(units = channel//ratio ,activation='relu', activity_regularizer=tf.keras.regularizers.l1(activ_regularization),kernel_initializer='he_normal', use_bias=True,bias_initializer='zeros',trainable=True)(avg_pool)
    avg_pool = Dense(channel, kernel_initializer='he_normal',activity_regularizer=tf.keras.regularizers.l1(activ_regularization), use_bias=True, bias_initializer='zeros',trainable=True)(avg_pool)

    max_pool = tf.reduce_max(x, axis=[1,2], keepdims=True)
    max_pool = Dense(units = channel//ratio, activation='relu', activity_regularizer=tf.keras.regularizers.l1(activ_regularization),kernel_initializer='he_normal', use_bias=True,bias_initializer='zeros',trainable=True)(max_pool)
    max_pool = Dense(channel, activation='relu', kernel_initializer='he_normal', activity_regularizer=tf.keras.regularizers.l1(activ_regularization),use_bias=True, bias_initializer='zeros',trainable=True)(max_pool)
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

from skimage.transform import resize

try:  # SciPy >= 0.19
    from scipy.special import comb

except ImportError:

    from scipy.misc import comb


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

        #block_noise_size_z = random.randint(0, 3)

        noise_x = random.randint(0, img_rows-block_noise_size_x)
        noise_y = random.randint(0, img_cols-block_noise_size_y)

        #noise_z = random.randint(0, img_deps-block_noise_size_z)

        if img_deps >3 :
            filters = 3
            window = orig_image[noise_x:noise_x+block_noise_size_x, 
                               noise_y:noise_y+block_noise_size_y, 
                               #noise_z:noise_z+block_noise_size_z,
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
                      #noise_z:noise_z+block_noise_size_z] = window

        else :
            window = orig_image[noise_x:noise_x+block_noise_size_x, 
                               noise_y:noise_y+block_noise_size_y, 
                               #noise_z:noise_z+block_noise_size_z,
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
                      #noise_z:noise_z+block_noise_size_z] = window
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
        #noise_z = random.randint(0, img_deps-block_noise_size_z)
        
        x_point = random.randint(noise_x, noise_x+block_noise_size_x)
        y_point = random.randint(noise_y, noise_y+block_noise_size_y)
        
        inpainting = np.zeros((block_noise_size_x,block_noise_size_y,3))
        
        for row in range(0,inpainting.shape[0]-1):
            for col in range(0,inpainting.shape[1]-1):
                inpainting[row,col] = x[x_point,y_point]

        x[noise_x:noise_x+block_noise_size_x, 
          noise_y:noise_y+block_noise_size_y, 
          :] = inpainting*1.0

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
       # block_noise_size_z = img_deps - random.randint(3*img_deps//7, 4*img_deps//7)

        noise_x = random.randint(3, img_rows-block_noise_size_x-3)
        noise_y = random.randint(3, img_cols-block_noise_size_y-3)

       # noise_z = random.randint(0, img_deps-block_noise_size_z)

        x[noise_x:noise_x+block_noise_size_x, 
          noise_y:noise_y+block_noise_size_y, 
          :] = image_temp[ noise_x:noise_x+block_noise_size_x, noise_y:noise_y+block_noise_size_y, :] * 1.0
        cnt -= 1

         #noise_z:noise_z+block_noise_size_z]

    return x

def generate_pair(path):
    config = models_genesis_config

    img = Image.open(path)
    img = img.convert('RGB')
    img = img.resize((480,480))

    #img= tf.keras.preprocessing.image.img_to_array(img)

    img = np.array(img)

    img_rows, img_cols, img_deps = img.shape[0], img.shape[1], img.shape[2]

    while True:

        y = img/255
        x = copy.deepcopy(y)            

        # Autoencoder
        x = copy.deepcopy(y) 

        # Flip
        x, y = data_augmentation(x, y, config.flip_rate)

        # Local Shuffle Pixel
        x = local_pixel_shuffling(x, prob=config.local_rate)

        # Apply non-Linear transformation with an assigned probability
        x = nonlinear_transformation(x, config.nonlinear_rate)

        # Inpainting & Outpainting
        if random.random() < config.paint_rate:
            if random.random() < config.inpaint_rate:
                # Inpainting
                x = image_in_painting(x)
            else:
                # Outpainting
                x = image_out_painting(x)

        return x, y

# loss functions
def ssim_loss_minusone(y_true, y_pred):
    return 1-tf.reduce_mean(tf.image.ssim(tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred), 2.0))

def ssim_loss(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred), 2.0))


ratio=8
activ_regularization=0.1
kernel_size=7
kernel_initializer = tf.keras.initializers.VarianceScaling()
dr_ratio=0.2
targetdir = '/home/ubuntu/Desktop/data/pjh/images/images'

checkpoint_path1 = './pretrained_weights_nih'
checkpoint_dir1 = os.path.join(os.getcwd()+checkpoint_path1)
checkpoint_path2 = './pretrained_weights'
checkpoint_dir2 = os.path.join(os.getcwd()+checkpoint_path2)
cp_callback1 = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path1,
                                                save_weights_only=True,
                                                monitor='val_loss',
                                                save_best_only=True,
                                                mode='min')
cp_callback2 = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path2,
                                                save_weights_only=True,
                                                monitor='val_accuracy',
                                                save_best_only=True,
                                                mode='max')




print(unet_cbam_model.summary())

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    unet_cbam_model = unet_cbam(pretrained_weights = 'nih_unet_cbam_20.h5', input_size=(512,512,3),kernel_size=3, ratio=3, activ_regularization=0.01 )
    unet_cbam_model.compile(optimizer=tf.keras.optimizers.Adam(lr = 1e-4),
            loss='mse',
            metrics=['mse',ssim_loss])

history = unet_cbam_model.fit(train_dataset,
                            validation_data=valid_dataset, 
                            validation_steps=len(onlyfiles2)//8,
                            steps_per_epoch=len(onlyfiles1)//8, 
                            epochs=8,
                            #max_queue_size=configs.models_genesis_config.max_queue_size, 
                            #workers=configs.models_genesis_config.workers, 
                            use_multiprocessing=True, 
                            shuffle=True,
                            verbose=configs.models_genesis_config.verbose,
                            callbacks=[cp_callback1]
                           )

unet_cbam_model.save_weights('nih_unet_cbam_covid_classification.h5')


unet = unet(input_size=(480, 480, 3))
unet = unet(input_size=(480, 480, 3))

unet_cbam_model = unet_cbam(input_size=(480, 480, 3),
                            kernel_size=3, ratio=3, activ_regularization=0.01)
unet_cbam_model = unet_cbam(input_size=(480, 480, 3),
                            kernel_size=3, ratio=3, activ_regularization=0.01)

for layer in unet.layers:
    print(layer.name)

unet_cbam_model.summary()
unet.summary()

grad_model = tf.keras.models.Model([unet.inputs], [unet.get_layer('up_sampling3d').output, unet.output])