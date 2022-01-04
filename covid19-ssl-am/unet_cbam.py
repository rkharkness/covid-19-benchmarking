from detect_covid19_v100 import ssim_loss
import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1
from tensorflow.keras.models import *

class UnetCBAM(Model):
    def __init__(self, kernel_size=3, ratio=3, activ_regularization=0.01) -> None:
        super().__init__()
        self.input_shape=(480,480,3)
        self.kernel_size = kernel_size
        self.ratio = ratio

        self.dropout = Dropout(0.5)
        self.pool = MaxPooling2D(pool_size=(2,2))

        self.conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal'
        self.conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        
        self.upsample = Upsampling2D(size=(2,2))
        self.upsample3d = UpSampling3D(size=(1, 1, 3))

        self.conv4_up = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv3_up = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv2_up = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv1_up = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        
        self.conv0 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv = Conv2D(1, 1, activation='relu')
    
    def CBAM_attention(self, inputs):
        x = inputs
        channel = x.get_shape()[-1]

        ##channel attention##
        avg_pool = tf.reduce_mean(x, axis=[1,2], keepdims=True)
        avg_pool = Dense(units = channel//self.ratio ,activation='relu', activity_regularizer=tf.keras.regularizers.l1(activ_regularization),kernel_initializer='he_normal', use_bias=True,bias_initializer='zeros',trainable=True)(avg_pool)
        avg_pool = Dense(channel, kernel_initializer='he_normal',activity_regularizer=tf.keras.regularizers.l1(activ_regularization), use_bias=True, bias_initializer='zeros',trainable=True)(avg_pool)

        max_pool = tf.reduce_max(x, axis=[1,2], keepdims=True)
        max_pool = Dense(units = channel//self.ratio, activation='relu', activity_regularizer=tf.keras.regularizers.l1(activ_regularization),kernel_initializer='he_normal', use_bias=True,bias_initializer='zeros',trainable=True)(max_pool)
        max_pool = Dense(channel, activation='relu', kernel_initializer='he_normal', activity_regularizer=tf.keras.regularizers.l1(activ_regularization),use_bias=True, bias_initializer='zeros',trainable=True)(max_pool)
        f = Add()([avg_pool, max_pool])
        f = Activation('sigmoid')(f)

        after_channel_att = multiply([x, f])

        ##spatial attention##
        avg_pool_2 = tf.reduce_mean(x, axis=[1,2], keepdims=True) #? no spatial attn - check paper
        max_pool_2 = tf.reduce_max(x, axis=[1,2], keepdims=True) #?

        concat = tf.concat([avg_pool,max_pool],3)
        concat = Conv2D(filters=1, kernel_size=[self.kernel_size,self.kernel_size], strides=[1,1], padding='same', kernel_initializer=kernel_initializer,use_bias=False)(concat)
        concat = Activation('sigmoid')(concat)

        ##final_cbam##
        attention_feature = multiply([x,concat])
        return attention_feature        

    def call(self, x):
        x1 = self.conv1(x)
        x1 = self.conv1(x1)
        x1 = self.CBAM_attention(x1)
        pool1 = self.pool(x1)

        x2 = self.conv2(pool1)
        x2 = self.conv2(x2)
        x2 = self.CBAM_attention(x2)
        pool2 = self.pool(x2)

        x3 = self.conv3(pool2)
        x3 = self.conv3(x3)
        x3 = self.CBAM_attention(x3)
        pool3 = self.pool(x3)

        x4 = self.conv4(pool3)
        x4 = self.conv4(x4)
        x4 = self.CBAM_attention(x4)

        drop_4 = self.dropout(x4)
        x4 = self.pool(drop_4)

        x5 = self.conv5(x4)
        x5 = self.conv5(x5)

        x5 = self.dropout(x5)

        x6 = self.upsample(x5)
        x6 = self.conv4_up(x6)
        x6 = concatenate([drop_4, x6], axis=3)

        x6 = self.conv4(x6)
        x6 = self.conv4(x6)
        x6 = self.CBAM_attention(x6)

        x7 = self.upsample(x6)
        x7 = self.conv3_up(x7)

        x7 = concatenate([conv3, x7], axis=3)
        x7 = self.conv3(x7)
        x7 = self.conv3(x7)
        x7 = self.CBAM_attention(x7)

        x8 = self.upsample(x7)
        x8 = self.conv2_up(x8)
        x8 = concatenate([x2, x8], axis=3)
        x8 = self.conv1(x8)
        x8 = self.conv1(x8)
        
        x9 = self.upsample(x8)
        x9 = self.conv1_up(x9)
        x9 = concatenate([conv1, x9], axis=3)
        x9 = self.conv1(x9)
        x9 = self.conv1(x9)

        x9 = self.conv0(x9)
        x10 = self.conv(x10)
        output = self.upsample3d(x10)

        return output

    def get_optimizer():
        @staticmethod
        return tf.keras.optimizers.Adam()

    def get_loss_fn():
        @staticmethod
        return ssim_loss # unsupervised, - supervised loss?

# adapt into main training function
#def pretraining(nih):
    # ssl
    

    






