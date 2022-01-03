from __future__ import print_function
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Input
from tensorflow.keras import activations
from tensorflow.keras import utils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
import tensorflow as tf

K.set_image_data_format('channels_last')

def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (1 + s_squared_norm)
    return scale * x

def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)

def margin_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    lamb, margin = 0.5, 0.1
    return K.sum((y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (
        1 - y_true) * K.square(K.relu(y_pred - margin))), axis=-1)

class Capsule(Layer):
   
    def __init__(self,
                 num_capsule,
                 dim_capsule,
                 routings=3,
                 share_weights=True,
                 activation='squash',
                 **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)
            
    def get_config(self):
        config = super().get_config().copy()
        config.update({
        'num_capsule':  self.num_capsule,
        'dim_capsule' : self.dim_capsule,
        'routings':  self.routings,
        'share_weight':self.share_weights,
      
        })
        return config

    def build(self, input_shape):
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(1, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(input_num_capsule, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)

    def call(self, inputs):
        
        if self.share_weights:
            hat_inputs = K.conv1d(inputs, self.kernel)
        else:
            hat_inputs = K.local_conv1d(inputs, self.kernel, [1], [1])

        batch_size = K.shape(inputs)[0]
        input_num_capsule = K.shape(inputs)[1]
        hat_inputs = K.reshape(hat_inputs,
                               (batch_size, input_num_capsule,
                                self.num_capsule, self.dim_capsule))
        hat_inputs = K.permute_dimensions(hat_inputs, (0, 2, 1, 3))

        b = K.zeros_like(hat_inputs[:, :, :, 0])
        for i in range(self.routings):
            c = softmax(b, 1)
            o = self.activation(keras.backend.batch_dot(c, hat_inputs, [2, 2]))
            if i < self.routings - 1:
                b = keras.backend.batch_dot(o, hat_inputs, [2, 3])
                if K.backend() == 'theano':
                    o = K.sum(o, axis=1)

        return o

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

class CovidCaps(Model):
    def __init__(self, optimizer, loss) -> None:
        super().__init__()
        self.shape = 480
        self.optimizer = optimizers.Adam()
        self.loss = margin_loss
        self.model_type = 'keras'

        self.reshape = Reshape((-1,128))

        self.conv1 = Conv2D(64, (3, 3), activation='relu')
        self.conv2 = Conv2D(128, (3, 3), activation='relu')
        self.batchnorm = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
        self.avgpool = AveragePooling2D((2, 2))
        self.dropout = Dropout(0.5)
        self.capsule1 = Capsule(32, 8, 3, True)
        self.capsule2 = Capsule(2, 16, 3, True)
        self.lam = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))

    def forward(self, input):
      #  input_image = Input(shape=(None, None, 3))
        x = self.conv1(input)
        x = self.batchnorm(x)
        x = self.conv1(x)
        x = self.avgpool(x)
        x = self.conv2(x)
        x = self.conv2(x)

        x = self.reshape(x)
        x = self.capsule1(x)
        x = self.capsule2(x)

        output = self.lam(x)

        return output
    
    def get_loss(self):
        return self.loss
    
    def get_optimizer(self):
        return self.optimizer

    def get_model_type(self):
        return self.model_type

    def build_graph(self, input_shape):
        x = Input(shape=(None, input_shape), ragged=True)
        return Model(inputs=[x], outputs=self.forward(x))