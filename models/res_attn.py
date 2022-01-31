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
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Multiply
from tensorflow.keras.layers import Lambda
from tensorflow.keras import Model

from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

import tensorflow as tf

from tensorflow.keras import backend as K
import tensorflow as keras
from tensorflow.keras.regularizers import l2

#K.set_image_data_format('channels_first')

def residual_block(input, input_channels=None, output_channels=None, kernel_size=(3, 3), stride=1):
    if output_channels is None:
        output_channels = input.get_shape()[-1]
    if input_channels is None:
        input_channels = output_channels // 4

    strides = (stride, stride)

    x = Activation('relu')(input)
    x = BatchNormalization()(x)
    x = Conv2D(input_channels, (1, 1))(x)

    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(input_channels, kernel_size, padding='same', strides=stride)(x)

    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(output_channels, (1, 1), padding='same')(x)

    if input_channels != output_channels or stride != 1:
        input = Conv2D(output_channels, (1, 1), padding='same', strides=strides)(input)

    x = Add()([x, input])
    return x


def attention_block(input, input_channels=None, output_channels=None, encoder_depth=1):

    p = 1
    t = 2
    r = 1

    if input_channels is None:
        input_channels = input.get_shape()[-1]
    if output_channels is None:
        output_channels = input_channels

    # First Residual Block
    for i in range(p):
        input = residual_block(input)

    # Trunc Branch
    output_trunk = input
    for i in range(t):
        output_trunk = residual_block(output_trunk)


    # Soft Mask Branch
    ## encoder
    ### first down sampling
    output_soft_mask = MaxPool2D(padding='same')(input)  # 32x32
    for i in range(r):
        output_soft_mask = residual_block(output_soft_mask)

    skip_connections = []
    for i in range(encoder_depth - 1):

        ## skip connections
        output_skip_connection = residual_block(output_soft_mask)
        skip_connections.append(output_skip_connection)
        # print ('skip shape:', output_skip_connection.get_shape())

        ## down sampling
        output_soft_mask = MaxPool2D(padding='same')(output_soft_mask)
        for _ in range(r):
            output_soft_mask = residual_block(output_soft_mask)


    ## decoder
    skip_connections = list(reversed(skip_connections))
    for i in range(encoder_depth - 1):
        ## upsampling666
        for _ in range(r):
            output_soft_mask = residual_block(output_soft_mask)
        output_soft_mask = UpSampling2D()(output_soft_mask)
        ## skip connections
        output_soft_mask = Add()([output_soft_mask, skip_connections[i]])

    ### last upsampling
    for i in range(r):
        output_soft_mask = residual_block(output_soft_mask)
    output_soft_mask = UpSampling2D()(output_soft_mask)

    ## Output
    output_soft_mask = Conv2D(input_channels, (1, 1))(output_soft_mask)
    output_soft_mask = Conv2D(input_channels, (1, 1))(output_soft_mask)
    output_soft_mask = Activation('sigmoid')(output_soft_mask)

    # Attention: (1 + output_soft_mask) * output_trunk
    output = Lambda(lambda x: x + 1)(output_soft_mask)
    output = Multiply()([output, output_trunk])  #

    # Last Residual Block
    for i in range(p):
        output = residual_block(output)

    return output

class WeightedBCE(keras.losses.Loss):
  def __init__(self, trial=None):
    """adapted from: https://stackoverflow.com/questions/46009619/keras-weighted-binary-crossentropy"""            
    super().__init__()
    self.trial = trial
    self.weights = {'0':1.28105274, '1':0.82008082}
   # self.weights = {'0':1.0, '1':1.0}

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

class AttentionResNetModified(Model):

    def __init__(self, shape=(480,480,3), n_channels=64, n_classes=1,
                      dropout=0.5, regularization=0.01, dropout_act=True, supervised=True):
      super(AttentionResNetModified, self).__init__()

      self.shape = shape
      self.n_channels = n_channels
      self.n_classes = n_classes
      self.dropout = dropout
      self.regularization = regularization
      self.dropout_act = dropout_act

      self.supervised = supervised

      self.optimizer = Adam()
      self.loss_fn = WeightedBCE()
      self.model_name = 'res_attn'
      self.model_type = 'keras'

      self.model = self.build_model()
    
    def build_model(self):

      input_ = Input(shape=self.shape)

      x = Conv2D(self.n_channels/2, (7,7), strides=(2,2))(input_)
      x = BatchNormalization()(x)
      x = Activation('relu')(x)

      x = Conv2D(self.n_channels,(7,7), strides=(2,2))(x)
      x = BatchNormalization()(x)
      x = Activation('relu')(x)

      x = Conv2D(self.n_channels, (5,5), padding='same')(x)
      x = BatchNormalization()(x)
      x = Activation('relu')(x)

      x = Conv2D(self.n_channels, (5,5), strides=(2,2))(x)
      x = BatchNormalization()(x)
      x = Activation('relu')(x)

      x = residual_block(x, input_channels=32, output_channels=64) # 16x16
      x = attention_block(x, encoder_depth=2)

      x = residual_block(x, input_channels=64, output_channels=128, stride=2)  # 8x8
      x = attention_block(x, encoder_depth=1)

      x = residual_block(x, input_channels=128, output_channels=256, stride=2)  # 4x4
      x = attention_block(x, encoder_depth=1)

      x = residual_block(x, input_channels=256, output_channels=512)
      x = residual_block(x, input_channels=512, output_channels=512)
      x = residual_block(x, input_channels=512, output_channels=512)
      
      x = AveragePooling2D(pool_size=(7,7), strides=(1, 1))(x)  # 1x1
      x = Flatten()(x)

      if self.dropout:
        x = Dropout(self.dropout)(x, training=self.dropout_act)

      output = Dense(self.n_classes, activation='sigmoid')(x)

      model = Model(input_, output)

      model = {'model':model, 'optimizer':self.optimizer, 'loss_fn':self.loss_fn,
            'model_name':self.model_name, 'model_type':self.model_type,'supervised':self.supervised}
      return model

    def call(self, inputs):
      output = self.model['model'](inputs)
      return output
