from __future__ import print_function 

from tensorflow.keras import backend as K 
from tensorflow.keras.layers import Layer, Input 
from tensorflow.keras import activations 
from tensorflow.keras import utils

from tensorflow.keras.models import Model 
from tensorflow.keras.layers import * 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
import numpy as np 
import tensorflow.keras as keras 

from tensorflow.keras.callbacks import ModelCheckpoint 
from tensorflow.keras import optimizers 
import tensorflow as tf

from tensorflow.keras import initializers

K.set_image_data_format('channels_last')
###https://github.com/XifengGuo/CapsNet-Keras/blob/9d7e641e3f30f0e8227bb6ad521a61e908c2408a/capsulelayers.py#L87

def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (1 + s_squared_norm)
    return scale * x

def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)

class WeightedBCE(keras.losses.Loss):
  def __init__(self, trial=None):
    """adapted from: https://stackoverflow.com/questions/46009619/keras-weighted-binary-crossentropy"""            
    super().__init__()
    self.trial = trial
    self.weights = {'0':0.2, '1':0.8}

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
#class MarginLoss(keras.losses.Loss):
 # def __init__(self, trial=None):
 ##     super().__init__()
#      self.trial = trial
#def margin_loss(y_true, y_pred):
#    lamb, margin = 0.5, 0.1 
 #   return K.sum(y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (
  #      1 - y_true) * K.square(K.relu(y_pred - margin)), axis=-1)

def MarginLoss():
    def loss(y_true, y_pred):
   ##   y_true = tf.cast(y_true, tf.float32)
     # y_pred = tf.cast(y_pred, tf.float32)
        lamb, margin = 0.5, 0.1
        return K.mean(K.sum((y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (
              1 - y_true) * K.square(K.relu(y_pred - margin))), axis=-1))
    return loss

#def MarginLoss():
#    def margin_loss(y_true, y_pred):
#        L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
 #           0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

  #      return K.mean(K.sum(L, 1))
    
  #  return margin_loss
          

class LDAMLoss(keras.losses.Loss):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = tf.zeros(x.shape, dtype=tf.dtypes.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(tf.float32)
        batch_m = tf.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = tf.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)

class Capsule(Layer):
    """
    The capsule layer. It is similar to Dense layer. Dense layer has `in_num` inputs, each is a scalar, the output of the
    neuron from the former layer, and it has `out_num` output neurons. CapsuleLayer just expand the output of the neuron
    from scalar to vector. So its input shape = [None, input_num_capsule, input_dim_capsule] and output shape = \
    [None, num_capsule, dim_capsule]. For Dense Layer, input_dim_capsule = dim_capsule = 1.
    :param num_capsule: number of capsules in this layer
    :param dim_capsule: dimension of the output vectors of the capsules in this layer
    :param routings: number of iterations for the routing algorithm
    """
    def __init__(self, num_capsule, dim_capsule, routings=3,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_initializer = tf.keras.initializers.GlorotNormal()
    
    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        # Transform matrix, from each input capsule to each output capsule, there's a unique weight as in Dense layer.
        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                        self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.built = True

    def call(self, inputs, training=None):
        # inputs.shape=[None, input_num_capsule, input_dim_capsule]
        # inputs_expand.shape=[None, 1, input_num_capsule, input_dim_capsule, 1]
        inputs_expand = tf.expand_dims(tf.expand_dims(inputs, 1), -1)

        # Replicate num_capsule dimension to prepare being multiplied by W
        # inputs_tiled.shape=[None, num_capsule, input_num_capsule, input_dim_capsule, 1]
        inputs_tiled = tf.tile(inputs_expand, [1, self.num_capsule, 1, 1, 1])

        # Compute `inputs * W` by scanning inputs_tiled on dimension 0.
        # W.shape=[num_capsule, input_num_capsule, dim_capsule, input_dim_capsule]
        # x.shape=[num_capsule, input_num_capsule, input_dim_capsule, 1]
        # Regard the first two dimensions as `batch` dimension, then
        # matmul(W, x): [..., dim_capsule, input_dim_capsule] x [..., input_dim_capsule, 1] -> [..., dim_capsule, 1].
        # inputs_hat.shape = [None, num_capsule, input_num_capsule, dim_capsule]
        inputs_hat = tf.squeeze(tf.map_fn(lambda x: tf.matmul(self.W, x), elems=inputs_tiled))

        # Begin: Routing algorithm ---------------------------------------------------------------------#
        # The prior for coupling coefficient, initialized as zeros.
        # b.shape = [None, self.num_capsule, 1, self.input_num_capsule].
        b = tf.zeros(shape=[inputs.shape[0], self.num_capsule, 1, self.input_num_capsule])

        assert self.routings > 0, 'The routings should be > 0.'
        for i in range(self.routings):
            # c.shape=[batch_size, num_capsule, 1, input_num_capsule]
            c = tf.nn.softmax(b, axis=1)

            # c.shape = [batch_size, num_capsule, 1, input_num_capsule]
            # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
            # The first two dimensions as `batch` dimension,
            # then matmal: [..., 1, input_num_capsule] x [..., input_num_capsule, dim_capsule] -> [..., 1, dim_capsule].
            # outputs.shape=[None, num_capsule, 1, dim_capsule]
            outputs = squash(tf.matmul(c, inputs_hat))  # [None, 10, 1, 16]

            if i < self.routings - 1:
                # outputs.shape =  [None, num_capsule, 1, dim_capsule]
                # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
                # The first two dimensions as `batch` dimension, then
                # matmal:[..., 1, dim_capsule] x [..., input_num_capsule, dim_capsule]^T -> [..., 1, input_num_capsule].
                # b.shape=[batch_size, num_capsule, 1, input_num_capsule]
                b += tf.matmul(outputs, inputs_hat, transpose_b=True)
        # End: Routing algorithm -----------------------------------------------------------------------#

        return tf.squeeze(outputs)

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])

    def get_config(self):
        config = {
            'num_capsule': self.num_capsule,
            'dim_capsule': self.dim_capsule,
            'routings': self.routings
        }
        base_config = super(Capsule, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CovidCaps(Model):

    #model_type = 'keras'
   # model_name = 'capsnet'

    def __init__(self, supervised=True, pretrained=None) -> None:
        super(CovidCaps, self).__init__()

        self.supervised = supervised
        self.pretrained = pretrained

        self.loss_fn = MarginLoss() #WeightedBCE() #keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE) #MarginLoss()
        self.optimizer = optimizers.Adam()
        self.model_type = 'keras'
        self.model_name = 'capsnet'
        self.shape = 480
        self.lr =1e-3

        self.reshape = Reshape((-1,128))

        self.conv1 = Conv2D(64, (4, 4), strides=(2,2), activation='relu') #, input_shape = (None,480, 480, 3))
        self.conv2 = Conv2D(64, (3, 3), strides=(2,2), activation='relu')
        self.conv3 = Conv2D(64, (3, 3), strides=(2,2), activation='relu')
        self.conv4 = Conv2D(128, (3, 3), strides=(2,2), activation='relu')
        #self.conv5 = Conv2D(, (3, 3), strides=(3,3), activation='relu')
       
        self.batchnorm1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
        self.batchnorm2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
        self.batchnorm3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
        self.avgpool = AveragePooling2D((2,2))
        self.dropout = Dropout(0.3)
 #       self.capsule1 = Capsule(32, 8, 4, True) #32
#        self.capsule2 = Capsule(32,8, 4, True)

        if self.pretrained !=None:
          self.capsule_bin = Capsule(2, 16, 4, True) # num classes, - , -
        #else:
        self.capsule1 = Capsule(5, 16, 4, True) 

        self.lam = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))

    def call(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = self.conv3(x)
        x = self.batchnorm2(x)
        x = self.avgpool(x) # try without on dgx
        x = self.conv4(x)
        x = self.batchnorm3(x)
#        x = self.avgpool(x)
 #       x = self.conv5(x)
        print(x.shape)
#        x = self.avgpool(x)
       # x = self.batchnorm(x)
        x = self.reshape(x)
        print('r',x.shape)
 #       x = self.dropout(x)

        x = self.capsule1(x)
        #x = self.capsule2(x)
       # x = self.capsule3(x)
        output = self.lam(x)
        print('o',output.shape)
        return output

    def build_model(self, input_shape, batch_size):
        x = Input(shape=input_shape, batch_size=batch_size) #, ragged=True)
        model = Model(inputs=[x], outputs=self.call(x))
        if self.pretrained != None:
             if self.pretrained != True:
                 model.load_weights(self.pretrained)
             x = self.capsule_bin(model.layers[-2].output)
             output = self.lam(x)
             model = Model(model.input, output)
             #x = self.capsule_bin(model.layers[-2].output)
             #output = self.lam(x)
#             modelf = Model(model.input, output)
#             model1 = keras.Sequential()
             #model1.add(Input(shape=input_shape, batch_size=batch_size))
 #            for l in model.layers:
#                 model1.add(l)
             
        return {'model':model, 'optimizer':self.optimizer, 'loss_fn':self.loss_fn, 'lr':self.lr,
        'model_name':self.model_name, 'model_type':self.model_type, 'supervised':self.supervised, 'pretrained':self.pretrained}

if __name__ == "__main__":
    model = CovidCaps(pretrained=True).build_model((480,480,3),24)
    print(model['model'].summary())

