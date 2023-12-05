from tensorflow.keras import layers
from tensorflow.keras.layers import AveragePooling2D, BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D,ZeroPadding2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD

from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Activation
from tensorflow.keras.regularizers import l1_l2

from tensorflow.keras import optimizers
import efficientnet.tfkeras as enet

from tensorflow.keras.backend import sigmoid
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import SGD

import keras
from keras import backend as K

import tensorflow as tf 

class SwishActivation(Activation):
    
    def __init__(self, activation, **kwargs):
        super(SwishActivation, self).__init__(activation, **kwargs)
        self.__name__ = 'swish_act'

def swish_act(x, beta = 1):
    return (x * sigmoid(beta * x))


# class WeightedBCE(keras.losses.Loss):
#   def __init__(self, trial=None):
#     """adapted from: https://stackoverflow.com/questions/46009619/keras-weighted-binary-crossentropy"""            
#     super().__init__()
#     self.trial = trial
#     self.weights = {'0':1.32571275, '1':0.80276873}

#   def call(self, y_true, y_pred):
#         # Original binary crossentropy (see losses.py):
#         # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
#         y_true = tf.cast(y_true, tf.float32)
#         y_pred = tf.cast((y_pred+1e-10), tf.float32)

#         # Calculate the binary crossentropy
#         b_ce = K.binary_crossentropy(y_true, y_pred)

#         # Apply the weights
#         weight_vector = y_true * self.weights['1'] + (1. - y_true) * self.weights['0']
#         weighted_b_ce = weight_vector * b_ce

#         # Return the mean error
#         return K.mean(weighted_b_ce)
    

get_custom_objects().update({'swish_act': SwishActivation(swish_act)})


class ECovNet(Model):
    def __init__(self, dropout_act=True) -> None:
        super().__init__()

        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.0, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        self.optimizer = SGD()
        self.model_type = 'keras'
        self.model_name = 'ecovnet'
        self.supervised = True
        self.dropout_act = dropout_act
        self.lr = 1e-4
        self.model = self.build_model()
    
    def build_model(self):
        base_model = enet.EfficientNetB1(include_top=False, input_shape=(480,480,3), pooling='avg', weights="imagenet",classes=1)
        
        self.base_model = Model(base_model.input, base_model.output)
        
        x = self.base_model.output

#        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        #x = Dropout(0.3)(x, training=self.dropout_act)
        x = Dense(512,kernel_regularizer=l1_l2(l1=1e-5, l2=1e-3))(x)
        x = BatchNormalization()(x)
        x = Activation(swish_act)(x)
        x = Dropout(0.3)(x, training=self.dropout_act)

        x = Dense(256,kernel_regularizer=l1_l2(l1=1e-5, l2=1e-3))(x)
        x = BatchNormalization()(x)
        x = Activation(swish_act)(x) 
        x = Dropout(0.3)(x, training=self.dropout_act)

        # Output layer
        predictions = Dense(1, activation="sigmoid")(x)
        model = Model(inputs = base_model.input, outputs = predictions)

        model = {'model':model, 'optimizer':self.optimizer, 'loss_fn':self.loss_fn, 'lr':self.lr,
        'model_name':self.model_name, 'model_type':self.model_type, 'supervised':self.supervised}

        return model
    
    def freeze_base(self):
        for layer in self.base_model.layers:
            layer.trainable = False
    
    def unfreeze_base(self):
        for layer in self.base_model.layers:
            layer.trainable = True  


    def call(self, inputs):
        output = self.model['model'](inputs)
        return output


if __name__ == "__main__":
    model = ECovNet().build_model()

    for k, v in model.items():
        print(k, v)

    print(model['model'].summary())
