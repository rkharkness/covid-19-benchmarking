from pandas.core import base
from tensorflow.keras import layers
from tensorflow.keras.layers import AveragePooling2D, BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D,ZeroPadding2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
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
from tensorflow.keras import backend

import tensorflow as tf
from keras import backend as K
import keras

from math import pi
from math import cos
from math import floor
 
# snapshot ensemble with custom learning rate schedule

# class SnapshotEnsemble(Callback):
# 	# constructor
# 	def __init__(self, n_epochs, n_cycles, lrate_max, verbose=0):
# 		self.epochs = n_epochs
# 		self.cycles = n_cycles
# 		self.lr_max = lrate_max
# 		self.lrates = list()

# 	# calculate learning rate for epoch
# 	def cosine_annealing(self, epoch, n_epochs, n_cycles, lrate_max):
# 		epochs_per_cycle = floor(n_epochs/n_cycles)
# 		cos_inner = (pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
# 		return lrate_max/2 * (cos(cos_inner) + 1)

# 	# calculate and set learning rate at the start of the epoch
# 	def on_epoch_begin(self, epoch, logs={}):
# 		# calculate learning rate
# 		lr = self.cosine_annealing(epoch, self.epochs, self.cycles, self.lr_max)
# 		# set learning rate
# 		backend.set_value(self.model.optimizer.lr, lr)
# 		# log value
# 		self.lrates.append(lr)

# 	# save models at the end of each cycle
# 	def on_epoch_end(self, epoch, logs={}):
# 		# check if we can save model
# 		epochs_per_cycle = floor(self.epochs / self.cycles)
	
# 		if epoch != 0 and (epoch + 1) % epochs_per_cycle == 0:
# 			# save model to file
# 			#filename = "snapshot_model_%d.h5" % int((epoch + 1) / epochs_per_cycle)
			
# 			filename = "/content/drive/My Drive/Colab Notebooks/Snapshot Ensemble/snapshot_model_%d.h5" % int((epoch + 1) / epochs_per_cycle)
	
# 			self.model.save(filename)
# 			print('>saved snapshot %s, epoch %d' % (filename, epoch))
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
        y_pred = tf.cast(y_pred, tf.float32)

        # Calculate the binary crossentropy
        b_ce = K.binary_crossentropy(y_true, y_pred)

        # Apply the weights
        weight_vector = y_true * self.weights['1'] + (1. - y_true) * self.weights['0']
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return K.mean(weighted_b_ce)

class SwishActivation(Activation):
    
    def __init__(self, activation, **kwargs):
        super(SwishActivation, self).__init__(activation, **kwargs)
        self.__name__ = 'swish_act'

def swish_act(x, beta = 1):
    return (x * sigmoid(beta * x))


get_custom_objects().update({'swish_act': SwishActivation(swish_act)})

class ECovNet(Model):
    #loss_fn = BinaryCrossentropy(from_logits=False)
    loss_fn = WeightedBCE()
    optimizer = Adam()
    model_type = 'keras'
    model_name = 'ecovnet'

    def __init__(self, trial=None) -> None:
        super().__init__()
        self.base_model = enet.EfficientNetB1(include_top=False, input_shape=(480,480,3), pooling='avg', weights="imagenet",classes=1)
        self.batchnorm1 = BatchNormalization()
        self.batchnorm2 = BatchNormalization()

        self.dense1 = Dense(512, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-3))
        self.dense2 = Dense(512, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-3))

        self.activation = swish_act 
        self.dropout = Dropout(0.5)
        self.dense3 = Dense(1, activation="sigmoid")

    def call(self, input):
     #   base_model = enet.EfficientNetB1(include_top=False, input_shape=(240,240,3), pooling='avg', weights="imagenet",classes=2)
        x = self.base_model(input)

        x = self.batchnorm1(x)
        x = self.dense1(x)
        x = self.activation(x)

        x = self.dense2(x)
        x = self.dropout(x)

        x = self.batchnorm2(x)
        x = self.activation(x)

        output = self.dense3(x)
        return output