from pandas.core import base
from keras import layers
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D,ZeroPadding2D
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.applications import DenseNet121
from keras.applications import MobileNetV2
from keras.regularizers import l2
from keras.layers import Activation
from keras.regularizers import l1_l2

from keras import optimizers
import efficientnet.tfkeras as enet

from keras.backend import sigmoid
from keras.utils import get_custom_objects
from keras.layers import Activation

from keras.callbacks import Callback
from keras.optimizers import SGD
from keras import backend

from keras.optimizers import CategoricalCrossentropy

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

class SwishActivation(Activation):
    
    def __init__(self, activation, **kwargs):
        super(SwishActivation, self).__init__(activation, **kwargs)
        self.__name__ = 'swish_act'

def swish_act(x, beta = 1):
    return (x * sigmoid(beta * x))


get_custom_objects().update({'swish_act': SwishActivation(swish_act)})
class ECovNet(Model):
    def __init__(self, model_type, optimizer, loss) -> None:
        super().__init__()
        self.base_model = enet.EfficientNetB1(include_top=False, input_shape=(480,480,3), pooling='avg', weights="imagenet",classes=2)
        self.batchnorm = BatchNormalization()
        self.dense1 = Dense(512, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-3))
        self.activation(swish_act)
        self.dropout = Dropout(0.5)
        self.dense2 = Dense(2, activation="softmax")

    def call(self, input):
     #   base_model = enet.EfficientNetB1(include_top=False, input_shape=(240,240,3), pooling='avg', weights="imagenet",classes=2)
        x = self.base_model.output(input)
        x = self.batchnorm(x)
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.dense2(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        x = self.dropout(x)

        output = self.dense2(x)
        return output
    
    @staticmethod
    def get_loss_fn():
        return CategoricalCrossentropy()
    
    @staticmethod
    def get_optimizer():
        return Adam()
    
    @staticmethod
    def get_model_type():
        return 'keras'