from tensorflow.keras import Model
from tensorlayer.layers import *
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import Input, Dense, DeConv2d, Reshape, BatchNorm2d, Conv2d, Flatten
import keras
import keras.backend as K

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from tensorflow.keras.optimizers import Adam

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier


estimators = [
               ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
               ('svm', make_pipeline(StandardScaler(),
                                     LinearSVC(random_state=42)))
]
lung_gan_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

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

class Generator():
    def __init__(self) -> None:
        super().__init__()
        self.shape = (None, 100)
        self.gf_dim = 16
        self.model_name = 'lung_gan_g'
        self.model_type  = 'keras'

        self.optimizer = Adam()

        self.lr = 2e-4
        self.supervised = False

        self.loss_fn = WeightedBCE()

    def build_model(self): # Dimension of gen filters in first conv layer. [64]
        image_size = 480
        s2, s4, s8, s16, s32, s64, s128 = int(image_size/2), int(image_size/4), int(image_size/8), int(image_size/16), int(image_size/32), int(image_size/64), int(image_size/128)
        k=4

        w_init = tf.random_normal_initializer(stddev=0.02)
        gamma_init = tf.random_normal_initializer(1., 0.02)

        ni = Input(self.shape)
        nn = Dense(n_units=(self.gf_dim * 64 * s128 * s128), W_init=w_init, b_init=None)(ni)
        nn = Reshape(shape=[-1, s128, s128, self.gf_dim*64])(nn)

        nn = BatchNorm2d(act=tf.nn.relu, gamma_init=gamma_init, name=None)(nn)
        nn = DeConv2d(self.gf_dim * 32, (k, k), (2, 2), W_init=w_init, b_init=None, padding='SAME')(nn)
        nn = BatchNorm2d(act=tf.nn.relu, gamma_init=gamma_init)(nn)
        nn = DeConv2d(self.gf_dim * 16, (5, 5), (2, 2), W_init=w_init, b_init=None, padding='VALID')(nn)
        nn = BatchNorm2d(act=tf.nn.relu, gamma_init=gamma_init)(nn)
        nn = DeConv2d(self.gf_dim*8, (k, k), (2, 2), W_init=w_init, b_init=None, padding='SAME')(nn)
        nn = BatchNorm2d(act=tf.nn.relu, gamma_init=gamma_init)(nn)
        nn = DeConv2d(self.gf_dim*4, (k, k), (2, 2), W_init=w_init, b_init=None, padding='SAME')(nn)
        nn = BatchNorm2d(act=tf.nn.relu, gamma_init=gamma_init)(nn)
        nn = DeConv2d(self.gf_dim*2, (k, k), (2, 2), W_init=w_init, b_init=None, padding='SAME')(nn)
        nn = BatchNorm2d(act=tf.nn.relu, gamma_init=gamma_init)(nn)
        nn = DeConv2d(self.gf_dim*1, (k, k), (2, 2), W_init=w_init, b_init=None, padding='SAME')(nn)
        nn = BatchNorm2d(act=tf.nn.relu, gamma_init=gamma_init)(nn)
        
        nn = DeConv2d(3, (k, k), (2, 2), act=tf.nn.tanh, W_init=w_init, padding='SAME')(nn)
        print(nn.shape)
        model = tl.models.Model(inputs=ni, outputs=nn, name='generator')
        return {'model':model, 'model_name':self.model_name,
                'loss_fn':self.loss_fn, 'lr':self.lr, 'optimizer':self.optimizer, 'supervised':self.supervised}

class Discriminator():
    def __init__(self) -> None:
        super().__init__()
        self.shape = (None, 480, 480, 3)
        self.df_dim = 32

        self.model_name = 'lung_gan_d'
        self.model_type  = 'keras'

        self.optimizer = Adam()

        self.lr = 2e-4
        self.loss_fn = WeightedBCE()

        self.supervised = True


    def build_model(self): # Dimension of discrim filters in first conv layer. [64]
        # w_init = tf.glorot_normal_initializer()
        w_init = tf.random_normal_initializer(stddev=0.02)
        gamma_init = tf.random_normal_initializer(1., 0.02)
        lrelu = lambda x : tf.nn.leaky_relu(x, 0.2)
        k=5

        ni = Input(self.shape)
        nn = Conv2d(16, (k, k), (2, 2), act=lrelu, W_init=w_init, padding='SAME')(ni)
        nn = Conv2d(self.df_dim, (k, k), (2, 2), act=lrelu, W_init=w_init, padding='SAME')(nn)
        nn = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(nn)

        nn = Conv2d(self.df_dim*2, (k, k), (2, 2), W_init=w_init, b_init=None, padding='SAME')(nn)
        nn = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(nn)

        nn = Conv2d(self.df_dim*4, (k, k), (2, 2), W_init=w_init, b_init=None, padding='SAME')(nn)
        nn = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(nn)

        nn = Conv2d(self.df_dim*8, (k, k), (2, 2), W_init=w_init, b_init=None, padding='SAME')(nn)
        nn = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(nn)

        maxpool1 = MaxPool2d(filter_size=(2, 2), padding='SAME')(nn)
        maxpool1 = Flatten()(maxpool1)

        nn = Conv2d(self.df_dim*16, (k, k), (2, 2), W_init=w_init, b_init=None, padding='SAME')(nn)
        nn = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(nn)
        maxpool2 = MaxPool2d(padding='SAME')(nn)
        maxpool2 = Flatten()(maxpool2)

        nn = Conv2d(self.df_dim*32, (k, k), (2, 2), W_init=w_init, b_init=None, padding='SAME')(nn)
        nn = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(nn)

        nn = Flatten()(nn)
        maxpool3 = Flatten()(nn)

        feature = Concat()([maxpool1,maxpool2,maxpool3])
        logits = Dense(n_units=1, act=tf.identity, W_init=w_init, b_init=None)(feature)
        model = tl.models.Model(inputs=ni, outputs=[logits, feature], name='discriminator')

        return {'model':model, 'model_name':self.model_name,
                'loss_fn':self.loss_fn, 'lr':self.lr, 'optimizer':self.optimizer, 'supervised':self.supervised}


d = Discriminator()
discriminator = d.build_model()

g = Generator()
generator = g.build_model()

Lung_GAN = {'model_name':'lung_gan', 'model_type':'keras', 'G': generator, 'D':discriminator}

if __name__ == "__main__":
    print('G')
    G = Lung_GAN['G']
    for layer in G['model'].all_layers:
      print(layer)
    print(G['model'].n_weights)
    
    print('D')
    D = Lung_GAN['D']
    for layer in D['model'].all_layers:
      print(layer)
    print(D['model'].n_weights)
