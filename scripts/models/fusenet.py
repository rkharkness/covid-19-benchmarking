

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.layers import Conv2D, SeparableConv2D, MaxPool2D
import tensorflow.keras.backend as K
import keras 

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
        y_pred = tf.cast(y_pred+1e-10, tf.float32)

        # Calculate the binary crossentropy
        b_ce = K.binary_crossentropy(y_true, y_pred)

        # Apply the weights
        weight_vector = y_true * self.weights['1'] + (1. - y_true) * self.weights['0']
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return K.mean(weighted_b_ce)

def get_dropout(input_tensor, rate, mc=False):
    if mc:
        return Dropout(rate=rate)(input_tensor, training=True)
    else:
        return Dropout(rate=rate)(input_tensor)

# Our Proposed Fusion Model:

class FuseNet(Model):
    def __init__(self, dropout_act=True):
        super(FuseNet, self).__init__()
        self.dropout_act = dropout_act
        self.lr = 1e-4
        self.model_name = 'fusenet'
        self.model_type = 'keras'
        self.optimizer = tf.keras.optimizers.Adam()
        self.supervised = True
        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)  #WeightedBCE()


    def build_model(self, image_size=480):
        inputs = Input(shape=(image_size, image_size, 3))
        #input2 = tf.stack([inputs, inputs, inputs], axis=3)[:, :, :, :, 0]
        vgg_model = tf.keras.applications.VGG16(weights='imagenet',
                                                include_top=False,
                                                input_shape=(image_size, image_size, 3))
        vgg_model.trainable = False   
        vgg_model_plus = keras.Sequential()
        for layer in vgg_model.layers:
            vgg_model_plus.add(layer)
        vgg_model_plus.add(MaxPool2D(pool_size=(2,2)))
        vgg_model_plus.add(Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), activation='relu'))
        vgg_model_plus.add(MaxPool2D(pool_size=(2,2)))
 #       vgg_model_plus.add(Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), activation='relu')) 
       # vgg_model_plus.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu'))
       # vgg_model_plus.layers[:-4].trainable = False
        vgg_feature = vgg_model_plus(inputs)

        # First conv block
        conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(2,2), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
        conv1 = MaxPool2D(pool_size=(2, 2))(conv1)

        # Second conv block
        conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(2,2), activation='relu', padding='same')(conv1)
        conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = MaxPool2D(pool_size=(2, 2))(conv2)

        # Third conv block
        conv3 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(2,2), activation='relu', padding='same')(conv2)
        conv3 = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = MaxPool2D(pool_size=(2, 2))(conv3)

        # Fourth conv block
        conv4 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(2,2), activation='relu', padding='same')(conv3)
        conv4 = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', name='target_layer')(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = MaxPool2D(pool_size=(2, 2))(conv4)

        # Fifth conv block
        conv5 = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv4)
        conv5 = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv5)
        conv5 = BatchNormalization()(conv5)
        conv5 = MaxPool2D(pool_size=(2, 2))(conv5)

        concatenated_tensor = Concatenate(axis=1)(
            [Flatten()(conv3), Flatten()(conv4), Flatten()(conv5), Flatten()(vgg_feature)])

        # FC layer
        x = Flatten()(concatenated_tensor)
        x = Dense(units=512, activation='relu')(x)
#        x = get_dropout(x, rate=0.3, mc=self.dropout_act)
 #       x = Dense(units=256, activation='relu')(x)
        x = get_dropout(x, rate=0.3, mc=self.dropout_act)
        x = Dense(units=128, activation='relu')(x)
        x = get_dropout(x, rate=0.3, mc=self.dropout_act)
        # Output layer
        x = Dense(units=64, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)

        # Creating model and compiling
        model = Model(inputs=inputs, outputs=output)
        return {'model':model, 'optimizer':self.optimizer, 'loss_fn':self.loss_fn, 'lr':self.lr,
        'model_name':self.model_name, 'model_type':self.model_type, 'supervised':self.supervised}

if __name__ == "__main__":
    fusenet = FuseNet()
    model = fusenet.build_model()
    model['model'].load_weights('/MULTIX/DATA/HOME/covid-19-benchmarking/weights/fusenet/fusenet_supervised_1.h5')
    print(model['model'].summary())
