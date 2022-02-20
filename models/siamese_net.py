# import required libraries
from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Flatten, Lambda, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
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
        y_pred = tf.cast((y_pred+1e-10), tf.float32)

        # Calculate the binary crossentropy
        b_ce = K.binary_crossentropy(y_true, y_pred)

        # Apply the weights
        weight_vector = y_true * self.weights['1'] + (1. - y_true) * self.weights['0']
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return K.mean(weighted_b_ce)

""" L1 mistance - manhattan """
def manhattan_distance(vects):
    x, y = vects
    return K.sum(K.abs(x-y), axis=1, keepdims=True)

""" L2 distance """
def euclidean_distance(vects):
    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

def loss(margin=1.25):
    def contrastive_loss(y_true, y_pred):
        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean((1 - y_true) * square_pred + (y_true) * margin_square)

    return contrastive_loss

class SiameseNetwork():
    def __init__(self, pretrained=None, dropout_act=True):
        super(SiameseNetwork, self).__init__()
        self.model_type= 'keras'
        self.model_name = 'siamese_net'
        self.pretrained = pretrained
        self.dropout_act = dropout_act

        self.supervised = True

        self.optimizer = Adam()

        if self.pretrained:
            self.lr = 1e-4
            self.loss_fn = loss(margin=1.25)
        else:
            self.lr = 1e-4
            self.loss_fn = WeightedBCE()

    def build_encoder(self):
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(480,480,3))       
        output = base_model.output
        x = Flatten()(output)
        prediction = Dense(1, activation='sigmoid')(x) # pred to train encoder/embedder
        
        return Model(inputs=[base_model.input], outputs=[prediction])

        
    def build_model(self):

        if self.pretrained: # build encoder
            input_1 = Input((480,480,3))
            input_2 = Input((480,480,3))

            embedding_network = self.build_encoder()
            embedding_network.load_weights(self.pretrained)
            embedding_network.trainable = False

            model = tf.keras.Sequential() 
            for layer in embedding_network.layers: # might neeed to remove last layer? 
                model.add(layer) 

            model.add(Flatten(name='flat'))
            model.add(Dropout(0.5)) # training=self.dropout_act))
            model.add(Dense(5120, name='den', activation='sigmoid', kernel_regularizer='l2')) 

            output_1 = embedding_network(input_1)
            output_2 = embedding_network(input_2)

            merge_layer = Lambda(manhattan_distance)([output_1, output_2]) 
            output_layer = Dense(1, activation="sigmoid")(merge_layer)

            self.model = Model(inputs=[input_1, input_2], outputs=output_layer) 
                
        else:
            self.model = self.build_encoder() 
        
        return {'model':self.model, 'optimizer':self.optimizer, 'loss_fn':self.loss_fn, 'lr':self.lr,
        'model_name':self.model_name, 'model_type':self.model_type, 'supervised':self.supervised, 'pretrained':self.pretrained}



    def call(self, inputs):
        output = self.model['model'](inputs)
        return output


if __name__ == "__main__":
    siamese_net = SiameseNetwork()
    model = siamese_net.build_model()

    for k, v in model.items():
        print(k, v)

    print(model['model'].summary())
