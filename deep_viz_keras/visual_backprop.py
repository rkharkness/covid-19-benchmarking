
from deep_viz_keras.saliency import SaliencyMask
import numpy as np
import keras.backend as K
from keras.layers import Input, Conv2DTranspose
from keras.models import Model
from keras.initializers import Ones, Zeros
import tensorflow as tf

class VisualBackprop(SaliencyMask):
    """A SaliencyMask class that computes saliency masks with VisualBackprop (https://arxiv.org/abs/1611.05418).
    """

    def __init__(self, model, output_index=0):
        """Constructs a VisualProp SaliencyMask."""
        inps = [model['model'].input]           # input placeholder
        outs = [layer.output for layer in model['model'].layers]    # all layer outputs
        self.forward_pass = K.function(inps, outs)         # evaluation function
        
        self.model = model['model']

    def get_mask(self, input_image):
        """Returns a VisualBackprop mask."""
        #x_value = tf.convert_to_tensor(input_image, dtype=tf.float32)
        x_value = input_image
#        print(x_value.shape)
        visual_bpr = None
        layer_outs = self.forward_pass([x_value,0]) #tf.convert_to_tensor(0, dtype=tf.float32)])

        for i in range(len(self.model.layers)-1, -1, -1):
            if 'Conv2D' in str(type(self.model.layers[i])):
                layer = np.mean(layer_outs[i], axis=3, keepdims=True)
                layer = layer - np.min(layer)
                layer = layer/(np.max(layer)-np.min(layer)+1e-6)

                if visual_bpr is not None:
                    if visual_bpr.shape != layer.shape:
                        visual_bpr = self._deconv(visual_bpr)
                    visual_bpr = visual_bpr * layer
                else:
                    visual_bpr = layer

        return visual_bpr[0]
    
    def _deconv(self, feature_map):
        """The deconvolution operation to upsample the average feature map downstream"""
        x = Input(shape=(None, None, 1))
        y = Conv2DTranspose(filters=1, 
                            kernel_size=(3,3), 
                            strides=(2,2), 
                            padding='same', 
                            kernel_initializer=Ones(), 
                            bias_initializer=Zeros())(x)

        deconv_model = Model(inputs=[x], outputs=[y])

        inps = [deconv_model.input]   # input placeholder                                
        outs = [deconv_model.layers[-1].output]           # output placeholder
        deconv_func = K.function(inps, outs)              # evaluation function
        
        return deconv_func([feature_map, 0])[0]
