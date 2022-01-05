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
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Multiply
from tensorflow.keras.layers import Lambda
from tensorflow.keras import Model

from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

#for local debugging without gpu
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class AttentionResNet56(Model):
    def __init__(self) -> None:
        super().__init__()
        self.shape = 480
        self.n_channels = 4
        self.loss = CategoricalCrossentropy()
        self.optimizer = Adam()
        self.model_type = 'keras'

        self.conv1 = Conv2D(self.n_channels, (4, 4), strides=(2, 2), padding='same')
        self.batchnorm= BatchNormalization()
        self.activation = Activation('relu')
        self.maxpool = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same') 
       # self.avgpool = AveragePooling2D(pool_size=self.get_pool_size(), strides=(1, 1))
        self.avgpool = AveragePooling2D(pool_size=(3,3), strides=(1, 1))
        self.flatten = Flatten()
        self.dropout = Dropout(0.5)
        self.dense = Dense(2, activation='softmax')

    def get_pool_size(self, x):
        return (x.get_shape()[1], x.get_shape()[2])

    def residual_block(input, input_channels=None, output_channels=None, kernel_size=(3, 3), stride=1):
        '''
        Source: https://github.com/qubvel/residual_attention_network
        '''
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
        x = Conv2D(input_channels, kernel_size, padding='same',  strides=stride)(x)

        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(output_channels, (1, 1), padding='same')(x)

        if input_channels != output_channels or stride != 1:
            input = Conv2D(output_channels, (1, 1), padding='same', strides=strides)(input)

        x = Add()([x, input])
        return x


    def attention_block(self, input, input_channels=None, output_channels=None, encoder_depth=1):
        '''
        Source: https://github.com/qubvel/residual_attention_network
        '''

        p = 1
        t = 2
        r = 1

        if input_channels is None:
            input_channels = input.get_shape()[-1]
        if output_channels is None:
            output_channels = input_channels

        # First Residual Block
        for i in range(p):
            input = self.residual_block(input)

        # Trunc Branch
        output_trunk = input
        for i in range(t):
            output_trunk = self.residual_block(output_trunk)


        # Soft Mask Branch
        ## encoder
        ### first down sampling
        output_soft_mask = MaxPool2D(padding='same')(input)  # 32x32
        for i in range(r):
            output_soft_mask = self.residual_block(output_soft_mask)

        skip_connections = []
        for i in range(encoder_depth - 1):

            ## skip connections
            output_skip_connection = self.residual_block(output_soft_mask)
            skip_connections.append(output_skip_connection)
            # print ('skip shape:', output_skip_connection.get_shape())

            ## down sampling
            output_soft_mask = MaxPool2D(padding='same')(output_soft_mask)
            for _ in range(r):
                output_soft_mask = self.residual_block(output_soft_mask)


        ## decoder
        skip_connections = list(reversed(skip_connections))
        for i in range(encoder_depth - 1):
            ## upsampling666
            for _ in range(r):
                output_soft_mask = self.residual_block(output_soft_mask)
            output_soft_mask = UpSampling2D()(output_soft_mask)
            ## skip connections
            output_soft_mask = Add()([output_soft_mask, skip_connections[i]])

        ### last upsampling
        for i in range(r):
            output_soft_mask = self.residual_block(output_soft_mask)
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
            output = self.residual_block(output)

        return output


    def call(self, input):
        '''
        Source: https://github.com/qubvel/residual_attention_network
        '''
        x = self.conv1(input)
        x = self.batchnorm(x)
        x = self.activation(x)
        x = self.maxpool(x)
        
        # x = self.residual_block(x, output_channels=n_channels * 4)  # 56x56
        # x = self.attention_block(x, encoder_depth=4)  # bottleneck 7x7

        x = self.residual_block(x, output_channels=self.n_channels * 4)  # 56x56
        x = self.attention_block(x, encoder_depth=3)  # bottleneck 7x7

        x = self.residual_block(x, output_channels=self.n_channels * 8, stride=2)  # 28x28
        x = self.attention_block(x, encoder_depth=2)  # bottleneck 7x7

        x = self.residual_block(x, output_channels=self.n_channels * 16, stride=2)  # 14x14
        x = self.attention_block(x, encoder_depth=1)  # bottleneck 7x7

        x = self.residual_block(x, output_channels=self.n_channels * 32, stride=2)  # 7x7
        x = self.residual_block(x, output_channels=self.n_channels * 32)
        x = self.residual_block(x, output_channels=self.n_channels * 32)

       # pool_size = (x.get_shape()[1], x.get_shape()[2])
        x = self.avgpool(x)
        x = self.flatten(x)

        x = self.dropout(x)
        output = self.dense(x)

        return output

    def get_loss_fn(self):
        return self.loss
    
    def get_optimizer(self):
        return self.optimizer
    
    def get_model_type(self):
        return self.model_type
    
    def build_graph(self, input_shape):
        x = Input(shape=(None, input_shape), ragged=True)
        return Model(inputs=[x], outputs=self.forward(x))

res_attn = AttentionResNet56()