
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import Input, Dense, DeConv2d, Reshape, BatchNorm2d, Conv2d, Flatten

def get_generator(shape=(None, 100), gf_dim=64): # Dimension of gen filters in first conv layer. [64]
    image_size = 480
    s16 = image_size // 16
    k=4

    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    ni = Input(shape)
    nn = Dense(n_units=(gf_dim * 64 * s16 * s16), W_init=w_init, b_init=None)(ni)
    nn = Reshape(shape=[-1, s16, s16, gf_dim*64])(nn)

    nn = BatchNorm2d(act=tf.nn.relu, gamma_init=gamma_init, name=None)(nn)
    nn = DeConv2d(gf_dim * 32, (k, k), (2, 2), W_init=w_init, b_init=None, padding='SAME')(nn)
    nn = BatchNorm2d(act=tf.nn.relu, gamma_init=gamma_init)(nn)
    nn = DeConv2d(gf_dim * 16, (k, k), (2, 2), W_init=w_init, b_init=None, padding='SAME')(nn)
    nn = BatchNorm2d(act=tf.nn.relu, gamma_init=gamma_init)(nn)
    nn = DeConv2d(gf_dim*8, (k, k), (1, 1), W_init=w_init, b_init=None, padding='SAME')(nn)
    nn = BatchNorm2d(act=tf.nn.relu, gamma_init=gamma_init)(nn)
    nn = DeConv2d(gf_dim*4, (k, k), (2, 2), W_init=w_init, b_init=None, padding='SAME')(nn)
    nn = BatchNorm2d(act=tf.nn.relu, gamma_init=gamma_init)(nn)
    nn = DeConv2d(gf_dim*2, (k, k), (1, 1), W_init=w_init, b_init=None, padding='SAME')(nn)
    nn = BatchNorm2d(act=tf.nn.relu, gamma_init=gamma_init)(nn)
    nn = DeConv2d(gf_dim*1, (k, k), (2, 2), W_init=w_init, b_init=None, padding='SAME')(nn)
    nn = BatchNorm2d(act=tf.nn.relu, gamma_init=gamma_init)(nn)
    nn = DeConv2d(3, (k, k), (1, 1), act=tf.nn.tanh, W_init=w_init, padding='SAME')(nn)

    return tl.models.Model(inputs=ni, outputs=nn, name='generator')

def get_discriminator(shape=(None,480,480,3), df_dim=32): # Dimension of discrim filters in first conv layer. [64]
    # w_init = tf.glorot_normal_initializer()
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x : tf.nn.leaky_relu(x, 0.2)
    k=5

    ni = Input(shape)
    nn = Conv2d(16, (k, k), (2, 2), act=lrelu, W_init=w_init, padding='SAME')(ni)
    nn = Conv2d(df_dim, (k, k), (2, 2), act=lrelu, W_init=w_init, padding='SAME')(nn)
    nn = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(nn)

    nn = Conv2d(df_dim*2, (k, k), (2, 2), W_init=w_init, b_init=None, padding='SAME')(nn)
    nn = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(nn)

    nn = Conv2d(df_dim*4, (k, k), (2, 2), W_init=w_init, b_init=None, padding='SAME')(nn)
    nn = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(nn)

    nn = Conv2d(df_dim*8, (k, k), (2, 2), W_init=w_init, b_init=None, padding='SAME')(nn)
    nn = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(nn)

    maxpool1 = MaxPool2d(filter_size=(2, 2), padding='SAME')(nn)
    maxpool1 = Flatten()(maxpool1)

    nn = Conv2d(df_dim*16, (k, k), (2, 2), W_init=w_init, b_init=None, padding='SAME')(nn)
    nn = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(nn)

    maxpool2 = MaxPool2d(padding='SAME')(nn)
    maxpool2 = Flatten()(maxpool2)

    nn = Conv2d(df_dim*32, (k, k), (2, 2), W_init=w_init, b_init=None, padding='SAME')(nn)
    nn = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(nn)

    nn = Flatten()(nn)
    maxpool3 = Flatten()(nn)

    feature = Concat()([maxpool1,maxpool2,maxpool3])
    logits = Dense(n_units=1, act=tf.identity, W_init=w_init)(feature)

    return tl.models.Model(inputs=ni, outputs=[logits, feature], name='discriminator')

Lung_GAN = {'G': get_generator(), 'D':get_discriminator()}