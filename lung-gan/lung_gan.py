from glob import glob
from tensorflow.keras import Model
import tensorflow as tf
from tensorflow.layers import *
import tensorlayer as tl
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.optimizers import Adam


def lung_gan_training(model_dict, dataloader, k, patience=20, supervised=True, pretrained_weights=None):
    z_dim = 100

    assert type(model_dict) == dict

    G = model_dict['gen']
    D = model_dict['disc']

    writer1 = SummaryWriter(G.model_name)
    writer2 = SummaryWriter(D.model_name)

    if (pretrained_weights):
        assert type(pretrained_weights) == dict
        D.load_weights(pretrained_weights['disc'])
        G.load_weights(pretrained_weights['gen'])

    g_loss_dict = {'train': np.zeros(shape=(500,), dtype=np.float32),
                'val': np.zeros(shape=(500,), dtype=np.float32)}
    
    d_loss_dict = {'train': np.zeros(shape=(500,), dtype=np.float32),
                'val':np.zeros(shape=(500,), dtype=np.float32)}

    no_improvement = 0

    # for colab testing - before hp tuning
    optimizer_g = G.optimizer
    optimizer_d = D.optimizer

    optimizer_g.lr.assign(1e-4)
    lr_g = optimizer_g.lr

    optimizer_d.lr.assign(1e-4)
    lr_d = optimizer_d.lr

    best_val_loss = np.inf

    loss_fn_g = G.loss_fn
    loss_fn_d = D.loss_fn

    end_epoch = 0
    plt.figure()
    for epoch in range(500):
        g_loss_avg = {'train':[],'val':[]}
        d_loss_avg = {'train':[],'val':[]}

        end_epoch =+1
        for phase in ['train', 'val']:
            for batch in tqdm(dataloader[phase]):
                batch_size = len(batch)

                if len(batch) > 1:
                    batch_x, batch_y = batch # batch_y can be paired image 
                    with tf.GradientTape() as tape:
                        z = np.random.normal(loc=0.0, scale=1.0, size=[batch_size, z_dim]).astype(np.float32)
                        gen, g_logits = G(z)
                        x, d_logits, feature_fake = D(gen)
                        pred_real, d2_logits, feature_real = D(batch_x)
                       # discriminator: real images are labelled as 1
                        d_loss_real = loss_fn_d(d2_logits, tf.ones_like(d2_logits), name='dreal')
                         # discriminator: images from generator (fake) are labelled as 0
                        d_loss_fake = loss_fn_d(d_logits, tf.zeros_like(d_logits), name='dfake')
                         # combined loss for updating discriminator
                        d_loss = d_loss_real + d_loss_fake
                         # generator: try to fool discriminator to output 1
                        g_loss1 = tf.reduce_mean(loss_fn_d(d_logits, tf.ones_like(d_logits)))

                        g_loss2 = tf.reduce_mean(loss_fn_g(feature_real-feature_fake))/(480*480)
                        g_loss = g_loss1 + g_loss2
                else:
                    assert supervised == False

                if phase =='train':
                    grad = tape.gradient(g_loss, G.trainable_weights)
                    optimizer_g.apply_gradients(zip(grad, G.trainable_weights))
                    grad = tape.gradient(d_loss, D.trainable_weights)
                    optimizer_d.apply_gradients(zip(grad, D.trainable_weights))
                    del tape     
                else:
                    pass

                g_loss_avg[phase].append(g_loss)
                d_loss_avg[phase].append(d_loss)
        
        dataloader[phase].on_epoch_end()
        
        g_loss_dict[phase][epoch] = np.mean(g_loss_avg[phase])
        d_loss_dict[phase][epoch] = np.mean(g_loss_avg[phase])
        
        writer1.add_scalars('loss', {phase: g_loss_dict[phase][epoch]}, epoch)
        writer1.add_scalars('accuracy', {phase: d_loss_dict[phase][epoch]}, epoch)

        print(f'\n-----------{phase}-----------')
        print('Loss  =  {0:.3f}'.format(g_loss_dict[phase][epoch]))
        print('Acc   =  {0:.3f}'.format(d_loss_dict[phase][epoch]))

        plt.plot(range(len(g_loss_dict['train'][:epoch])), g_loss_dict['train'][:epoch], 'r')
        plt.plot(range(len(g_loss_dict['val'][:epoch])), g_loss_dict['val'][:epoch], 'b')
          
        plt.plot(range(len(d_loss_dict['train'][:epoch])), d_loss_dict['train'][:epoch], 'y')
        plt.plot(range(len(d_loss_dict['val'][:epoch])), d_loss_dict['val'][:epoch], 'g')
                 
        plt.legend(['G: train loss', 'G: val loss', 'D: train loss', 'D: val loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(f'/MULTIX/DATA/nccid/{G.model_name}_{D.model_name}_metrics_k{k}.png')
        
        g_loss_df = pd.DataFrame.from_dict(g_loss_dict)
        g_loss_df.to_csv(f'/MULTIX/DATA/nccid/{G.model_name}_loss_k{k}.csv')
        
        d_loss_df = pd.DataFrame.from_dict(d_loss_dict)
        d_loss_df.to_csv(f'/MULTIX/DATA/nccid/{D.model_name}_loss_k{k}.csv') 


    g_loss_dict = dict(itertools.islice(g_loss_dict.items(), end_epoch))
    d_loss_dict = dict(itertools.islice(d_loss_dict.items(), end_epoch))

    return g_loss_dict, d_loss_dict                          
                

                

    

class Generator(tl.models.Model):
    model_name = 'lung_gan_g'
    model_type = 'keras' # tensorlayer
    loss_fn = tf.nn.l2_loss()
    optimizer = Adam()

    def __init__(self, trial=None, is_train=True, reuse=False) -> None:
        super(Generator, self).__init__()
        self.image_size = (480,480,3)
        
        self.is_train = is_train
        self.reuse = reuse

        self.s2 = int(self.image_size//2)
        self.s4 = int(self.image_size//4)
        self.s8 = int(self.image_size//8)
        self.s16 = int(self.image_size//16)
        self.s32 = int(self.image_size//32)
        self.s64 = int(self.image_size//64)
        self.s128 = int(self.image_size//128)

        self.bs = 12
        self.k = 4

        self.gf_dim = 16
        self.w_init = tf.random_normal_initializer(stddev=0.02)
        self.gamma_init = tf.random_normal_initializer(1., 0.02)

        self.dense1 = Dense(n_units=(self.gf_dim * 64 * self.s128 * self.s128), W_init=self.w_init, act=tf.identity)
        self.reshape1 = Reshape(shape=[-1,self.s128, self.s128, self.gf_dim*64])
        self.batchnorm1 = BatchNorm2d(act=tf.nn.relu, is_train=self.is_train, gamma_init=self.gamma_init, name=None)

        self.deconv1 = DeConv2d(self.gf_dim * 32, (self.k, self.k), out_size=(self.s64, self.s64), strides=(2, 2), padding='SAME', 
        batch_size=self.bs, act=None, W_init=self.w_init)

        self.batchnorm2 = BatchNorm2d(act=tf.nn.relu, is_train=self.is_train, gamma_init=self.gamma_init, name=None)
        self.deconv2 = DeConv2d(self.gf_dim * 16, (self.k, self.k), out_size=(self.s32, self.s32), strides=(2, 2),
                    padding='SAME', batch_size=self.bs, act=None, W_init=self.w_init)
        self.batchnorm3 = BatchNorm2d(act=tf.nn.relu, is_train=self.is_train,gamma_init=self.gamma_init)
        self.deconv3 = DeConv2d(self.gf_dim*8, (self.k, self.k), out_size=(self.s16, self.s16), strides=(2, 2),
                    padding='SAME', batch_size=self.bs, act=None, W_init=self.w_init)
        self.batchnorm4 =BatchNorm2d(act=tf.nn.relu, is_train=self.is_train,gamma_init=self.gamma_init)
        self.deconv4 = DeConv2d(self.gf_dim*4, (self.k, self.k), out_size=(self.s8, self.s8), strides=(2, 2),
                    padding='SAME', batch_size=self.bs, act=None, W_init=self.w_init)
        self.batchnorm5 = BatchNorm2d(act=tf.nn.relu, is_train=self.is_train,gamma_init=self.gamma_init)
        self.deconv5 = DeConv2d(self.gf_dim*2, (self.k, self.k), out_size=(self.s4, self.s4), strides=(2, 2),
                    padding='SAME', batch_size=self.bs, act=None, W_init=self.w_init)
        self.batchnorm6 = BatchNorm2d(act=tf.nn.relu, is_train=self.is_train,gamma_init=self.gamma_init)
        self.deconv6 = DeConv2d(self.gf_dim*1, (self.k, self.k), out_size=(self.s2, self.s2), strides=(2, 2),
                    padding='SAME', batch_size=self.bs, act=None, W_init=self.w_init)
        self.batchnorm7 = BatchNorm2d(act=tf.nn.relu, is_train=self.is_train,gamma_init=self.gamma_init)
        self.deconv7 = DeConv2d(3, (self.k, self.k), out_size=(self.image_size, self.image_size), strides=(2, 2),
                    padding='SAME', batch_size=self.bs, act=None, W_init=self.w_init)
        self.tanh = tf.nn.tanh()

    def call(self, x):
        x = self.dense1(x)
        x = self.reshape1(x)

        x = self.batchnorm1(x)
        x = self.deconv1(x)

        x = self.batchnorm2(x)
        x = self.deconv2(x)

        x = self.batchnorm3(x)
        x = self.deconv3(x)

        x = self.batchnorm4(x)
        x = self.deconv4(x)

        x = self.batchnorm5(x)
        x = self.deconv5(x)

        x = self.batchnorm6(x)
        x = self.deconv6(x)

        x = self.batchnorm7(x)
        logits = self.deconv7(x)

        x = self.tanh(logits)

        return x, logits



class Discriminator(tl.models.Model):
    model_name = 'lung_gan_d'
    model_type = 'keras' # tensorlayer
    loss_fn = tf.nn.sigmoid_cross_entropy_with_logits()
    optimizer = Adam()

    def __init__(self, trial=None, is_train=True, reuse=False) -> None:
        super(Discriminator, self).__init__()
        self.image_size = (480,480,3)
        
        self.is_train = is_train
        self.reuse = reuse

        self.s2 = int(self.image_size//2)
        self.s4 = int(self.image_size//4)
        self.s8 = int(self.image_size//8)
        self.s16 = int(self.image_size//16)
        self.s32 = int(self.image_size//32)
        self.s64 = int(self.image_size//64)
        self.s128 = int(self.image_size//128)

        self.bs = 12
        self.k = 4

        self.gf_dim = 16
        self.w_init = tf.random_normal_initializer(stddev=0.02)
        self.gamma_init = tf.random_normal_initializer(1., 0.02)

        lrelu = lambda x : tf.nn.leaky_relu(x, 0.2)
        self.conv1 = Conv2d(16 (self.k, self.k), (2, 2), act=lrelu,padding='SAME', W_init=self.w_init)
        self.conv2 = Conv2d(self.df_dim, (self.k, self.k), (2, 2), act=None, padding='SAME', W_init=self.w_init)
        self.batchnorm1 = BatchNorm2d(act=lrelu,is_train=self.is_train, gamma_init=self.gamma_init)
        self.conv3 = Conv2d(self.df_dim*2, (self.k, self.k), (2, 2), act=None, padding='SAME', W_init=self.w_init)
        self.batchnorm2 = BatchNorm2d(act=lrelu,is_train=self.is_train, gamma_init=self.gamma_init)
        self.conv4 = Conv2d(self.df_dim*4, (self.k, self.k), (2, 2), act=None,padding='SAME', W_init=self.w_init)
        self.batchnorm3 = BatchNorm2d(act=lrelu,is_train=self.is_train, gamma_init=self.gamma_init)
        self.conv5 =  Conv2d(self.df_dim*8, (self.k, self.k), (2, 2), act=None, padding='SAME', W_init=self.w_init)
        self.batchnorm4 = BatchNorm2d(act=lrelu,is_train=self.is_train, gamma_init=self.gamma_init)

        self.maxpool1 = MaxPool2d(filter_size=(4, 4), strides=None, padding='SAME')
        self.flatten1 = Flatten()

        self.conv6 = Conv2d(self.df_dim*16, (self.k, self.k), (2, 2), act=None,padding='SAME', W_init=self.w_init)
        self.batchnorm5 = BatchNorm2d(act=lrelu,is_train=self.is_train, gamma_init=self.gamma_init)

        self.maxpool2 = MaxPool2d(filter_size=(2, 2), strides=None, padding='SAME')
        self.flatten2 = Flatten()

        self.conv7 = Conv2d(self.df_dim*32, (self.k, self.k), (2, 2), act=None,padding='SAME', W_init=self.w_init)
        self.batchnorm6 = BatchNorm2d(act=lrelu,is_train=self.is_train, gamma_init=self.gamma_init)
        self.flatten3 = Flatten()

        self.concat = Concat()
        self.dense1 = Dense(n_units=1, act=tf.identity, W_init=self.w_init)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batchnorm1(x)
        x = self.conv3(x)
        x = self.batchnorm2(x)
        x = self.conv4(x)
        x = self.batchnorm3(x)
        x = self.conv5(x)
        x - self.batchnorm4(x)
        
        globalmax1 = self.maxpool1(x)
        globalmax1 = self.flatten1(globalmax1)
    
        x = self.conv6(x)
        x = self.batchnorm5(x)

        globalmax2 = self.maxpool2(x)
        globalmax2 = self.flatten2(globalmax2)

        x = self.conv7(x)
        x = self.batchnorm6(x)
        globalmax3 = self.flatten3(x)

        feature = self.concat([globalmax1, globalmax2, globalmax3])
        logits = self.dense1(feature)
        x = tf.nn.sigmoid(logits)

        return x, logits, feature


class GAN(tl.models.Model):
    model_name = 'lung_gan'
    model_type = 'keras' # tensorlayer
    loss_fn = {'g': tf.nn.l2_loss(), 'd':tf.nn.sigmoid_cross_entropy_with_logits()}
    optimizer = Adam()

    def __init__(self, trial=None, is_train=True, reuse=False, model='g'):
        super(GAN, self).__init__()

        if model == 'g':
            self.model = Generator(trial=trial, is_train=is_train, reuse=reuse)
        elif model == 'd':
            self.model = Discriminator(trial=trial, is_train=is_train, reuse=reuse)
        else:
            raise ValueError("Model argument must be 'g' or 'b'")

    
    def call(self, x):
        return self.model(x)