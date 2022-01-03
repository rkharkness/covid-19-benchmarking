
from __future__ import print_function
import pandas as pd
import numpy as np

from keras import backend as K
from keras.models import Model
from keras.layers import *
import numpy as np
import keras
from keras.callbacks import ModelCheckpoint
from utils.tf_dataloaders import make_generators

from keras.regularizers import l2

import tensorflow as tf
import pandas as pd

from hyperparameter_tuning import tuner

from sklearn.model_selection import StratifiedKFold
from utils.tf_callbacks import recall_m, precision_m, f1_m, PlotLosses
from covid_caps import Capsule, squash, margin_loss, capsule_net

import argparse
K.set_image_data_format('channels_last')


# read in csv of image info
total_data = pd.read_csv('/MULTIX/DATA/HOME/custom_data_ablation_1500.csv')
total_data['pneumonia_binary'] = total_data['pneumonia_binary'].astype(str)

train_df = total_data[total_data['split']=='train']
train_df = train_df.reset_index(drop=True)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
assert tf.test.is_gpu_available()
assert tf.test.is_built_with_cuda()

seed = 0
np.random.seed(seed)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
target = train_df.finding

# Define COVID-CAPS
def capsule_net(shape, n_classes, regularizer):  
    input_image = Input(shape=(None, None, 3))
    x = Conv2D(64, (3, 3), activation='relu')(input_image)
    x=BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = AveragePooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)




    x = Reshape((-1, 128))(x)
    x = Capsule(32, 8, 3, True)(x)  
    x = Capsule(32, 8, 3, True)(x)   
    capsule = Capsule(2, 16, 3, True)(x)
    output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(capsule)

    model = Model(inputs=[input_image], outputs=[output])

    adam = optimizers.Adam(lr=0.001) 

    model.compile(loss=margin_loss, optimizer=adam, metrics=['accuracy'])

    return model

if __name__=="main":
    parser = argparse.ArgumentParser(description='COVID-CAPS hyperparameter tuning')
    
    parser.add_argument('--epochs', default=5, type=int, help='Number of epochs')
    parser.add_argument('--bs', default=16, type=int, help='Batch size')
    parser.add_argument('--dropout', default=0.2, type=float, help='Dropout rate')
    parser.add_argument('--img_size', default=480, type=int, help='Image size')
    parser.add_argument('--img_channel', default=1, type=int, help='Image channel')
    parser.add_argument('--data_csv', default='/MULTIX/DATA/INPUT/binary_data.csv', type=str, help='Path to data file')
    parser.add_argument('--save_dir', default='/home/ubuntu/', type=str, help='Name of folder to store training checkpoints i.e., /home/ubuntu/')
    parser.add_argument('--project_name', default='res_attn', type=str, help='Name of file to store training checkpoints i.e., res_attn')

    args = parser.parse_args()

    total_data = pd.read_csv(args.data_csv, dtype=str)
    train_df = total_data[total_data['split']=='train']
    train_df = train_df.reset_index(drop=True)

    seed = 0
    np.random.seed(seed)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    target = train_df.finding

    # build model with best hyperparameters
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)

    fold_no = 1
    for train_idx, val_idx in kfold.split(train_df, target):
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')

        callback = [checkpoint_weights, checkpoint_model, early_stopper, lr_reducer, plot_losses]

        train_kfold = train_df.iloc[train_idx]
        val_kfold = train_df.iloc[val_idx]

        train_generator, val_generator = make_generators(train_kfold, val_kfold, batch_size=24, 
                target='finding',target_size=(args.img_size,args.img_size), directory=args.data_dir)

        STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
        STEP_SIZE_VALID=val_generator.n//val_generator.batch_size

        # prepare usefull callbacks
        filepath = args.save_dir + args.project_name + '-' + str(fold_no) # dir/filename_fold-no

        checkpoint_weights = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        checkpoint_model = ModelCheckpoint(args.save_dir + f"{args.project_name}_{fold_no}", monitor='val_loss', verbose=1, save_best_only=True, mode='min',save_weights_only=False)
        early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=25, verbose=1)
        lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, min_lr=10e-9, epsilon=0.01, verbose=1)
        plot_losses = PlotLosses(args.save_dir + args.project_name, fold_no)

        with tf.device('/device:GPU:0'):
            H = model.fit_generator(
                generator=train_generator,
                steps_per_epoch=STEP_SIZE_TRAIN,
                epochs=epochs,
                validation_data=val_generator,
                validation_steps=STEP_SIZE_VALID,
                callbacks=callbacks_list)



