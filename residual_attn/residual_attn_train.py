# from __future__ import print_function

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import argparse

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

import tensorflow.keras as keras

from tensorflow.keras.models import load_model

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from sklearn.utils import class_weight, resample
import os

from utils.tf_callbacks import recall_m, precision_m, f1_m, PlotLosses

from keras_tuner import HyperModel
import keras_tuner as kt

from res_attn import AttentionResNet56

# update to be a shared hp ?
transforms = A.Compose([
                A.VerticalFlip(p=0.5),              
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                # A.Affine(translate_percent=10,p=0.5),
                A.CLAHE(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),    
                A.RandomGamma(p=0.5),
    ])

def aug_fn(image):
    data = {"image":image}
    aug_data = transforms(**data)
    aug_img = aug_data["image"]
    aug_img= tf.cast(aug_img/255.0, tf.float32)
    return image

def process_data(image, label, img_size):
    aug_img = tf.numpy_function(func=aug_fn, inp=[image, img_size], Tout=tf.float32)
    return aug_img, label

# create dataset
ds_alb = data.map(partial(process_data, img_size=120),
                  num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

print(ds_alb)
def make_generators(train_df, val_df, batch_size, target, target_size, directory, transofmrs =transforms):

        train_datagen=ImageDataGenerator(
        #    horizontal_flip=True,
        #    rotation_range=20,
        #    width_shift_range=0.2,
        #    height_shift_range=0.2,
        #    zoom_range=0.2,
            preprocessing_function=transform,
            rescale=1/255.0)

        test_datagen=ImageDataGenerator(
            # preprocessing_function=transform,
            rescale=1/255.0)

        train_generator=train_datagen.flow_from_dataframe(dataframe=train_df, directory=directory,
                                                    x_col="structured_path", y_col=target, class_mode="categorical", target_size=target_size, color_mode='grayscale',
                                                    batch_size=batch_size)


        val_generator=test_datagen.flow_from_dataframe(dataframe=val_df, directory=directory,
                                                    x_col="structured_path", y_col=target, class_mode="categorical", target_size=target_size, color_mode='grayscale',
                                                    batch_size=batch_size)

        return train_generator, val_generator


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Residual Attention Net Training Script')

    parser.add_argument('--epochs', default=1000, type=int, help='Number of epochs')
   # parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--bs', default=16, type=int, help='Batch size')
   # parser.add_argument('--dropout', default=0.5, type=float, help='Dropout rate')
   # parser.add_argument('--img_size', default=480, type=int, help='Image size')
   # parser.add_argument('--img_channel', default=1, type=int, help='Image channel')
    parser.add_argument('--data_csv', default='/MULTIX/DATA/INPUT/binary_data.csv', type=str, help='Path to data file')
    parser.add_argument('--save_dir', default='/home/ubuntu/', type=str, help='Name of folder to store training checkpoints')
    parser.add_argument('--data_dir', default='/MULTIX/DATA/INPUT/binary_data/', type=str, help='Path to data folder')
    parser.add_argument('--savefile', default='residual_attn', help='Filename for saved weights')

    args = parser.parse_args()

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    assert tf.test.is_gpu_available()
    assert tf.test.is_built_with_cuda()
    
    #def get_model_name(k):
     #   return args.savefile + str(k)+'.h5'

    total_data = pd.read_csv(args.data_csv, dtype=str)



   # train_df = total_data[total_data['split']=='train']
    #train_df = train_df.reset_index(drop=True)

    #seed = 0
    #np.random.seed(seed)
    #kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    #target = train_df.finding

    # Hyperparameter tuning

    # fold_no = 1
    # for train_idx, val_idx in kfold.split(train_df, target):
    #     train_kfold = train_df.iloc[train_idx]
    #     val_kfold = train_df.iloc[val_idx]

    #     train_generator, val_generator = make_generators(train_kfold, val_kfold, batch_size=16, 
    #         target='finding',target_size=(args.img_size,args.img_size), directory=args.data_dir)
    #     STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    #     STEP_SIZE_VALID=val_generator.n//val_generator.batch_size

    #     with tf.device('/device:GPU:0'):
    #         print('------------------------------------------------------------------------')
    #         print(f'Hyperparameter tuning for fold {fold_no} ...')
    #         tuner = kt.Hyperband(hypermodel=model_builder, objective = kt.Objective("val_loss", direction="min"),
    #             max_epochs=5, project_name=f'res_attn_tuner_new')

    #         tuner.search(train_generator,
    #             steps_per_epoch=STEP_SIZE_TRAIN,
    #             validation_data=val_generator,
    #             validation_steps=STEP_SIZE_VALID,
    #             epochs=5,
    #             shuffle=True,
    #             verbose=1,
    #             initial_epoch=0,
    #             # callbacks=[ClearTrainingOutput()],
    #             use_multiprocessing=False,
    #             workers=1)

    #         if fold_no ==1:
    #             break

   # tuner.results_summary()
   # best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

    # tuner = kt.Hyperband(hypermodel=model_builder, objective = kt.Objective("val_loss", direction="min"), 
    #     max_epochs = 10, project_name=f'res_attn_tuner')
    # tuner.results_summary()

    

    cv_accuracy = []
    cv_loss = []

    fold_no = 1
    for train_idx, val_idx in kfold.split(train_df, target):
        train_kfold = train_df.iloc[train_idx]
        val_kfold = train_df.iloc[val_idx]

        train_generator, val_generator = make_generators(train_kfold, val_kfold, batch_size=24, 
            target='finding',target_size=(args.img_size,args.img_size), directory=args.data_dir)
        STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
        STEP_SIZE_VALID=val_generator.n//val_generator.batch_size

            # prepare usefull callbacks
        filepath = args.save_dir+get_model_name(fold_no)

        checkpoint_weights = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        checkpoint_model = ModelCheckpoint(args.save_dir + f"res_attn_model_{fold_no}", monitor='val_loss', verbose=1, save_best_only=True, mode='min',save_weights_only=False)
        early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=25, verbose=1)
        lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, min_lr=10e-9, epsilon=0.01, verbose=1)
        plot_losses = PlotLosses('/home/ubuntu/res-attn-', fold_no)

        with tf.device('/device:GPU:0'):

            print('------------------------------------------------------------------------')
            print(f'Training for fold {fold_no} ...')

            callback = [checkpoint_weights, checkpoint_model, early_stopper, lr_reducer, plot_losses]

            model = tuner.hypermodel.build(best_hps)

            history = model.fit(train_generator, steps_per_epoch=STEP_SIZE_TRAIN, epochs=args.epochs, validation_data=val_generator, 
                validation_steps=STEP_SIZE_VALID, callbacks=callback)

            print(f"Loading weights from {filepath}")
            model.load_weights(filepath)
            scores = model.evaluate(val_generator, steps=STEP_SIZE_VALID)
            print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')

            cv_accuracy.append(scores[1]*100)
            cv_loss.append(scores[0])
            # keras.backend.clear_session()

            fold_no = fold_no + 1

            break


    # == Provide average scores ==
    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(cv_accuracy)):
      print('------------------------------------------------------------------------')
      print(f'> Fold {i+1} - Loss: {cv_loss[i]} - Accuracy: {cv_accuracy[i]}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(cv_accuracy)} (+- {np.std(cv_accuracy)})')
    print(f'> Loss: {np.mean(cv_loss)}')
    print('------------------------------------------------------------------------')



