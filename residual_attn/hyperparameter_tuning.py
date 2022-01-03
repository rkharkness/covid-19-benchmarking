import numpy as np
import pandas as pd
#from covid_caps import Capsule, squash, margin_loss, capsule_net
from res_attn import AttentionResNet56
from sklearn.model_selection import StratifiedKFold
from utils.tf_callbacks import recall_m, precision_m, f1_m
from utils.tf_dataloaders import make_generators
import argparse

from keras_tuner import HyperModel
import keras_tuner as kt
import tensorflow as tf
import keras


def model_builder(hp):
    regularization = hp.Choice(name ='regularization', values=[1e-2, 1e-3, 1e-4])
    learning_rate = hp.Choice(name='lr', values=[1e-3, 5e-4, 1e-4])

    input_size = 480
    n_classes= 2

    model = capsule_net(shape=input_size, n_classes=n_classes, regularization=regularization)
    model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), loss=margin_loss, metrics=['acc', f1_m, precision_m, recall_m])
    return model


if __name__ =="main":
    parser = argparse.ArgumentParser(description='RESIDUAL ATTN hyperparameter tuning')
    
    parser.add_argument('--epochs', default=5, type=int, help='Number of epochs')
    parser.add_argument('--bs', default=16, type=int, help='Batch size')
    parser.add_argument('--img_size', default=480, type=int, help='Image size')
    parser.add_argument('--img_channel', default=1, type=int, help='Image channel')
    parser.add_argument('--data_csv', default='', type=str, help='Path to data file')
    parser.add_argument('--save_dir', default='/MULTIX/DATA/HOME/covid-19-benchmarking/residual_attn/', type=str, help='Name of folder to store training checkpoints')

    args = parser.parse_args()

    # read in data
    total_data = pd.read_csv(args.data_csv)
    train_df = total_data[total_data['split']=='train']
    train_df = train_df.reset_index(drop=True)

    # kfold data split: training --> training + validation
    seed = 0
    np.random.seed(seed)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    target = train_df.finding

    # #hyperparameter tuning
    fold_no = 1
    for train_idx, val_idx in kfold.split(train_df, target):
        train_kfold = train_df.iloc[train_idx]
        val_kfold = train_df.iloc[val_idx]

        train_generator, val_generator = make_generators(train_kfold, val_kfold, batch_size=16, 
                target='finding',target_size=(args.img_size,args.img_size), directory=args.data_dir)
        STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
        STEP_SIZE_VALID=val_generator.n//val_generator.batch_size

        with tf.device('/device:GPU:0'):
            print('------------------------------------------------------------------------')
            print(f'Hyperparameter tuning for fold {fold_no} ...')
            tuner = kt.Hyperband(hypermodel=model_builder, objective = kt.Objective("val_loss", direction="min"),
                    max_epochs=5, project_name=f'res-attn-tuner')

            tuner.search(train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=val_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=5,
                    shuffle=True,
                    verbose=1,
                    initial_epoch=0,
                    use_multiprocessing=False,
                    workers=1)

            if fold_no ==1:
                break

            tuner.results_summary()
            best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
