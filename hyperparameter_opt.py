from pandas.core.base import DataError
import optuna
from optuna.trial import TrialState

import os
import argparse

import pandas as pd
import numpy as np

from dataloaders import make_generators

import torch
from torch import optim

import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from ecovnet.ecovnet import ECovNet

from residual_attn.res_attn import AttentionResNet56
import matplotlib.pyplot as plt
#from capsnet.covid_caps import CovidCaps

model = ECovNet()

def objective(trial, model=model, supervised=True):
    trial_lr = trial.suggest_categorical("lr", [1e-5, 5e-4, 1e-4, 5e-3, 1e-3])
   # trial_lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = model.optimizer

    optimizer.learning_rate = trial_lr
    loss_fn = model.loss_fn
    
    # make generators
    fold = 1
    params = {'batchsize':24, "num_workers":4, "k":fold}
    train_loader, val_loader, _ = make_generators('keras',train_df, val_df, test_df, params)
    # create dict of dataloaders
    dataloader = {'train':train_loader, 'val':val_loader}

    best_val_loss = np.inf

    if model.model_type == 'keras':
      loss_dict = {'train': [],'val': []}

      for epoch in range(10):
          print(f'epoch - {epoch}')
          loss_avg = {'train':[],'val':[]}
          for phase in ['train', 'val']:
              for batch in dataloader[phase]:
                  if len(batch) > 1:
                      batch_x, batch_y = batch # batch_y can be paired image
                      with tf.GradientTape() as tape:
                          pred = model(batch_x)
                          loss = loss_fn(batch_y, pred)
                  else:
                      assert supervised == False
                      ### more here for unsupervised approaches
                  if phase == 'train':
                      grad = tape.gradient(loss, model.trainable_variables)
                      optimizer.apply_gradients(zip(grad, model.trainable_variables))
                  else:
                      pred = model.call(batch_x)
                      loss = loss_fn(batch_y, pred)

                  loss_avg[phase].append(loss)

              dataloader[phase].on_epoch_end()
              loss_dict[phase].append(np.mean(loss_avg[phase]))
          
          plt.plot(range(len(loss_dict['train'][:epoch])), loss_dict['train'][:epoch], 'r')
          plt.plot(range(len(loss_dict['val'][:epoch])), loss_dict['val'][:epoch], 'b')

          plt.legend(['Training Loss', 'Val Loss'])
          plt.xlabel('Epoch')
          plt.ylabel('Loss')
          plt.savefig(f'/MULTIX/DATA/nccid/{model.model_name}_lr_{trial_lr}_optuna metrics_epoch.png')

          if loss_dict['val'][epoch] < best_val_loss:
              best_val_loss = loss_dict['val'][epoch]
              print(f'best val loss: {best_val_loss}')
             
      metric_df = pd.DataFrame.from_dict(loss_dict)
      metric_df.to_csv(f'/MULTIX/DATA/nccid/{model.model_name}_optuna_metrics_lr_{trial_lr}.csv')
      
      return best_val_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparameter tuning')
    parser.add_argument('--data_csv', default='/MULTIX/DATA/nccid/nccid_preprocessed.csv', type=str, help='Path to data file')
   # parser.add_argument('--save_dir', type=str)
  #  parser.add_argument('--model_name', type=str)
    args = parser.parse_args()
    
    df = pd.read_csv(args.data_csv)
    
    mapping = {'negative':0, 'positive':1}
    
    df = df[df['xray_status']!=np.nan]
    df = df.dropna(subset=['xray_status'])
    
    df['xray_status'] = df['xray_status'].map(mapping)
    
    train_df = df[df['kfold_1'] == "train"]
    val_df = df[df['kfold_1'] == "val"]
    test_df = df[df['kfold_1'] == 'test']

    study = optuna.create_study(study_name=model.model_name, direction="minimize") # minimize for val loss
    study.optimize(objective, n_trials=10)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print(" Saving best params...")
    hp_df = study.trials_dataframe()
    hp_df.to_csv('/MULTIX/DATA/nccid/'+ model.model_name + '_optuna' + '.csv')