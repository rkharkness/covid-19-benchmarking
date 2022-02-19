#from models.coronet_tfl import CoroNet
#from models.ecovnet import ECovNet
#from models.ac_covidnet import ACCovidNet
#from models.res_attn import AttentionResNetModified
from models.coronanet import CoronaNet
from models.coronet import CoroNet
from models.siamese_net import SiameseNetwork
import matplotlib.pyplot as plt
from torchinfo import summary

import pandas as pd
import numpy as np
from tqdm import tqdm
from dataloaders import make_generators

import argparse

import torch
from torch import optim

import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


def objective(lr, model, train_df, val_df, test_df, pretrained_weights=None):
    for k, val in model.items():
        print(k, val)

    supervised = model['supervised']
    optimizer = model['optimizer']
    loss_fn = model['loss_fn']
    
    # make generators
    fold = 1
    params = {'batchsize':24, "num_workers":8, "k":fold}

    train_loader, val_loader, _ = make_generators(model, train_df, val_df, test_df, params)
    # create dict of dataloaders
    dataloader = {'train':train_loader, 'val':val_loader}

    best_val_loss = 1e10

    if model['model_type'] == 'keras':
      optimizer.learning_rate =lr
      loss_dict = {'train': [],'val': []}


      for epoch in range(10):
          print(f'epoch - {epoch}')
          loss_avg = {'train':[],'val':[]}
          for phase in ['train', 'val']:
              for batch in dataloader[phase]:
                  if len(batch) > 1:
                      batch_x, batch_y = batch # batch_y can be paired image
                      with tf.GradientTape() as tape:
                          pred = model['model'](batch_x)
                          loss = loss_fn(batch_y, pred)
                         
                  else:
                      assert supervised == False
                      ### more here for unsupervised approaches
                  if phase == 'train':
                      grad = tape.gradient(loss, model['model'].trainable_variables)
                      optimizer.apply_gradients(zip(grad, model['model'].trainable_variables))                  

                  loss_avg[phase].append(loss)
                 
              dataloader[phase].on_epoch_end()
              loss_dict[phase].append(np.mean(loss_avg[phase]))

          plt.plot(range(len(loss_dict['train'][:epoch])), loss_dict['train'][:epoch], 'r')
          plt.plot(range(len(loss_dict['val'][:epoch])), loss_dict['val'][:epoch], 'b')

          plt.legend(['Training Loss', 'Val Loss'])
          plt.xlabel('Epoch')
          plt.ylabel('Loss')
          plt.savefig(f"/MULTIX/DATA/nccid/{model['model_name']}_lr_{lr}_hp metrics_epoch.png")          
          
          if loss_dict['val'][epoch] < best_val_loss:
              best_val_loss = loss_dict['val'][epoch]
              print(f'best val loss: {best_val_loss}')
          else:
              print(f"val loss - {loss_dict['val'][epoch]}")

      metric_df = pd.DataFrame.from_dict(loss_dict)

      if supervised == True:
          save_path = f"/MULTIX/DATA/nccid/{model['model_name']}_supervised_hp_metrics_lr_{lr}.csv"
      else:
          save_path = f"/MULTIX/DATA/nccid/{model['model_name']}_unsupervised_hp_metrics_lr_{lr}.csv"

      metric_df.to_csv(save_path)

      return best_val_loss

    elif model['model_type'] == 'pytorch':
        device = 'cuda'
        classifier = model['model']
        classifier.to(device)

        supervised = model['supervised']
        print('supervised: ', supervised)

        if model['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
            print('adam', lr)

        if (pretrained_weights):
            print(f'loading pretrained weights - {pretrained_weights}')
            classifier.load_state_dict(torch.load(pretrained_weights)) # load ae weights for coronet

        loss_dict = {'train': np.zeros(shape=(500,), dtype=np.float32),
                    'val': np.zeros(shape=(500,), dtype=np.float32)}
        
        no_improvement = 0
     #   lr = get_lr(optimizer)

        best_val_loss = 1e10

        loss_fn = model['loss_fn']

        for epoch in range(10):
            print(f'epoch - {epoch}')
            
            loss_avg = {'train':[],'val':[]}

            for phase in ['train', 'val']:
                if phase == 'train':
                    classifier.train()
                else:
                    classifier.eval()

                for batch in tqdm(dataloader[phase]):
                    batch_x, batch_y = batch
                    batch_y = batch_y.to(device)
                    batch_x = batch_x.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        pred = classifier(batch_x)

                        if len(pred) == 2:
                            pred, pred_img = pred[0], pred[1] # image, class

                        if len(pred) == 3:
                            pred, pred_img, z = pred[0], pred[1],pred[2]
                            
                        if supervised == False:
                            loss = loss_fn(pred_img, batch_x) # if unsupervised (no label) - loss_fn input = image
                            
                            if model['model_name'] == 'coronet':
                                assert len(pred[0]) > 2
                                assert all(batch_y.detach().cpu().numpy()==0.0) # double check only training encoder with 
                                pred_z = classifier(pred)
                                loss_z = loss_fn(pred_z[2], z)
                                loss = loss  + loss_z
                                
                        else:
                            loss = loss_fn(batch_y.long(),pred) # if unsupervised (no label) - loss_fn input = class pred

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                        
                    loss_avg[phase].append(loss.item())
                
                loss_dict[phase][epoch] = np.mean(loss_avg[phase])
        
            plt.plot(range(len(loss_dict['train'][:epoch])), loss_dict['train'][:epoch], 'r')
            plt.plot(range(len(loss_dict['val'][:epoch])), loss_dict['val'][:epoch], 'b')

            plt.legend(['Training Loss', 'Val Loss'])
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.savefig(f"/MULTIX/DATA/nccid/{model['model_name']}_lr_{lr}_hp metrics_epoch.png")          
            
            if loss_dict['val'][epoch] < best_val_loss:
                best_val_loss = loss_dict['val'][epoch]
                print(f'best val loss: {best_val_loss}')
            else:
                print(f"val loss - {loss_dict['val'][epoch]}")

        metric_df = pd.DataFrame.from_dict(loss_dict)

        if supervised == True:
            save_path = f"/MULTIX/DATA/nccid/{model['model_name']}_supervised_hp_metrics_lr_{lr}.csv"
        else:
            save_path = f"/MULTIX/DATA/nccid/{model['model_name']}_unsupervised_hp_metrics_lr_{lr}.csv"

        metric_df.to_csv(save_path)

        return best_val_loss


def main(model, train_df, val_df, test_df, pretrained_weights=None):
    
    trial_lr = [1e-6,5e-5,1e-5,5e-4,1e-4,5e-3,1e-3]
    all_best_vals = []
    for lr in trial_lr:
        best_val = objective(lr, model, train_df, val_df, test_df, pretrained_weights=pretrained_weights)
        all_best_vals.append(best_val)

        print(f'trial {lr} completed - best val loss: {best_val}')

    results = pd.DataFrame()
    results['trial_lr'] = trial_lr
    results['best_val_loss'] = all_best_vals
    if model['supervised']:
        results.to_csv(f"/MULTIX/DATA/nccid/{model['model_name']}_supervised_hp.csv")
    else:
        results.to_csv(f"/MULTIX/DATA/nccid/{model['model_name']}_unsupervised_hp.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparameter tuning')
    parser.add_argument('--data_csv', default='/MULTIX/DATA/nccid_preprocessed.csv', type=str, help='Path to data file')
    args = parser.parse_args()
    
    df = pd.read_csv(args.data_csv)
    
    mapping = {'negative':0, 'positive':1}
    df = df[df['xray_status']!=np.nan]
    df = df.dropna(subset=['xray_status'])
    
    df['xray_status'] = df['xray_status'].map(mapping)

    train_df = df[df['kfold_1'] == "train"]
    val_df = df[df['kfold_1'] == "val"]
    test_df = df[df['kfold_1'] == 'test']

    siamese_net = SiameseNetwork()
    model = siamese_net.build_model()
    summary(model['model'])    
    main(model, train_df, val_df, test_df) #, pretrained_weights="/MULTIX/DATA/nccid/coronanet_unsupervised_1.pth")


    
