from models.coronet_tfl import CoroNet
from models.ecovnet import ECovNet
from models.fusenet import FuseNet
#from models.ac_covidnet import ACCovidNet
from models.res_attn import AttentionResNetModified
#from models.coronanet import CoronaNet
#from models.coronet import CoroNet
from models.siamese_net import SiameseNetwork
from models.mag_sd import MAG_SD, config, batch_augment
from models.capsnet import CovidCaps
import matplotlib.pyplot as plt
from models.xvitcos import xVitCOS
#from torchinfo import summary
#import traceback
from mpl_toolkits.axes_grid1 import ImageGrid


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
import tensorflow.keras.backend as K


def mag_sd_objective(lr, train_df, val_df, test_df, pretrained_weights=None):

    model = MAG_SD(config, lr).build_model()
    for k, val in model.items():
        print(k, val)

    supervised = model['supervised']
    optimizer = model['optimizer']
#    loss_fn = model['loss_fn']
    
    # make generators
    fold = 1
    params = {'batchsize':48, "num_workers":4, "k":fold}

    train_loader, val_loader, _ = make_generators(model, train_df, val_df, test_df, params)
    # create dict of dataloaders
    dataloader = {'train':train_loader, 'val':val_loader}

    best_val_loss = 1e10

    device = 'cuda'
    classifier = model['model']
    #classifier.to(device)

    supervised = model['supervised']
    print('supervised: ', supervised)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if (pretrained_weights):
        print(f'loading pretrained weights - {pretrained_weights}')
        classifier.load_state_dict(torch.load(pretrained_weights)) # load ae weights for coronet


    loss_dict = {'train': np.zeros(shape=(500,), dtype=np.float32),
                'val': np.zeros(shape=(500,), dtype=np.float32)}
    acc_dict = {'train': np.zeros(shape=(500,), dtype=np.float32),
                'val': np.zeros(shape=(500,), dtype=np.float32)}
    no_improvement = 0

    best_val_loss = 1e10
#
 #   loss_fn = model['loss_fn']

    for epoch in range(10):
        print(f'epoch - {epoch}')

        loss_avg = {'train':[],'val':[]}
        acc_avg = {'train':[], 'val':[]}
        for phase in ['train', 'val']:
#             if phase == 'train':
 #                model['model'].set_train()
  #           else:
   #              model['model'].set_eval()

             for batch in tqdm(dataloader[phase]):
                batch_x, batch_y = batch
                batch_x = batch_x.float().to('cuda')
                batch_y = batch_y.float().to('cuda')

                with torch.set_grad_enabled(phase == 'train'):
                     ### forward1
                    logit_raw, feature_1, attention_map = model['model'].encoder(batch_x)
                    ### batch augs
                    # mixup
                    mixup_images = batch_augment(batch_x, attention_map[:, 0:1, :, :], mode='mixup', theta=(0.4, 0.6), padding_ratio=0.1)
                    logit_mixup, _, _ = model['model'].encoder(mixup_images)
                    #
                    # # dropping
                    drop_images = batch_augment(batch_x, attention_map[:, 1:2, :, :], mode='dim', theta=(0.2, 0.5))
                    logit_dim, _, _ = model['model'].encoder(drop_images)
                    #
                    # ## patching
                    patch_images = batch_augment(batch_x, attention_map[:, 2:3, :, :], mode='patch', theta=(0.4, 0.6), padding_ratio=0.1)
                    logit_patch, _, _= model['model'].encoder(patch_images)

                    ### loss###
                    acc_raw, loss_raw = model['model'].compute_classification_loss(logit_raw, batch_y)
                    acc_mixup, loss_mixup = model['model'].compute_classification_loss(logit_mixup, batch_y)
                    acc_dim, loss_dim = model['model'].compute_classification_loss(logit_dim, batch_y)
                    acc_patch, loss_patch = model['model'].compute_classification_loss(logit_patch, batch_y)
                    # L2 loss
                    # loss = (loss_raw + loss_mixup + loss_dim + loss_patch)/4
                    # logit_patch = 0
                    # soft distance loss
                    loss = model['model'].gen_refine_loss(logit_raw, logit_mixup, logit_dim, logit_patch, batch_y)
                    variance = loss - loss_raw                   
                    if phase == 'train':
                        model['model'].optimizer.zero_grad()
                        loss.backward()
                        model['model'].optimizer.step()

                     #loss and top-3 acc
                    label_pids = model['model'].onehot_2_label(batch_y)
                    acc = model['model'].accuracy(logit_raw, label_pids, topk=(1,), cuda = False)

                acc_avg[phase].append(acc)
                loss_avg[phase].append(loss.item())
  
             loss_dict[phase][epoch] = np.mean(loss_avg[phase])

             if len(acc_avg[phase]) > 0:
                 acc_dict[phase][epoch] = np.mean(acc_avg[phase])
        
        print(f"val acc: {acc_dict['val'][epoch]}")
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

def objective(lr, model, train_df, val_df, test_df, pretrained_weights=None):
    for k, val in model.items():
        print(k, val)

    supervised = model['supervised']
    optimizer = model['optimizer']
    loss_fn = model['loss_fn']
    
    # make generators
    fold = 1
    params = {'batchsize':12, "num_workers":4, "k":fold}

    train_loader, val_loader, _ = make_generators(model, train_df, val_df, test_df, params)
    # create dict of dataloaders
    dataloader = {'train':train_loader, 'val':val_loader}

    best_val_loss = 1e10

    if model['model_type'] == 'keras':

#        # create dict of dataloaders
        x, y, _ = next(iter(train_loader))
        print(x.shape)
        print(y.shape)
        plt.figure()
        img = x.numpy().flatten()
        plt.hist(img, bins=20)
        plt.savefig('/MULTIX/DATA/INPUT_NCCID/pixel_hist1.png')

        plt.imshow(x[0].numpy())
        plt.savefig('/MULTIX/DATA/INPUT_NCCID/single_nccid_img.png')
 #      #x, y = next(iter(train_loader))
        fig = plt.figure(figsize=(12.,12.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                        nrows_ncols=(6,6),  # creates 2x2 grid of axes
                        axes_pad=0.1,  # pad between axes in inch.
                        )

        for ax, im in zip(grid, [x[i] for i in range(len(x)-1)]):
            # Iterating over the grid returns the Axes.

                ax.imshow(im)
        
        plt.savefig('/MULTIX/DATA/INPUT_NCCID/nccid_batch_img1.png')

        optimizer.learning_rate = lr
        loss_dict = {'train': [],'val': []}

        for epoch in range(10):
            print(f'epoch - {epoch}')
            loss_avg = {'train':[],'val':[]}
            for phase in ['train', 'val']:
                for batch in tqdm(dataloader[phase]):
                    if len(batch) > 1:
                        batch_x, batch_y, _ = batch # batch_y can be paired image
 #                       print(batch_x.shape)
#                        print(batch_y.shape)
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

               # dataloader[phase].on_epoch_end()
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

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

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

def weight_reset(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        m.reset_parameters()

def main(model, train_df, val_df, test_df, root, pretrained_weights=None):
    
    trial_lr = [1e-5,5e-5,1e-4,5e-4,1e-3,5e-3]
    all_best_vals = []
    for lr in trial_lr:
        if model['model_name']=='mag_sd':
            best_val = mag_sd_objective(lr, train_df, val_df, test_df, pretrained_weights=pretrained_weights)
        else:
            best_val = objective(lr, model, train_df, val_df, test_df, pretrained_weights=pretrained_weights)

            if model['model_type']=='keras':
                K.clear_session()
            else:
                model['model'].apply(weight_reset)

        print(f'trial {lr} completed - best val loss: {best_val}')
        all_best_vals.append(best_val)

    results = pd.DataFrame()
    results['trial_lr'] = trial_lr
    results['best_val_loss'] = all_best_vals
    if model['supervised']:
        results.to_csv(f"{root}/{model['model_name']}_supervised_hp.csv")
    else:
        results.to_csv(f"{root}/{model['model_name']}_unsupervised_hp.csv")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparameter tuning')
    parser.add_argument('--data_csv', default='/MULTIX/DATA/INPUT_NCCID/nccid_preprocessed14_21.csv', type=str, help='Path to data file')
    parser.add_argument('--root', default='/MULTIX/DATA/nccid', type=str, help='Path to save results')
    p
    args = parser.parse_args()

    tf.config.run_functions_eagerly(True)

    if 'chexpert' in args.data_csv:
        chexpert = True
    else:
        chexpert = False
    
    df = pd.read_csv(args.data_csv)
    
    if chexpert == True:
        train_df = df[df['split'] == "train"]
        train_df['kfold_1'] = 'train'
        val_df = df[df['split'] == "val"]
        val_df['kfold_1'] = 'val'
        test_df = val_df # not using
    else:
        df = df[df['xray_status']!=np.nan]
        df = df.dropna(subset=['xray_status'])
        
        train_df = df[df['kfold_1'] == "train"]
        val_df = df[df['kfold_1'] == "val"]
        test_df = df[df['kfold_1'] == 'test']

   # coronet = CoroNet(supervised=True, pretrained="/MULTIX/DATA/nccid/coronet_unsupervised_1.pth")
    #model = coronet.build_model()

   # mag_sd = MAG_SD(config)
   # model = mag_sd.build_model()

    model = xVitCOS().build_model()
    main(model, train_df, val_df, args.root, test_df) # pretrained_weights="/MULTIX/DATA/nccid/coronanet_unsupervised_1.pth")


    

