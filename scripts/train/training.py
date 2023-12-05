import os

import tensorflow as tf
from tensorflow import keras
from dataloaders import *

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import itertools

from models.ecovnet import ECovNet
from models.res_attn import AttentionResNetModified
from models.coronet import CoroNet
from models.coronet_tfl import CoroNet_Tfl
from models.coronet_tfl_seg import CoroNet_Tfl_Seg
from models.fusenet import FuseNet
from models.mag_sd import MAG_SD, config, batch_augment
from models.ssl_am import SSL_AM
from models.ssl_am_seg import SSL_AM_Seg
from models.xvitcos import xVitCOS
from models.xvitcos_seg import xVitCOS_Seg
from models.capsnet import CovidCaps
from models.covidnet import CovidNet

from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import tensorflow.keras.backend as K


def train_mag_sd(base, dataloader, k, root, patience=20):
    writer = SummaryWriter('mag_sd')
#
    loss_dict = {'train': np.zeros(shape=(500,), dtype=np.float32),
                'val': np.zeros(shape=(500,), dtype=np.float32)}
    
    acc_dict = {'train': np.zeros(shape=(500,), dtype=np.float32),
                'val':np.zeros(shape=(500,), dtype=np.float32)}

    no_improvement = 0

    optimizer = base['optimizer']
    lr = base['lr']
    end_epoch = 0

    supervised = base['supervised']
    best_val_loss = np.inf
    for epoch in range(500):
        end_epoch =+1
        loss_avg = {'train':[],'val':[]}
        acc_avg = {'train':[],'val':[]}

        for phase in ['train', 'val']:
             if phase == 'train':
                 base['model'].set_train()
             else:
                 base['model'].set_eval()

             for batch in tqdm(dataloader[phase]):
                batch_x, batch_y = batch
                batch_x = batch_x.float().to('cuda')
                batch_y = batch_y.float().to('cuda')

                with torch.set_grad_enabled(phase == 'train'):
                    ### forward1
                    logit_raw, feature_1,attention_map = base['model'].encoder(batch_x)

                    ### batch augs
                    # mixup
                    mixup_images = batch_augment(batch_x, attention_map[:, 0:1, :, :], mode='mixup', theta=(0.4, 0.6), padding_ratio=0.1)
                    logit_mixup, _, _ = base['model'].encoder(mixup_images)
                    #
                    # # dropping
                    drop_images = batch_augment(batch_x, attention_map[:, 1:2, :, :], mode='dim', theta=(0.2, 0.5))
                    logit_dim, _, _ = base['model'].encoder(drop_images)
                    #
                    # ## patching
                    patch_images = batch_augment(batch_x, attention_map[:, 2:3, :, :], mode='patch', theta=(0.4, 0.6), padding_ratio=0.1)
                    logit_patch, _, _= base['model'].encoder(patch_images)

                     ### loss###
                    acc_raw, loss_raw = base['model'].compute_classification_loss(logit_raw, batch_y)
                    acc_mixup, loss_mixup = base['model'].compute_classification_loss(logit_mixup, batch_y)
                    acc_dim, loss_dim = base['model'].compute_classification_loss(logit_dim, batch_y)
                    acc_patch, loss_patch = base['model'].compute_classification_loss(logit_patch, batch_y)
                    # L2 loss
                    # loss = (loss_raw + loss_mixup + loss_dim + loss_patch)/4
                    # logit_patch = 0

                    # soft distance loss
                    loss = base['model'].gen_refine_loss(logit_raw, logit_mixup, logit_dim, logit_patch, batch_y)
                    variance = loss - loss_raw

                    if phase == 'train':
                      base['model'].optimizer.zero_grad()
                      loss.backward()
                      base['model'].optimizer.step()

                    #loss and top-3 acc
                    label_pids = base['model'].onehot_2_label(batch_y)
                    acc = base['model'].accuracy(logit_raw, label_pids, topk=(1,), cuda = False)

                acc_avg[phase].append(acc)
                loss_avg[phase].append(loss.item())

             loss_dict[phase][epoch] = np.mean(loss_avg[phase])

             if len(acc_avg[phase]) > 0:
               acc_dict[phase][epoch] = np.mean(acc_avg[phase])

             writer.add_scalars('loss', {phase: loss_dict[phase][epoch]}, epoch)
             writer.add_scalars('accuracy', {phase: acc_dict[phase][epoch]}, epoch)

             print(f'-----------{phase}-----------')
             print('Loss  =  {0:.3f}'.format(loss_dict[phase][epoch]))
             print('Acc   =  {0:.3f}'.format(acc_dict[phase][epoch]))

        plt.plot(range(len(loss_dict['train'][:epoch])), loss_dict['train'][:epoch], 'r')
        plt.plot(range(len(loss_dict['val'][:epoch])), loss_dict['val'][:epoch], 'b')

        plt.plot(range(len(acc_dict['train'][:epoch])), acc_dict['train'][:epoch], 'y')
        plt.plot(range(len(acc_dict['val'][:epoch])), acc_dict['val'][:epoch], 'g')

        plt.legend(['Training Loss', 'Val Loss', 'Training Acc', 'Val Acc'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        if supervised: #/MULTIX/DATA/nccid
            metrics_savepath = f"{root}/{model['model_name']}_supervised_metrics_k{k}.png"
            acc_savepath = f"{root}/{model['model_name']}_supervised_acc_k{k}.csv"
            loss_savepath = f"{root}/{model['model_name']}_supervised_loss_k{k}.csv"
        else:
            metrics_savepath = f"{root}/{model['model_name']}_unsupervised_metrics_k{k}.png"
            acc_savepath = f"{root}/{model['model_name']}_unsupervised_acc_k{k}.csv"
            loss_savepath = f"{root}/{model['model_name']}_unsupervised_loss_k{k}.csv"

        plt.savefig(metrics_savepath)

        loss_df = pd.DataFrame.from_dict(loss_dict)
        loss_df.to_csv(loss_savepath)

        acc_df = pd.DataFrame.from_dict(acc_dict)
        acc_df.to_csv(acc_savepath)

        if loss_dict['val'][epoch] > best_val_loss:
                no_improvement += 1
                print(f"No improvement for {no_improvement}")

                if no_improvement % 5 == 0:
                    lr = lr*0.8
                    optimizer.param_groups[0]['lr'] = lr
                    print(f"Reducing lr to {lr}")

                if no_improvement == patience:
                    print(f"No improvement for {no_improvement}, early stopping at epoch {epoch}")
                    break
        else:
            no_improvement = 0
            print(f"saving model weights to {model['model_name']}_{k}.pth")
            model['model'].save_model(k)

            best_val_loss = loss_dict['val'][epoch]

    loss_dict = dict(itertools.islice(loss_dict.items(), end_epoch))
    acc_dict = dict(itertools.islice(acc_dict.items(), end_epoch))
    return loss_dict, acc_dict

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_pytorch(model, dataloader, k, root, patience=20, pretrained_weights=None):    
    assert model['model_type'] == 'pytorch'
    writer = SummaryWriter(model['model_name'])
    device = 'cuda'

    print(f"training {model['model_name']}...")
    print("with model spec & hyperparameters...")
    for key, value in model.items():
        if key in ['model', 'model_name']:
            pass
        else:
            print(key, value)

    classifier = model['model']
    classifier.to(device)

    supervised = model['supervised']
    print('supervised: ', supervised)

    loss_fn = model['loss_fn']

    if model['optimizer'] == 'adam':
      optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), lr=model['lr'])

    if (pretrained_weights):
        classifier.load_state_dict(torch.load(pretrained_weights)) # load ae weights for coronet

    loss_dict = {'train': np.zeros(shape=(500,), dtype=np.float32),
                'val': np.zeros(shape=(500,), dtype=np.float32)}
    
    acc_dict = {'train': np.zeros(shape=(500,), dtype=np.float32),
                'val':np.zeros(shape=(500,), dtype=np.float32)}

    no_improvement = 0
    lr = get_lr(optimizer)

    best_val_loss = np.inf

    end_epoch = 0
    for epoch in range(500):
        end_epoch =+1
        loss_avg = {'train':[],'val':[]}
        acc_avg = {'train':[],'val':[]}

        for phase in ['train', 'val']:
            if phase == 'train':
                classifier.train()
            else:
                classifier.eval()

            train_labels = []
            val_labels = []

            for batch in tqdm(dataloader[phase]):
                batch_x, batch_y = batch
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):

                    pred = classifier(batch_x)
                    if len(pred) == 2:
                        pred, pred_img = pred[0], pred[1] # image, class
                    if len(pred) == 3: # for unsup coronet
                        pred, pred_img, z = pred[0], pred[1],pred[2]
                        
                    if supervised == False:
                        loss = loss_fn(pred_img, pred) # if unsupervised (no label) - loss_fn input = image
    
                        if model['model_name'] == 'coronet':
                            assert len(pred[0]) > 2
                            assert all(batch_y.detach().cpu().numpy()==0.0) # double check only training encoder with 
                            pred_z = classifier(pred)
                            loss_z = loss_fn(pred_z[2], z)
                            loss = loss  + loss_z

                    else:
                        loss = loss_fn(pred, batch_y)
                        pred_binary = np.rint(pred.detach().cpu().numpy())   
                        acc = accuracy_score(batch_y.detach().cpu().numpy(), pred_binary)
                        acc_avg[phase].append(acc)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                        train_labels.append(np.array(batch_y.detach().cpu()))
                    
                    if phase == 'val':
                        val_labels.append(np.array(batch_y.detach().cpu()))
                    
                loss_avg[phase].append(loss.item())

            loss_dict[phase][epoch] = np.mean(loss_avg[phase])

            if len(acc_avg[phase]) > 0:
              acc_dict[phase][epoch] = np.mean(acc_avg[phase])

            writer.add_scalars('loss', {phase: loss_dict[phase][epoch]}, epoch)
            writer.add_scalars('accuracy', {phase: acc_dict[phase][epoch]}, epoch)

            print(f'-----------{phase}-----------')
            print('Loss  =  {0:.3f}'.format(loss_dict[phase][epoch]))
            print('Acc   =  {0:.3f}'.format(acc_dict[phase][epoch]))

            if phase == 'train':
                label_arr = np.array([item for sublist in train_labels for item in sublist])
                unique, counts = np.unique(label_arr, return_counts=True)
                print(unique, counts)
            else:
                label_arr = np.array([item for sublist in val_labels for item in sublist])
                unique, counts = np.unique(label_arr, return_counts=True)        
                print(unique, counts)
            
        plt.figure()
        plt.plot(range(len(loss_dict['train'][:epoch])), loss_dict['train'][:epoch], 'r')
        plt.plot(range(len(loss_dict['val'][:epoch])), loss_dict['val'][:epoch], 'b')
        
        plt.plot(range(len(acc_dict['train'][:epoch])), acc_dict['train'][:epoch], 'y')
        plt.plot(range(len(acc_dict['val'][:epoch])), acc_dict['val'][:epoch], 'g')
               
        plt.legend(['Training Loss', 'Val Loss', 'Training Acc', 'Val Acc'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        if supervised:
            metrics_savepath = f"{root}/{model['model_name']}_supervised_metrics_k{k}.png"
            acc_savepath = f"{root}/{model['model_name']}_supervised_acc_k{k}.csv"
            loss_savepath = f"{root}/{model['model_name']}_supervised_loss_k{k}.csv"
        else:
            metrics_savepath = f"{root}/{model['model_name']}_unsupervised_metrics_k{k}.png"
            acc_savepath = f"{root}/{model['model_name']}_unsupervised_acc_k{k}.csv"
            loss_savepath = f"{root}/{model['model_name']}_unsupervised_loss_k{k}.csv"
        
        plt.savefig(metrics_savepath)
        
        loss_df = pd.DataFrame.from_dict(loss_dict)
        loss_df.to_csv(loss_savepath)

        acc_df = pd.DataFrame.from_dict(acc_dict)
        acc_df.to_csv(acc_savepath)

        if loss_dict['val'][epoch] > best_val_loss:
                no_improvement += 1
                print(f"No improvement for {no_improvement}")

                if no_improvement % 5 == 0:
                    lr = lr*0.8
                    optimizer.param_groups[0]['lr'] = lr
                    print(f"Reducing lr to {lr}")

                if no_improvement == patience:
                    print(f"No improvement for {no_improvement}, early stopping at epoch {epoch}")
                    break
        else:
            no_improvement = 0
            print(f"saving model weights to {model['model_name']}_{k}.pth")
            if supervised:
                model_savepath = f"/MULTIX/DATA/nccid/{model['model_name']}_supervised_{k}.pth"
            else:
                model_savepath = f"/MULTIX/DATA/nccid/{model['model_name']}_unsupervised_{k}.pth"
            torch.save(classifier.state_dict(), model_savepath)
            best_val_loss = loss_dict['val'][epoch]
        
    loss_dict = dict(itertools.islice(loss_dict.items(), end_epoch))
    acc_dict = dict(itertools.islice(acc_dict.items(), end_epoch))
    return loss_dict, acc_dict


def train_keras(model, dataloader, k, root, patience=20, pretrained_weights=None):
    print(f"training {model['model_name']}...")
    print("with model spec & hyperparameters...")
    for key, value in model.items():
        if key in ['model', 'model_name']:
            pass
        else:
            print(key, value)

    supervised = model['supervised']

    writer = SummaryWriter(model['model_name'])

    if (pretrained_weights):
        model['model'].load_weights(pretrained_weights)

    loss_dict = {'train': np.zeros(shape=(500,), dtype=np.float32),
                'val': np.zeros(shape=(500,), dtype=np.float32)}
    
    acc_dict = {'train': np.zeros(shape=(500,), dtype=np.float32),
                'val':np.zeros(shape=(500,), dtype=np.float32)}

    no_improvement = 0

    optimizer = model['optimizer']
    optimizer.lr.assign(model['lr'])
    lr = optimizer.lr

    best_val_loss = np.inf

    loss_fn = model['loss_fn']

    end_epoch = 0

    plt.figure()
    train_paths = []
    val_paths = []
    for epoch in range(500):
        print(f"epoch - {epoch}")
        loss_avg = {'train':[],'val':[]}
        acc_avg = {'train':[],'val':[]}
        end_epoch =+1

        train_labels = []
        val_labels = []

        for phase in ['train', 'val']:
            counter = 0
            for batch in tqdm(dataloader[phase]):
                counter = counter + 1
                if model['model_name'] == 'siamese_net' and model['supervised']==False:
                     batch_x1,  batch_y1, path1 = batch[0]
                     batch_x2, batch_y2, path2 = batch[1]

                     batch_x1 = tf.convert_to_tensor(batch_x1)
                     batch_x2 = tf.convert_to_tensor(batch_x2)
                     batch_x = [batch_x1, batch_x2]
                     batch_y = tf.convert_to_tensor([1.0 if j == i else 0.0 for j,i in zip(batch_y1, batch_y2)]) 
                     path = list(path1.numpy())

                else:
                  batch_x, batch_y, path = batch

                with tf.GradientTape() as tape:
                    pred = model['model'](batch_x)
                    loss = loss_fn(batch_y, pred)

                ### more here for unsupervised approaches
                if phase == 'train':
                    grad = tape.gradient(loss, model['model'].trainable_variables)
                    optimizer.apply_gradients(zip(grad, model['model'].trainable_variables))
                    if counter == 1:
                        train_paths.append(path[0])

                    train_labels.append(np.array(batch_y))

                elif phase == 'val':
                    if counter == 1:
                        val_paths.append(path[0])
                    val_labels.append(np.array(batch_y))
                
                pred = np.rint(np.array(pred))
               
                acc = accuracy_score(batch_y, pred)
                acc = np.array(acc).flatten()
                acc_avg[phase].append(acc)

                loss_avg[phase].append(loss.numpy())
            
            loss_dict[phase][epoch] = np.mean(loss_avg[phase])
            acc_dict[phase][epoch] = np.mean(acc_avg[phase])

            writer.add_scalars('loss', {phase: loss_dict[phase][epoch]}, epoch)
            writer.add_scalars('accuracy', {phase: acc_dict[phase][epoch]}, epoch)

            print(f'\n-----------{phase}-----------')
            print('Loss  =  {0:.3f}'.format(loss_dict[phase][epoch]))
            print('Acc   =  {0:.3f}'.format(acc_dict[phase][epoch]))
            if phase == 'train':
                label_arr = np.array([item for sublist in train_labels for item in sublist])
                unique, counts = np.unique(label_arr, return_counts=True)
                print(unique, counts)
            else:
                label_arr = np.array([item for sublist in val_labels for item in sublist])
                unique, counts = np.unique(label_arr, return_counts=True)        
                print(unique, counts)

            if epoch == 1: # checking shuffles
                if phase == 'train':
                    print(train_paths[0])
                    print(train_paths[1])
                    assert len(train_paths) == epoch + 1
                    assert train_paths[0] != train_paths[1], f"No Shuffling during training!"
                elif phase == 'val':
                    assert len(val_paths) == epoch + 1
                    assert val_paths[0] == val_paths[1], f"Shuffling during validation!"

        plt.figure()
        plt.plot(range(len(loss_dict['train'][:epoch])), loss_dict['train'][:epoch], 'r')
        plt.plot(range(len(loss_dict['val'][:epoch])), loss_dict['val'][:epoch], 'b')
          
        plt.plot(range(len(acc_dict['train'][:epoch])), acc_dict['train'][:epoch], 'y')
        plt.plot(range(len(acc_dict['val'][:epoch])), acc_dict['val'][:epoch], 'g')
                 
        plt.legend(['Training Loss', 'Val Loss', 'Training Acc', 'Val Acc'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        if supervised:
            metrics_savepath = f"{root}/{model['model_name']}_supervised_metrics_k{k}.png"
            acc_savepath = f"{root}/{model['model_name']}_supervised_acc_k{k}.csv"
            loss_savepath = f"{root}/{model['model_name']}_supervised_loss_k{k}.csv"
        else:
            metrics_savepath = f"{root}/{model['model_name']}_unsupervised_metrics_k{k}.png"
            acc_savepath = f"{root}/{model['model_name']}_unsupervised_acc_k{k}.csv"
            loss_savepath = f"{root}/{model['model_name']}_unsupervised_loss_k{k}.csv"
        
        plt.savefig(metrics_savepath)
        
        loss_df = pd.DataFrame.from_dict(loss_dict)
        loss_df.to_csv(loss_savepath)

        acc_df = pd.DataFrame.from_dict(acc_dict)
        acc_df.to_csv(acc_savepath)

        if loss_dict['val'][epoch] >= best_val_loss:
                no_improvement += 1
                print(f"No improvement for {no_improvement}")

                if no_improvement % 5 == 0:
                    lr = lr*0.8
                    optimizer.lr.assign(lr)
                    print(f"Reducing lr to {lr}")
                    
                if no_improvement == patience:
                    print(f"No improvement for {no_improvement}, early stopping at epoch {epoch}")
                    break
        
        else:
            no_improvement = 0
            print(f"saving model weights to {model['model_name']}.h5")

            if supervised:
                model_savepath = f"{root}/{model['model_name']}_supervised_{k}.h5"
            else:
                model_savepath = f"{root}/{model['model_name']}_unsupervised_{k}.h5"

            model['model'].save_weights(model_savepath)

            best_val_loss = loss_dict['val'][epoch]
    
    loss_dict = dict(itertools.islice(loss_dict.items(), end_epoch))
    acc_dict = dict(itertools.islice(acc_dict.items(), end_epoch))
    return loss_dict, acc_dict



def training(model, dataloader, k, root, patience=20, pretrained_weights=None):
    if model['model_type'] == 'keras':
        loss, acc = train_keras(model, dataloader, k, patience, root, pretrained_weights=pretrained_weights)
    elif model['model_type'] == 'pytorch':
        if model['model_name'] == 'mag_sd':
            loss, acc = train_mag_sd(model, dataloader, k, root, patience)
        else:
            loss, acc = train_pytorch(model, dataloader, k, root, patience, pretrained_weights=pretrained_weights)

    return loss, acc
    

def main(model, fold, df, bs):

        train_df = df[df[f"kfold_{fold}"] == "train"]
        val_df = df[df[f"kfold_{fold}"] == "val"]
        test_df = df[df[f"kfold_{fold}"] == 'test']
        
        #make generators
        params = {"batchsize":bs, "num_workers":12, "k":fold}
        train_df = train_df.sample(frac=1)

        train_loader, val_loader, _ = make_generators(model, train_df, val_df, test_df, params)

        dataloaders = {'train':train_loader, 'val':val_loader}
        loss, acc = training(model, dataloaders, k=fold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training Script')
    parser.add_argument('--data_csv', default='/MULTIX/DATA/INPUT_NCCID/nccid_preprocessed14_21.csv', type=str, help='Path to data file')
    parser.add_argument('--chexpert', type=bool, default=False)
    parser.add_argument('--pretrained_weights', type=str, default=None)
    parser.add_argument('--model_name', type=str, default=None)
    args = parser.parse_args()

    assert args.model_name in ['coronet', 
                               'coronet_tfl', 
                               'coronet_tfl_seg'
                               'covidnet', 
                               'covidcaps', 
                               'fusenet', 
                               'mag_sd', 
                               'ssl_am', 
                               'ssl_am_seg',
                               'xvitcos', 
                               'xvitcos_seg', 
                               'ecovnet', 
                               'res_attn']

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    assert tf.test.is_gpu_available()
    assert tf.test.is_built_with_cuda()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'
    tf.config.run_functions_eagerly(True)

    df = pd.read_csv(args.data_csv)
    print(f'loading {args.data_csv} ...')

    if args.chexpert == True:
        train_df = df[df['split']=='train']
        train_df['kfold_1'] = 'train'
        val_df = df[df['split']=='val']
        val_df['kfold_1'] = 'val'
        test_df = val_df
        df = pd.concat([train_df,val_df,test_df])
        top_range = 2
    else:
        df = df[df['xray_status']!=np.nan]
        df = df.dropna(subset=['xray_status'])
        top_range = 6


    model_dict = {'coronet':CoroNet(supervised=args.supervised, pretrained=args.pretrained_weights),
                  'coronet_tfl':CoroNet_Tfl(),
                  'cornet_tfl_seg':CoroNet_Tfl_Seg(),
                  'covidnet':CovidNet(pretrained=args.pretrained_weights),
                  'capsnet':CovidCaps(supervised=args.supervised, pretrained=args.pretrained_weights),
                  'fusenet':FuseNet(),
                  'mag_sd':MAG_SD(config=config),
                  'ssl_am':SSL_AM(supervised=args.supervised, pretrained=args.pretrained_weights),
                  'ssl_am_seg':SSL_AM_Seg(supervised=args.supervised, pretrained=args.pretrained_weights),
                  'xvitcos':xVitCOS(pretrained=args.pretrained_weights),
                  'xvitcos_seg':xVitCOS_Seg(pretrained=args.pretrained_weights),
                  'ecovnet':ECovNet(),
                  'res_attn':AttentionResNetModified(),
                  }

    for fold in range(1,top_range):

        model = model_dict[args.model_name]

        if args.model_name == 'capsnet':
            model = model.build_model((480,480,3), batch_size=24)

        if args.model_name == "coronet":
            model = model.pretrained = f"/MULTIX/DATA/nccid/coronet_unsupervised_{fold}.pth"
            model = model.build_model() 

        else:
            model = AttentionResNetModified().build_model()

        main(model, fold, df, bs=24)
