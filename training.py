from dataloaders import *
import tensorflow as tf
import torch
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import itertools
from ecovnet.ecovnet import ECovNet
from residual_attn.res_attn import AttentionResNet56
from torch.utils.tensorboard import SummaryWriter

import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


## checklist
######################
# - log training data
# - plot losses
# - cv
# - reduce on plateau
# - early stopping
# - save best weights
# - for tf, fastai and pytorch
# - import dataloaders


def train_mag_sd(model, model_name, dataloader, optimizer, loss_fn, patience, supervised=True, pretrained_weights=None):
    writer = SummaryWriter(model_name)

    if (pretrained_weights):
        model.load_state_dict(torch.load(pretrained_weights))

    loss_dict = {'train': np.zeros(shape=(500,), dtype=np.float32),
                'val': np.zeros(shape=(500,), dtype=np.float32)}
    
    acc_dict = {'train': np.zeros(shape=(500,), dtype=np.float32),
                'val':np.zeros(shape=(500,), dtype=np.float32)}

    no_improvement = 0
    lr = optimizer.lr
    end_epoch = 0

    for epoch in range(500):
        end_epoch =+1
        loss_avg = {'train':[],'val':[]}
        acc_avg = {'train':[],'val':[]}

        for phase in ['train', 'val']:
            if phase == 'train':
                model.set_train()
            else:
                model.set_eval()

            for batch in tqdm(range(len(dataloader[phase]))):
                if len(batch) > 1:
                    batch_x, batch_y = batch
                    batch_x.to(device)
                    batch_y.to(device)
                
                else:
                    batch_x = batch # assume image input
                    batch_x.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    pred, _ = model.encoder(batch_x)
                    acc, loss = model.compute_classification_loss(pred, batch_y)

                    loss.backward()
                    optimizer.step()
                
                loss_avg[phase].append(loss.item())
                acc_avg[phase].append(acc[0])

            loss_dict[phase][epoch] = np.mean(loss_avg[phase])
            acc_dict[phase][epoch] = np.mean(acc_avg[phase])

            writer.add_scalars('loss', {phase: loss_dict[phase][epoch]})
            writer.add_scalars('accuracy', {phase: acc_dict[phase][epoch]})

            print(f'-----------{phase}-----------')
            print('Loss  =  {0:.3f}'.format(loss_dict[phase][epoch]))
            print('Acc   =  {0:.3f}'.format(acc_dict[phase][epoch]))

        if loss_avg['val'][epoch] > loss_avg['train'][epoch]:
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
            print(f'saving model weights to {model_name}.h5')
            model.save_model()

    loss_dict = dict(itertools.islice(loss_dict.items(), end_epoch))
    acc_dict = dict(itertools.islice(acc_dict.items(), end_epoch))
    return loss_dict, acc_dict


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_pytorch(model, dataloader, k, patience=20, pretrained_weights=None):    
    assert model.model_type == 'pytorch'
    writer = SummaryWriter(model.model_name)
    
    device = 'cuda'
    model = model.to(device)

    supervised = model.supervised
    print('supervised: ', supervised)

    if model.optimizer == 'adam':
      optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    if (pretrained_weights):
        model.load_state_dict(torch.load(pretrained_weights)) # load ae weights for coronet

    loss_dict = {'train': np.zeros(shape=(500,), dtype=np.float32),
                'val': np.zeros(shape=(500,), dtype=np.float32)}
    
    acc_dict = {'train': np.zeros(shape=(500,), dtype=np.float32),
                'val':np.zeros(shape=(500,), dtype=np.float32)}

    no_improvement = 0
    lr = get_lr(optimizer)

    best_val_loss = np.inf

    loss_avg = {'train':[],'val':[]}
    acc_avg = {'train':[],'val':[]}

    end_epoch = 0
    for epoch in range(500):
        end_epoch =+1
        #loss_avg = {'train':[],'val':[]}
       # acc_avg = {'train':[],'val':[]}

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            for batch in tqdm(dataloader[phase]):
                batch_x, batch_y = batch
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    pred = model(batch_x)
                    if len(pred) == 2:
                        pred, pred_img = pred[0], pred[1] # image, class
                    if len(pred) == 3:
                        pred, pred_img, z = pred[0], pred[1],pred[3]
                        
                    if supervised == False:
                        loss_fn = model.loss_fn['ae']
                        loss = loss_fn(pred_img, batch_x) # if unsupervised (no label) - loss_fn input = image
                        
                        if model.model_name == 'coronet':
                            assert len(pred) > 2
                            assert all(batch_y.detach.cpu().numpy()==0.0)
                            pred_z = model(pred)
                            loss_z = loss_fn(pred_z, z)
                            loss = loss  + loss_z
                    else:
                        loss_fn = model.loss_fn['classifier']
                        loss = loss_fn(pred, batch_y.long()) # if unsupervised (no label) - loss_fn input = class pred
                        
                        pred_binary = [1 * (x[0].cpu().numpy() >=0.5) for x in pred]
                        acc = accuracy_score(batch_y.detach().cpu().numpy(), pred_binary)
                        acc_avg[phase].append(acc)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                loss_avg[phase].append(loss.item())

            loss_dict[phase][epoch] = np.mean(loss_avg[phase])

            if len(acc_avg[phase]) > 0:
              acc_dict[phase][epoch] = np.mean(acc_avg[phase])

            writer.add_scalars('loss', {phase: loss_dict[phase][epoch]}, epoch)
            writer.add_scalars('accuracy', {phase: acc_dict[phase][epoch]}, epoch)

            print(f'-----------{phase}-----------')
            print('Loss  =  {0:.3f}'.format(loss_dict[phase][epoch]))
            print('Acc   =  {0:.3f}'.format(acc_dict[phase][epoch]))
            
        plt.plot(len(loss_dict['train'][:epoch]), loss_dict['train'][:epoch], 'r')
        plt.plot(len(loss_dict['val'][:epoch]), loss_dict['val'][:epoch], 'b')
        
        plt.plot(len(acc_dict['train'][:epoch]), acc_dict['train'][:epoch], 'y')
        plt.plot(len(acc_dict['val'][:epoch]), acc_dict['val'][:epoch], 'g')
               
        plt.legend(['Training Loss', 'Val Loss', 'Training Acc', 'Val Acc'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(f'/MULTIX/DATA/nccid/{model.model_name}_metrics_epoch_k{k}.png')
        
        metric_df = pd.DataFrame.from_dict(loss_dict, orient="index")
        metric_df.to_csv(f'/MULTIX/DATA/nccid/{model.model_name}_metrics_epoch_k{k}.csv')


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
            print(f'saving model weights to {model.model_name}_{k}.pth')
            torch.save(model.state_dict(), model.model_name + '_' + str(k) +'.pth')
            best_val_loss = loss_dict['val'][epoch]
        
    loss_dict = dict(itertools.islice(loss_dict.items(), end_epoch))
    acc_dict = dict(itertools.islice(acc_dict.items(), end_epoch))
    return loss_dict, acc_dict


def train_keras(model, dataloader, k, patience=20, supervised=True, pretrained_weights=None):
    # supervised included self-supervised methods i.e. two inputs

    writer = SummaryWriter(model.model_name)

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    loss_dict = {'train': np.zeros(shape=(500,), dtype=np.float32),
                'val': np.zeros(shape=(500,), dtype=np.float32)}
    
    acc_dict = {'train': np.zeros(shape=(500,), dtype=np.float32),
                'val':np.zeros(shape=(500,), dtype=np.float32)}

    no_improvement = 0
    # for colab testing - before hp tuning
    optimizer = model.optimizer
    optimizer.lr.assign(1e-4)
    lr = optimizer.lr

    best_val_loss = np.inf

    loss_fn = model.loss_fn

    end_epoch = 0
    
    plt.figure()
    for epoch in range(500):
        loss_avg = {'train':[],'val':[]}
        acc_avg = {'train':[],'val':[]}
        end_epoch =+1
        for phase in ['train', 'val']:
            for batch in tqdm(dataloader[phase]):
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
                
                if len(batch_y[0]) == len(batch_x[0]): # check if batch_y is an image
                    pass
                    
                else:
                    pred = [1 * (x[0].numpy() >= 0.5) for x in pred]
                    acc = accuracy_score(batch_y, pred)
                    acc_avg[phase].append(acc)

                
                loss_avg[phase].append(loss.numpy())

            dataloader[phase].on_epoch_end()
            
            loss_dict[phase][epoch] = np.mean(loss_avg[phase])
            acc_dict[phase][epoch] = np.mean(acc_avg[phase])

            writer.add_scalars('loss', {phase: loss_dict[phase][epoch]}, epoch)
            writer.add_scalars('accuracy', {phase: acc_dict[phase][epoch]}, epoch)

            print(f'\n-----------{phase}-----------')
            print('Loss  =  {0:.3f}'.format(loss_dict[phase][epoch]))
            print('Acc   =  {0:.3f}'.format(acc_dict[phase][epoch]))
        
        plt.plot(range(len(loss_dict['train'][:epoch])), loss_dict['train'][:epoch], 'r')
        plt.plot(range(len(loss_dict['val'][:epoch])), loss_dict['val'][:epoch], 'b')
          
        plt.plot(range(len(acc_dict['train'][:epoch])), acc_dict['train'][:epoch], 'y')
        plt.plot(range(len(acc_dict['val'][:epoch])), acc_dict['val'][:epoch], 'g')
                 
        plt.legend(['Training Loss', 'Val Loss', 'Training Acc', 'Val Acc'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(f'/MULTIX/DATA/nccid/{model.model_name}_metrics_k{k}.png')
        
        loss_df = pd.DataFrame.from_dict(loss_dict)
        loss_df.to_csv(f'/MULTIX/DATA/nccid/{model.model_name}_loss_k{k}.csv')

        acc_df = pd.DataFrame.from_dict(acc_dict)
        acc_df.to_csv(f'/MULTIX/DATA/nccid/{model.model_name}_acc_k{k}.csv')


        if loss_dict['val'][epoch] > best_val_loss:
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
            print(f'saving model weights to {model.model_name}.h5')
            model.save_weights(f'/MULTIX/DATA/nccid/{model.model_name}_{k}.h5')
            best_val_loss = loss_dict['val'][epoch]

    loss_dict = dict(itertools.islice(loss_dict.items(), end_epoch))
    acc_dict = dict(itertools.islice(acc_dict.items(), end_epoch))
    return loss_dict, acc_dict


def training(model, dataloader, k, patience=20):
    if model.model_type == 'keras':
        loss, acc = train_keras(model, dataloader, k, patience)
    elif model.model_type == 'pytorch':
        loss, acc = train_pytorch()

    return loss, acc
    

def main(model, fold, df):
        train_df = df[df[f'kfold_{fold}'] == "train"]
        val_df = df[df[f'kfold_{fold}'] == "val"]
        test_df = df[df[f'kfold_{fold}'] == 'test']
        
        #make generators
        print(train_df['xray_status'].value_counts())

        params = {'batchsize':24, "num_workers":4, "k":fold}
        
        train_loader, val_loader, _ = make_generators(model.model_type, train_df, val_df, test_df, params)
        # create dict of dataloaders
        x, y = next(iter(train_loader))
        
        img = x[0].numpy().flatten()
        plt.hist(img, bins=20)
        plt.savefig('/MULTIX/DATA/nccid/pixel_hist.png')

        x, y = next(iter(train_loader))
        fig = plt.figure(figsize=(4., 4.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                        nrows_ncols=(4,4),  # creates 2x2 grid of axes
                        axes_pad=0.1,  # pad between axes in inch.
                        )

        for ax, im in zip(grid, [x[i] for i in range(16)]):
            # Iterating over the grid returns the Axes.
            if model.model_type == 'pytorch':
              im = im.permute(1,2,0)
            else:
              ax.imshow(im)
        plt.savefig('/MULTIX/DATA/nccid/batch_img.png')


        dataloaders = {'train':train_loader, 'val':val_loader}
        loss, acc = training(model, dataloaders, k=fold)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training Script')
    parser.add_argument('--data_csv', default='/MULTIX/DATA/nccid/nccid_preprocessed.csv', type=str, help='Path to data file')
    args = parser.parse_args()

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    assert tf.test.is_gpu_available()
    assert tf.test.is_built_with_cuda()
    
    df = pd.read_csv(args.data_csv)
    mapping = {'negative':0, 'positive':1}
    
    df = df[df['xray_status']!=np.nan]
    df = df.dropna(subset=['xray_status'])
    
    df['xray_status'] = df['xray_status'].map(mapping)

    for fold in range(1,6):
        model = ECovNet()
        main(model, fold, df)