from dataloaders import *
import tensorflow as tf
import torch
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import itertools
from residual_attn.res_attn import AttentionResNet56

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

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def train_mag_sd(model, model_name, dataloader, optimizer, loss_fn, patience, supervised=True, pretrained_weights=None):
    from torch.utils.tensorboard import SummaryWriter
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

            for batch in tqdm(len(dataloader[phase])):
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


def train_pytorch(model, model_name, dataloader, optimizer, loss_fn, patience, supervised=True, pretrained_weights=None):    
    assert model.get_model_type() == 'pytorch'

    from torch.utils.tensorboard import SummaryWriter
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
                model.train()
            else:
                model.eval()

            for batch in tqdm(len(dataloader[phase])):
                if len(batch) > 1:
                    batch_x, batch_y = batch
                    batch_x.to(device)
                    batch_y.to(device)

                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        pred = model(batch_x, batch_y)
                        loss = loss_fn(pred, batch_y)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    if len(batch_y[0]) == len(batch_x[0]): # check if batch_y is an image
                        pass
                    
                    else:
                        batch_y = np.argmax(batch_y, axis=1)
                        acc = accuracy_score(batch_y, np.argmax(pred, axis=-1))
                        
                    loss_avg[phase].append(loss.item())
                    acc_avg[phase].append(acc)

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
                    optimizer.param_groups[0]['lr'] = lr
                    print(f"Reducing lr to {lr}")

                if no_improvement == patience:
                    print(f"No improvement for {no_improvement}, early stopping at epoch {epoch}")
                    break
        else:
            no_improvement = 0
            print(f'saving model weights to {model_name}.pth')
            torch.save(model.state_dict(), model_name + '.pth')
        
    loss_dict = dict(itertools.islice(loss_dict.items(), end_epoch))
    acc_dict = dict(itertools.islice(acc_dict.items(), end_epoch))
    return loss_dict, acc_dict



def train_keras(model, model_name, dataloader, optimizer, loss_fn, patience, supervised=True, pretrained_weights=None):
    # supervised included self-supervised methods i.e. two inputs
    assert model.get_model_type() == 'keras'

    from tensorboardx import SummaryWriter
    writer = SummaryWriter(model_name)

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    loss_dict = {'train': np.zeros(shape=(500,), dtype=np.float32),
                'val': np.zeros(shape=(500,), dtype=np.float32)}
    
    acc_dict = {'train': np.zeros(shape=(500,), dtype=np.float32),
                'val':np.zeros(shape=(500,), dtype=np.float32)}

    no_improvement = 0
    lr = optimizer.lr
    end_epoch = 0

    for epoch in range(500):
        end_epoch =+1
        loss_avg = {'train':tf.keras.metrics.Mean(),
                    'val':tf.keras.metrics.Mean()}

        acc_avg = {'train':tf.keras.metrics.Mean(),
                    'val':tf.keras.metrics.Mean()}

        for phase in ['train', 'val']:
            for batch in tqdm(len(dataloader[phase])):
                if len(batch) > 1:
                    batch_x, batch_y = dataloader[phase][batch] # batch_y can be paired image 
                    with tf.GradientTape() as tape:
                        pred = model(batch_x)
                        loss = loss_fn(batch_y, pred)
                else:
                    assert supervised == False
                    ### more here for unsupervised approaches
                    
                if phase == 'train':
                    grad = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(grad, model.trainable_variables))

                    loss_avg[phase](loss)
                    
                else:
                    pred = model.forward(batch_x)

                    if len(batch_y[0]) == len(batch_x[0]): # check if batch_y is an image
                        pass
                    
                    else:
                        batch_y = np.argmax(batch_y, axis=1)
                        acc_avg[phase](accuracy_score(batch_y, np.argmax(pred, axis=-1)))

                dataloader[phase].on_epoch_end()

                loss_dict[phase][epoch] = loss_avg[phase].result()
                acc_dict[phase][epoch] = acc_avg[phase].result()

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
            model.save_weights(model_name + ".h5")

    loss_dict = dict(itertools.islice(loss_dict.items(), end_epoch))
    acc_dict = dict(itertools.islice(acc_dict.items(), end_epoch))
    return loss_dict, acc_dict


def training(model, model_name, dataloader, patience):
    optimizer = model.get_optimizer()
    loss_fn = model.get_loss_fn()

    if model.get_model_type() == 'keras':
        loss, acc = train_keras()
    elif model.get_model_type() == 'pytorch':
        loss, acc = train_pytorch()
    

def main(model, df):
    for fold in range(1,6):
        train_df = df[df[f'kfold_{fold}'] == "train"]
        val_df = df[df[f'kfold_{fold}'] == "val"]
        test_df = df[df[f'kfold_{fold}'] == 'test']
        #make generators
        
        train_loader, val_loader, test_loader = make_generators(train_df, val_df, test_df, params)
        # create dict of dataloaders
        x, y = next(iter(train_loader))
        fig = plt.figure(figsize=(4., 4.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                        nrows_ncols=(2, 2),  # creates 2x2 grid of axes
                        axes_pad=0.1,  # pad between axes in inch.
                        )

        for ax, im in zip(grid, [x[0], x[1], x[2], x[3]]):
            # Iterating over the grid returns the Axes.
            ax.imshow(im)

        plt.savefig('/content/batch_example.png')

        dataloaders = {'train':train_loader, 'val':val_loader}
        loss, acc = training(model, 'res_attn', dataloaders, 20)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training Script')
    parser.add_argument('--data_csv', default='~/repos/covid-19-benchmarking/nccid_sample.csv', type=str, help='Path to data file')
    args = parser.parse_args()

   # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
   # assert tf.test.is_gpu_available()
   # assert tf.test.is_built_with_cuda()
        
    df = pd.read_csv(args.data_csv)
    model = AttentionResNet56()
    main(model, df)