from selectors import EpollSelector
from pandas.core.algorithms import mode
import tensorflow as tf
import tensorflow.keras as keras
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import cv2
import os

import random

from functools import partial
import numpy as np

AUTOTUNE = tf.data.experimental.AUTOTUNE

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from torch.utils.data import Dataset, DataLoader
import torch


to_tensor = A.Compose([ToTensorV2()])

def train_aug_fn(image, mean, sd):
    transforms = A.Compose([
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                A.CLAHE(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),    
                A.ColorJitter(),
                A.Normalize(mean=mean, std=sd),
             ])
    data = {"image":image}
    aug_data = transforms(**data)
    aug_img = aug_data["image"]
    aug_img= tf.cast(aug_img, tf.float32)   
    return aug_img

def val_aug_fn(image, mean, sd):
    val_transforms = A.Compose([
      A.Normalize(mean=mean, std=sd),
    ])

    data = {"image":image}
    aug_data = val_transforms(**data)
    aug_img = aug_data["image"]
   # aug_img = to_tensor(aug_img)
    aug_img= tf.cast(aug_img, tf.float32)
    return aug_img

class PytorchDataGen(Dataset):
    def __init__(self, data, train, k, label=None):
        self.train = train

        self.label = label

        if self.label != None:
            print('filtering down to one class ...')
            self.data = data[data['xray_status']==label]
        else:
            self.data = data

        print(f"loading images with labels {np.unique(self.data['xray_status'].values)} ...")

        self.k = k
        self.mean = [(0.5331, 0.5331, 0.5331),(0.5336, 0.5336, 0.5336),
                     (0.5337, 0.5337, 0.5337),(0.5333, 0.5333, 0.5333),(0.5336, 0.5336, 0.5336)]

        self.sd = [(0.2225, 0.2225, 0.2225),(0.2226, 0.2226, 0.2226),
                   (0.2224, 0.2224, 0.2224),(0.2225, 0.2225, 0.2225), (0.2226, 0.2226, 0.2226)]

        if self.train:
            self.transforms = A.Compose([
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.CLAHE(p=0.5),A.HorizontalFlip(p=0.5),A.RandomBrightnessContrast(p=0.5),    
            A.ColorJitter(),A.Normalize(mean=self.mean[self.k-1], std=self.sd[self.k-1]),ToTensorV2()]) 
        else:
           self.transforms = A.Compose([
                             A.Normalize(mean=self.mean[self.k-1], std=self.sd[self.k-1]),
                             ToTensorV2()
                             ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data['dgx_structured_path'].iloc[idx]
        image = cv2.imread(path)
       # image = image/255.0
        label = self.data['xray_status'].iloc[idx]
        image = self.transforms(image=image)["image"]
            
        return image, torch.tensor(label, dtype=torch.float)
    
class KerasDataGen(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size, transforms, k, shuffle=True, chexpert=False):
    
        self.data = data.copy()
        self.bs = batch_size
        self.shuffle = shuffle
        self.transforms = transforms
        self.k = k
        self.mean = [(0.5331, 0.5331, 0.5331),(0.5336, 0.5336, 0.5336),
                     (0.5337, 0.5337, 0.5337),(0.5333, 0.5333, 0.5333),(0.5336, 0.5336, 0.5336)]

        self.sd = [(0.2225, 0.2225, 0.2225),(0.2226, 0.2226, 0.2226),
                   (0.2224, 0.2224, 0.2224),(0.2225, 0.2225, 0.2225), (0.2226, 0.2226, 0.2226)]
        self.n = len(self.data)

        self.chexpert = chexpert

    def __len__(self):
        return self.n // self.bs

    def on_epoch_end(self):
        if self.shuffle == True:
            self.data = self.data.sample(frac=1).reset_index(drop=True)

    def get_image(self, path):
        image = cv2.imread(path)
        if self.chexpert == True:
            image = cv2.resize(image, (480, 480), interpolation = cv2.INTER_AREA)
        return image

    def get_label(self, label, num_classes):
        return tf.keras.utils.to_categorical(label, num_classes)

    def get_data(self, batch):
        path_batch = batch['dgx_structured_path']
        y_batch = batch['xray_status'].values

        x_batch = [self.get_image(x) for x in path_batch]

        if self.chexpert == True:
            y_batch = [self.get_label(y,5) for y in y_batch]

        return x_batch, y_batch        

    def __getitem__(self, index):
        mean = self.mean[self.k-1]
        sd = self.mean[self.k-1]
        
        batches = self.data[index*self.bs: (index + 1)*self.bs]
        batch_x, batch_y = self.get_data(batches)

        if self.transforms is not None:
            batch_x = tf.cast([self.transforms(x, mean, sd) for x in batch_x], tf.float32)
        return batch_x, tf.expand_dims(batch_y, axis=-1)

class SiameseDataGen(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size, transforms, k, shuffle=True):
    
        self.generator1 = KerasDataGen(data, 1, transforms, k)
        self.generator2 = KerasDataGen(data, 1, transforms, k)

    def __len__(self):
       return len(self.generator1)

    def __getitem__(self, index):
       x1,y1 = self.generator1[index]
       x2,y2 = self.generator2[index]

       if y1 == y2:
           label = [1.0]
       else:
           label = [0.0]

       return (x1,x2), tf.cast(label, float)


def make_generators(model, train_df, val_df, test_df, params, chexpert=False):
    assert model['model_type'] in ['keras', 'pytorch', 'fastai']
    
    if model['model_type'] == "keras":
        if model["model_name"] == "siamese_net" and model['pretrained']:
            Generator = SiameseDataGen
        else:
            Generator = KerasDataGen

        train_dg = Generator(train_df, params["batchsize"], transforms=train_aug_fn, k=params["k"])   
        val_dg = Generator(val_df, params["batchsize"], transforms=val_aug_fn, k=params["k"])
        test_dg = Generator(test_df, params["batchsize"], transforms=val_aug_fn, k=params["k"])

    elif model['model_type'] == "pytorch":
        if model['model_name']=='coronet' and model['supervised']==False:
            print('filtering labels for coronet ...')
            label = 0
        else:
            label = None

        train_dataset = PytorchDataGen(train_df, train=True, k=params["k"], label=label)
        train_dg = DataLoader(train_dataset, batch_size=params["batchsize"], 
        shuffle=True, num_workers = params["num_workers"], pin_memory=True)
        
        val_dataset = PytorchDataGen(val_df, train=False, k=params["k"], label=label)
        val_dg = DataLoader(val_dataset, batch_size=params["batchsize"], 
        shuffle=True, num_workers = params["num_workers"], pin_memory=True)

        test_dataset = PytorchDataGen(test_df, train=False, k=params["k"], label=label)
        test_dg = DataLoader(test_dataset, batch_size=params["batchsize"], 
        shuffle=False, num_workers = params["num_workers"], pin_memory=True)

        #siamese net is a special case

    return train_dg, val_dg, test_dg
