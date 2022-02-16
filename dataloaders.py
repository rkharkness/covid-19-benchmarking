from pandas.core.algorithms import mode
import tensorflow as tf
import tensorflow.keras as keras
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import cv2
import os

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
               # ToTensorV2()
             ])
    data = {"image":image}
    aug_data = transforms(**data)
    aug_img = aug_data["image"]
   # aug_img = to_tensor(aug_img) 
    aug_img= tf.cast(aug_img, tf.float32)   
    return aug_img

def val_aug_fn(image, mean, sd):
    val_transforms = A.Compose([
     # A.HorizontalFlip(p=0.5)
      A.Normalize(mean=mean, std=sd),
    #  ToTensorV2()     
    ])
    data = {"image":image}
    aug_data = val_transforms(**data)
    aug_img = aug_data["image"]
   # aug_img = to_tensor(aug_img)
    aug_img= tf.cast(aug_img, tf.float32)
    return aug_img

class ClassWiseDataGen(Dataset):
    def __init__(self, data, train, k):
        self.data = data
        self.train = train
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

class PytorchDataGen(Dataset):
    def __init__(self, data, train, k, label=None):
        self.train = train

        self.label = label

        if self.label:
            self.data = data[data['xray_status']==0]
        else:
            self.data = data

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
    def __init__(self, data, batch_size, transforms, k, shuffle=True):
    
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

    def __len__(self):
        return self.n // self.bs

    def on_epoch_end(self):
        if self.shuffle == True:
            self.data = self.data.sample(frac=1).reset_index(drop=True)

    def get_image(self, path):
        image = cv2.imread(path)
        return image

    def get_label(self, label, num_classes):
        return tf.keras.utils.to_categorical(label, num_classes)

    def get_data(self, batch):
        path_batch = batch['dgx_structured_path']
        y_batch = batch['xray_status'].values

        x_batch = [self.get_image(x) for x in path_batch]
        #y_batch = [self.get_label(y,2) for y in label_batch]

        return x_batch, y_batch

    def __getitem__(self, index):
        mean = self.mean[self.k-1]
        sd = self.mean[self.k-1]
        
        batches = self.data[index*self.bs: (index + 1)*self.bs]
        batch_x, batch_y = self.get_data(batches)

        if self.transforms is not None:
            batch_x = tf.cast([self.transforms(x, mean, sd) for x in batch_x], tf.float32)
        return batch_x, tf.expand_dims(batch_y, axis=-1)


def make_generators(model_type, train_df, val_df, test_df, params):
    assert model_type in ['keras', 'pytorch', 'fastai']
    
    if model_type == "keras":
        train_dg = KerasDataGen(train_df, params["batchsize"], transforms=train_aug_fn, k=params["k"])   
        val_dg = KerasDataGen(val_df, params["batchsize"], transforms=val_aug_fn, k=params["k"])
        test_dg = KerasDataGen(test_df, params["batchsize"], transforms=val_aug_fn, k=params["k"])

    elif model_type == "pytorch":
        train_dataset = PytorchDataGen(train_df, train=True, k=params["k"])
        train_dg = DataLoader(train_dataset, batch_size=params["batchsize"], 
        shuffle=True, num_workers = params["num_workers"], pin_memory=True)
        
        val_dataset = PytorchDataGen(val_df, train=False, k=params["k"])
        val_dg = DataLoader(val_dataset, batch_size=params["batchsize"], 
        shuffle=True, num_workers = params["num_workers"], pin_memory=True)

        test_dataset = PytorchDataGen(test_df, train=False, k=params["k"])
        test_dg = DataLoader(test_dataset, batch_size=params["batchsize"], 
        shuffle=False, num_workers = params["num_workers"], pin_memory=True)

        # coronet and siamese net is a special case

    return train_dg, val_dg, test_dg
