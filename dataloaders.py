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

class ImbalancedSiameseNetworkDataset(Dataset):
    
    def __init__(self,trainImageFolderDataset,train, testImageFolderDataset=None, transform=None):
        # self.train = train
        self.trainImageFolderDataset = trainImageFolderDataset    
        self.transform = transform
        self.comparison_ds = trainImageFolderDataset

        if train == True:
          self.comparison_ds = trainImageFolderDataset
        else:
          self.comparison_ds = testImageFolderDataset
        
    def __getitem__(self,index):

        should_get_pneum = random.randint(0,1)

        if should_get_pneum:
          while True:
            img0_idx = np.random.choice(self.trainImageFolderDataset.index)
            img0_tuple = (self.trainImageFolderDataset.filename[img0_idx], self.trainImageFolderDataset.pneumonia_binary[img0_idx])
            if img0_tuple[1] == 1:
              break
        else:
          while True:
            img0_idx = np.random.choice(self.trainImageFolderDataset.index)
            img0_tuple = (self.trainImageFolderDataset.filename[img0_idx], self.trainImageFolderDataset.pneumonia_binary[img0_idx])
            if img0_tuple[1] == 0:
              break          

        #we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1)
        
        if should_get_same_class:
            while True:
                #keep looping till the same class image is found
                img1_idx = np.random.choice(self.comparison_ds.index) 
                img1_tuple = (self.comparison_ds.filename[img1_idx], self.comparison_ds.pneumonia_binary[img1_idx])
                
                if img0_tuple[1]==img1_tuple[1]:
                    break
        else:
            while True:
              img1_idx = np.random.choice(self.comparison_ds.index) 
              img1_tuple = (self.comparison_ds.filename[img1_idx], self.comparison_ds.pneumonia_binary[img1_idx])
              
              if img0_tuple[1]!=img1_tuple[1]:
                  break

        img0 = cv2.imread(img0_tuple[0])
        img1 = cv2.imread(img1_tuple[0])

        
        if self.transform is not None:
            img0 = self.transform(image=img0)["image"]
            img1 = self.transform(image=img1)["image"]
            
        return img0/255.0, img1/255.0 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32)), img0_tuple[1], img1_tuple[1]


    def __len__(self):
        return len(self.trainImageFolderDataset)

class PytorchDataGen(Dataset):
    def __init__(self, data, train, k, label=None):
        self.train = train

        self.label = label

        if self.label:
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


def make_generators(model, train_df, val_df, test_df, params):
    assert model['model_type'] in ['keras', 'pytorch', 'fastai']
    
    
    if model['model_type'] == "keras":
        train_dg = KerasDataGen(train_df, params["batchsize"], transforms=train_aug_fn, k=params["k"])   
        val_dg = KerasDataGen(val_df, params["batchsize"], transforms=val_aug_fn, k=params["k"])
        test_dg = KerasDataGen(test_df, params["batchsize"], transforms=val_aug_fn, k=params["k"])

    elif model['model_type'] == "pytorch":
        if model['model_name']=='coronet':
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
