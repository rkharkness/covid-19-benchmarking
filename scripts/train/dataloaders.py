import os 
import tensorflow as tf 
from tensorflow.keras import utils

import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import cv2
import random
import numpy as np

import pydicom
import tensorflow_io as tfio

from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.datasets as datasets



class PytorchDataGen(Dataset):
    def __init__(self, data, train, k, cat=False, png=None, label=None):
        self.train = train
        self.label = label
        self.cat = cat
        self.png = png
        if self.png == None:
            if self.label != None:
                print('filtering down to one class ...')
                data = data[data['xray_status']==label]
                print(f"loading images with labels {np.unique(data['xray_status'].values)} ...")
        self.data = data
        self.k = k

        if self.train:
            self.transforms = A.Compose([
                                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.2),
                                    A.HorizontalFlip(p=0.2),
                                    A.OneOf([
                                        A.RandomBrightnessContrast(p=0.2),  
                                        A.RandomContrast(p=0.2)
                                        ], p=0.2),
                                    A.RandomBrightnessContrast(p=0.2),  
                                    A.RandomContrast(p=0.2),  
                                    A.ColorJitter(0.2),
                                    ToTensorV2()])
        else:
           self.transforms = A.Compose([
                             ToTensorV2()
                             ])
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.png == None:
            path = self.data['cxr_path'].iloc[idx]
            y_label = self.data['xray_status'].iloc[idx]

            dcm = pydicom.dcmread(path)
            arr = dcm.pixel_array

        elif self.png == 'chexpert':
            path = self.data['Path'].iloc[idx]
            y_label = self.data[['No Finding', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']].iloc[idx]
            y_label = [0 if i==-1 or i==0 else 1 for i in y_label]
            arr  = cv2.imread(path, 1)

        elif self.png == 'covidgr':
            path = self.data['cxr_path'].iloc[idx]
            y_label = self.data['xray_status'].iloc[idx]
            arr  = cv2.imread(path, 1)

        arr = cv2.resize(arr, (480, 480), interpolation = cv2.INTER_AREA)
        
        arr = arr/np.max(arr)
        if self.png == None:
          try:
            if self.data['PhotometricInterpretation'].iloc[idx] == 'MONOCHROME1':
                arr = 1. - arr
          except:
            if dcm.PhotometricInterpretation == 'MONOCHROME1':
                arr = 1. - arr
          arr = np.dstack([arr,arr,arr])
        arr = arr.astype(np.float32)
        #arr = np.dstack([arr,arr,arr])

        image = self.transforms(image=arr)["image"]
        y_label = torch.tensor(y_label, dtype=torch.float)
        if self.cat == True:
            y_label = torch.nn.functional.one_hot(torch.tensor(y_label,dtype=torch.int64), num_classes=2) 
        else:
            y_label = torch.unsqueeze(y_label, 0)

        return image, y_label, path
    
    

@tf.function
def _fixup_shape(x, y, path):
    x.set_shape([None, 480, 480, 3]) # n, h, w, c
    y.set_shape([None, 1]) # n, nb_classes
    return x, y, path

@tf.function
def generator_w(df, train, ssl, png, cat):
    def generator(i):
        def aug_fn(image, train):
            image = image.numpy()
            if train == True:
                transforms = A.Compose([
                            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=15, p=0.2),
                            A.HorizontalFlip(p=0.3),
                            A.OneOf([
                                A.RandomBrightnessContrast(p=0.2),  
                                A.RandomContrast(p=0.2)
                                ], p=0.2),
                            A.ColorJitter(0.2),
                            ])
                data = {"image":image}
                aug_data = transforms(**data)
                aug_img = aug_data["image"]
                aug_img= tf.cast(aug_img, tf.float32)
            else:
                aug_img = tf.cast(image, tf.float32)

            return aug_img 
        
        i = i.numpy()
#        print(df)
        path = df['cxr_path'].values[i]

        image_bytes = tf.io.read_file(path)
        arr = tfio.image.decode_dicom_image(image_bytes, on_error='lossy')
        
        arr = tf.image.convert_image_dtype(arr, dtype=tf.float32)

        arr = tf.image.resize(
        arr,
        (480,480),
        method=tf.image.ResizeMethod.AREA,
        preserve_aspect_ratio=False,)

        arr = arr/tf.reduce_max(arr)
        arr = arr[0]
        arr = tf.concat([arr,arr,arr], axis=-1)

        img = aug_fn(arr, train)

        label = df['xray_status'].values[i]

        if cat==True:
            label = utils.to_categorical(label,2)
        else:
            label = np.expand_dims(label, 0)

        return img, label, path
    
    def png_generator(i):
        def aug_fn(image, train):
            image = image.numpy()
            if train == True:
                transforms = A.Compose([
                            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=15, p=0.2),
                            A.HorizontalFlip(p=0.3),
                            A.OneOf([
                                A.RandomBrightnessContrast(p=0.2),  
                                A.RandomContrast(p=0.2)
                                ], p=0.2),
                            A.ColorJitter(0.2),
                            ])
                data = {"image":image}
                aug_data = transforms(**data)
                aug_img = aug_data["image"]
                aug_img= tf.cast(aug_img, tf.float32)
            else:
                aug_img = tf.cast(image, tf.float32)

            return aug_img 
        
        i = i.numpy()
        if png == 'chexpert':
            path = df['Path'].values[i]
            
        elif png == 'covidgr':
            path = df['cxr_path'].values[i]

        arr = cv2.imread(path,1)      
        arr = tf.image.convert_image_dtype(arr, dtype=tf.float32)

        arr = tf.image.resize(
        arr,
        (480,480),
        method=tf.image.ResizeMethod.AREA,
        preserve_aspect_ratio=False)

        arr = arr/tf.reduce_max(arr)

        if ssl:
            return arr, arr, path
        else:
            img = aug_fn(arr, train)

        if png == 'chexpert':
            label = df[['No Finding', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']].values[i]
            label = [0 if i==-1 or i==0 else 1 for i in label]
            if cat:
                label = utils.to_categorical(label,5)
                
        elif png == 'covid_qu_ex' or 'covidgr':
            label = df['xray_status'].values[i]
            label = np.expand_dims(label,0)

        return img, label, path
    
    if png == None:
        return generator
    else:
        return png_generator


def create_generator(df, train, ssl, png, batch_size, cat):
    idx = list(range(len(df))) # The index generator
    if train==True:
        dataset = tf.data.Dataset.from_generator(lambda: idx, tf.int16).shuffle(len(idx))
    else:
        dataset = tf.data.Dataset.from_generator(lambda: idx, tf.int16)

    generator = generator_w(df, train, ssl, png, cat)

    data_generator = dataset.map(lambda i: tf.py_function(func=generator, 
                                    inp=[i],
                                    Tout=[tf.float32,
                                          tf.float32, tf.string]
                                    ), num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).map(_fixup_shape, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    return data_generator

def create_siamese_generator(df, train, ssl,  png, batch_size=24, cat=False):
    data_generator1 = create_generator(df, train,  ssl, png, batch_size, cat)
    data_generator2 = create_generator(df, train, ssl, png, batch_size, cat)
    return zip(data_generator1, data_generator2)

def make_generators(model, train_df, val_df, test_df, params, png=None, ssl=False):
    assert model['model_type'] in ['keras', 'pytorch']
    #if type(test_df) != list:
    #  if test_df['cxr_path'].values[0].split('.')[-1] == 'jpg':
    #    png = 'covidgr'
     # fix above later   
    print('png', png)
    if model['model_type'] == "keras":
        cat = False
        if model['model_name']=="capsnet_chexpert" and model['pretrained']==None:
            png = 'chexpert'
        
        if model['model_name'] == "xvitcos_chexpert_no_nan" and model['pretrained']==None:
            png = 'chexpert'

        if model["model_name"]=="ssl_am_chexpert" and model["supervised"]==False:
            ssl = True
            png = 'chexpert'
        else:
            ssl = False

        if model["model_name"] == "siamese_net" and model['pretrained']!=None and model['supervised']==False:
            train_dg = create_siamese_generator(train_df, train=True, ssl=ssl, png=png, batch_size=params['batchsize'], cat=cat)
            val_dg = create_siamese_generator(val_df, train=False, ssl=ssl, png=png, batch_size=params['batchsize'],cat=cat)
            test_dg = create_siamese_generator(test_df, train=False, ssl=ssl, png=png, batch_size=params['batchsize'],cat=cat)

        else:
            train_dg = create_generator(train_df, train=True, ssl=ssl, png=png, batch_size=params['batchsize'], cat=cat)
            val_dg = create_generator(val_df, train=False, ssl=ssl, png=png, batch_size=params['batchsize'],cat=cat)
            test_dg = create_generator(test_df, train=False, ssl=ssl, png=png, batch_size=params['batchsize'],cat=cat)

    elif model['model_type'] == "pytorch":
        if model['model_name']=='coronet' and model['supervised']==False:
            print('filtering labels for coronet ...')
            label = 0 #keep only covid-negative for ae pretraining
        else:
            label = None

        if model['model_name']=='mag_sd':
            cat = True
        else:
            cat = False

        if model['model_name'] == 'covidnet_chexpert' and model['pretrained'] == None:
            png = 'chexpert'

        train_dataset = PytorchDataGen(train_df, train=True, k=params["k"], label=label, cat=cat, png=png)
        train_dg = DataLoader(train_dataset, batch_size=params["batchsize"], 
        shuffle=True, num_workers = params["num_workers"], pin_memory=True)
        
        val_dataset = PytorchDataGen(val_df, train=False, k=params["k"], label=label, cat=cat, png=png)
        val_dg = DataLoader(val_dataset, batch_size=params["batchsize"], 
        shuffle=False, num_workers = params["num_workers"], pin_memory=True)

        test_dataset = PytorchDataGen(test_df, train=False, k=params["k"], label=label, cat=cat, png=png)
        test_dg = DataLoader(test_dataset, batch_size=params["batchsize"], 
        shuffle=False, num_workers = params["num_workers"], pin_memory=True)

    return train_dg, val_dg, test_dg
