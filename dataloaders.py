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



#from transforms import transforms
# update to be a shared hp ?
transforms = A.Compose([
                A.VerticalFlip(p=0.5),              
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                A.Affine(translate_percent=10,p=0.5),
                A.CLAHE(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),    
                A.RandomGamma(p=0.5),
                A.ColorJitter
                
            # NORMALIZE?
            # SEGMENT WITH LAMBDA
             ])
to_tensor = A.Compose([ToTensorV2()])

val_transforms = A.Compose([
                # SEGMENT WITH LAMBDA
                # NORMALIZE?   
    ])

def train_aug_fn(image):
    data = {"image":image}
    aug_data = transforms(**data)
    aug_img = aug_data["image"]
    aug_img= tf.cast(aug_img/255.0, tf.float32)
    return aug_img

def val_aug_fn(image):
    data = {"image":image}
    aug_data = val_transforms(**data)
    aug_img = aug_data["image"]
    aug_img= tf.cast(aug_data/255.0, tf.float32)
    return aug_img

class PytorchDataGen(Dataset):
    def __init__(self, data, transforms):
        self.data = data
        self.transforms = transforms
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data['structured_path'][idx]
        image = cv2.imread(path)

        label = self.data['xray_status'][idx]

        if transforms is not None:
            image = self.transforms(image=image)["image"]
            image = to_tensor(image=image)["image"]

        return image, label
    
class KerasDataGen(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size, transforms, shuffle=True):
    
        self.data = data.copy()
        self.bs = batch_size
        self.shuffle = shuffle
        self.transforms = transforms
        
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
        path_batch = batch[self.data['structured_path']]
        label_batch = batch[self.data['xray_status']]

        x_batch = [self.get_image(x for x in path_batch)]
        y_batch = [self.get_label(y,2) for y in label_batch]

        return x_batch, y_batch

    def __getitem__(self, index):
        batches = self.data[index*self.bs: (index + 1)*self.bs]
        batch_x, batch_y = self.get_data(batches)

        if self.transform is not None:
            batch_x = tf.cast([self.transforms(x) for x in batch_x], tf.float32)
        return batch_x, batch_y


def make_generators(model_type, train_df, val_df, test_df, params):
    assert model_type in ['keras', 'pytorch', 'fastai']
    if model_type == "keras":
        train_dg = KerasDataGen(train_df, params["batchsize"], transforms=train_aug_fn)   
        val_dg = KerasDataGen(val_df, params["batchsize"], transforms=val_aug_fn)
        test_dg = KerasDataGen(test_df, params["batchsize"], transforms=val_aug_fn)

    elif model_type == "pytorch":
        train_dataset = PytorchDataGen(train_df, transforms=train_aug_fn)
        train_dg = DataLoader(train_dataset, batch_size=params["batchsize"], 
        shuffle=True, num_workers = params["num_workers"])
        
        val_dataset = PytorchDataGen(val_df, transforms=val_transforms)
        val_dg = DataLoader(val_dataset, batch_size=params["batchsize"], 
        shuffle=True, num_workers = params["num_workers"])

        test_dataset = PytorchDataGen(test_df, transforms=val_transforms)
        test_dg = DataLoader(test_dataset, batch_size=params["batchsize"], 
        shuffle=False, num_workers = params["num_workers"])

        # coronet and siamese net is a special case

    return train_dg, val_dg, test_dg


















# def process_data(train, image, label, img_size):
#     if train == True:
#         aug_img = tf.numpy_function(func=train_aug_fn, inp=[image, img_size], Tout=tf.float32)
#     else:
#         aug_img = tf.numpy_function(func=val_aug_fn, inp=[image, img_size], Tout=tf.float32)

#     return aug_img, label

# # create dataset
# ds_alb = data.map(partial(process_data, img_size=480),
#                   num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

# print(ds_alb)

# def set_shapes(img, label, img_shape=(480,480,1)):
#     img.set_shape(img_shape)
#     label.set_shape([])
#     return img, label

# ds_alb = ds_alb.map(set_shapes, num_parallel_calls=AUTOTUNE).batch(32).prefetch(AUTOTUNE)
# ds_alb