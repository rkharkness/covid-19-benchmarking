import tensorflow as tf
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import cv2
import random
import numpy as np


from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.datasets as datasets

from models.ssl_am import generate_pair


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
    def __init__(self, data, train, k, label=None, imagenet=False):
        self.train = train
        self.imagenet = imagenet
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

    # def get_dataset(self):
    #     """
    #     Uses torchvision.datasets.ImageNet to load dataset.
    #     Downloads dataset if doesn't exist already.
    #     Returns:
    #          torch.utils.data.TensorDataset: trainset, valset
    #     """

    #     trainset = datasets.ImageNet('datasets/ImageNet/train/', split='train', transform=self.train_transforms,
    #                                  target_transform=None, download=True)
    #     valset = datasets.ImageNet('datasets/ImageNet/val/', split='val', transform=self.val_transforms,
    #                                target_transform=None, download=True)
    #     data = pd.concat([trainset, valset])
    #     return data

    def __getitem__(self, idx):
        path = self.data['dgx_structured_path'].iloc[idx]
        image = cv2.imread(path)
       # image = image/255.0
        label = self.data['xray_status'].iloc[idx]
        image = self.transforms(image=image)["image"]
        
        label= torch.tensor(label, dtype=torch.float)
        return image, torch.unsqueeze(label,0)
    
class KerasDataGen(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size, transforms, k, shuffle=True, chexpert=False, ssl=False):
    
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
        self.ssl = ssl

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
        batch_y = tf.expand_dims(batch_y, axis=-1)

        if self.transforms is not None:
            batch_x = tf.cast([self.transforms(x, mean, sd) for x in batch_x], tf.float32)
        
        if self.ssl:
          batch_y = np.array([generate_pair(x) for x in batch_x])
          batch_y = tf.convert_to_tensor(batch_y, 'float32')

        return batch_x, batch_y

class SiameseDataGen(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size, transforms, k, shuffle=True, chexpert=False, ssl=False):
    
        self.generator1 = KerasDataGen(data, 1, transforms, k)
        self.generator2 = KerasDataGen(data, 1, transforms, k)

        self.data = data
        self.batch_size = batch_size

    def __len__(self):
       return len(self.data)//self.batch_size

    def __getitem__(self, index):
        x1_list = []
        x2_list = []
        y = []

        while len(y) != self.batch_size:

                should_get_pneum = random.randint(0,1)
                if should_get_pneum:
                    while True:
                        img1_idx = np.random.choice(self.data.index) 
                        x1,y1 = self.generator1[img1_idx]
                        if y1 == 1:
                            break

                else:
                    while True:
                        if should_get_pneum == 0:
                            img1_idx = np.random.choice(self.data.index) 
                            x1,y1 = self.generator1[img1_idx]
                            if y1 == 0:
                                break         

                #we need to make sure approx 50% of images are in the same class
                should_get_same_class = random.randint(0,1)
                
                if should_get_same_class:
                    while True:
                        #keep looping till the same class image is found
                        img2_idx = np.random.choice(self.data.index)
                        x2,y2 = self.generator2[img2_idx]

                        if y1 == y2:
                            label = 1.0
                            break
                else:
                    while True:
                        img2_idx = np.random.choice(self.data.index)
                        x2,y2 = self.generator2[img2_idx]
                    
                        if y1 != y2:
                            label = 0.0
                            break

                x1_list.append(x1)
                x2_list.append(x2)
                y.append([label])

        return (tf.convert_to_tensor(x1_list, tf.float32), tf.convert_to_tensor(x2_list, tf.float32)), tf.squeeze(tf.convert_to_tensor(y, tf.float32))


def make_generators(model, train_df, val_df, test_df, params, chexpert=False, ssl=False):
    assert model['model_type'] in ['keras', 'pytorch', 'fastai']
    
    if model['model_type'] == "keras":
        if model["model_name"] == "siamese_net" and model['pretrained']:
            Generator = SiameseDataGen
        else:
            Generator = KerasDataGen

        if model["model_name"]=="ssl_am" and model["supervised"]==False:
          ssl = True
          #train_aug_fn = val_aug_fn # eliminate additional augmentations during ssl
        else:
          ssl = False

        print('ssl', ssl)

        train_dg = Generator(train_df, params["batchsize"], transforms=train_aug_fn, k=params["k"], chexpert=chexpert, ssl=ssl)   
        val_dg = Generator(val_df, params["batchsize"], transforms=val_aug_fn, k=params["k"], chexpert=chexpert, ssl=ssl)
        test_dg = Generator(test_df, params["batchsize"], transforms=val_aug_fn, k=params["k"], chexpert=chexpert, ssl=ssl)

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

