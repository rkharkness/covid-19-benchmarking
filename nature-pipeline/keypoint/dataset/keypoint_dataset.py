import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import pandas as pd

from pathlib import Path
from functools import lru_cache
from itertools import chain

import skimage
import skimage.io
import albumentations as albu


def df2label(df):
    cols = [col for col in df.columns if col.startswith('kp_')]
    cols = sorted(cols)
    points = df[cols].values
    n, k2 = points.shape
    points = points.reshape(n,k2//2,2)
    return cols, points


class KeypointDataset(Dataset):
    def __init__(self, df, image_dir, height, width, p_aug=0.5):
        super(KeypointDataset, self).__init__()
        
        image_dir = str(image_dir)
        self.df = df
        
        self.image_dir = Path(image_dir)
        self.image_hight = height
        self.image_width = width

        self.xs = df['image_path'].values

        try:
            cols, ys = df2label(df)
            ys = ys.astype('float32')
            self.n_locations = ys.shape[1]
            self.cols = cols
        except:
            ys = None
        self.ys = ys

        transforms = [
            albu.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=30, p=0.5),
            albu.RandomBrightness(limit=0.5, p=0.5),
        ]
        transform = albu.Compose(transforms,p=p_aug,keypoint_params={'format': 'xy'})
        self.transform_aug = transform

        transforms = [
            albu.Resize(height=self.image_hight, width=self.image_width),
        ]
        transform = albu.Compose(transforms)
        self.transform_scale = transform

        self.to_tensor_func = torchvision.transforms.ToTensor()
    

    def _load_image(self, image_path):
        filepath = self.image_dir/image_path
        img = skimage.io.imread(filepath)
        if (len(img.shape) == 2) or (img.shape[2] == 1):
            img = skimage.color.grey2rgb(img)
        return img
        

    def __len__(self):
        return len(self.xs)


    def __getitem__(self, idx):
        image = self._load_image(self.xs[idx])
        h, w = image.shape[:2]
        shape = np.array([w,h])
        
        if self.ys is None:
            x = image
            y = 0
        else:
            points = self.ys[idx]
            points = points*shape
            result = self.transform_aug(image=image,keypoints=points)
            image_new = result['image']
            points_new = result['keypoints']
            if len(points_new) == len(points):
                image = image_new
                points = points_new
            x = self.transform_scale(image=image)['image']
            y = points / shape
            y = y.astype('float32')
            y = 2 * y - 1
            x = self.to_tensor_func(x)
        return x, y
