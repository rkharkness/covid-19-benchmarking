from pathlib import Path

import numpy as np
import pandas as pd

import skimage.io
import torch
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from fastai.vision import pil2tensor


class ImageDataset(Dataset):
    def __init__(self, images, data_dir, size):
        super(ImageDataset, self).__init__()
        if isinstance(images,pd.DataFrame):
            images = images['image_path']
        self._images = images
        self._size = size
        self._data_dir = Path(data_dir)


    def _array2tensor(self, image_arr):
        size = self._size
        image = Image.fromarray(image_arr)
        if isinstance(size, int):
            image = image.resize((size,size))
        elif isinstance(self._size, tuple) or isinstance(self._size, list):
            image = image.resize(size)
        image = image.convert('RGB')
        image_t = pil2tensor(image, dtype=np.float32).div_(255)
        return image_t


    def __len__(self):
        return len(self._images)


    def __getitem__(self, idx):
        image = self._images[idx]
        if isinstance(image,str) or isinstance(image,Path):
            image = skimage.io.imread(image)
        image = self._array2tensor(image)
        label = 0
        return image, label
