import torch
import torch.nn as nn
import os
from PIL import Image
import time
import torchvision
from torchvision import transforms
import copy
import torch.optim as optim
from torch.optim import lr_scheduler
import pandas as pd
import torch.nn.functional as F
#from tkinter import Tcl
# import Vggbn_conv as Vgg
import pickle
import numpy as np
import PIL
from efficientnet_pytorch import EfficientNet
import pretrainedmodels
from torch.utils.tensorboard import SummaryWriter
from barbar import Bar
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FusionDataset(object):
    def __init__(self, csv_file, dataset_1, dataset_2, transform1 = None, transform2 = None):
        super(FusionDataset, self).__init__()
        
        self.frame = pd.read_csv(csv_file, header = None)
        # self.frame_2 = pd.read_csv(csv_file_2)
        self.transform1 = transform1
        self.transform2 = transform2
        
        self.dataset_1 = dataset_1
        self.dataset_2 = dataset_2
                  
    def __len__(self):
        return len(self.frame)
    
    def __getitem__(self, idx):        
        #        img_path =  os.path.join(self.root, self.frame.loc[idx][0])
        cat = self.frame.loc[idx][0].split(' ')
        
        img_path_1 = os.path.join(cat[1], self.dataset_1, cat[2])
        img_path_2 = os.path.join(cat[1], self.dataset_2, cat[2])
        
        # img_path = cat[1] + cat[2] #wrong
        # img_path = cat[1] + ' ' + cat[2] #wrong
        img_path = cat[1] + '/' + cat[2]
        
        label = int(cat[3])
#		img = io.imread(img_path)
        img_1 = Image.open(img_path_1).convert('RGB')
        img_2 = Image.open(img_path_2)
        
        # if self.transform is not None:
        img_1 = self.transform1(img_1)
        img_2 = self.transform2(img_2)           
        return img_1, img_2, label, img_path
    


def online_mean_and_sd(loader):
    """Compute the mean and sd in an online fashion

        Var[x] = E[X^2] - E^2[X]
    """

    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    
    for iidx, content in enumerate(Bar(loader)):
        data, _, _ = content
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum/num_batches
    std = (channels_squared_sum/num_batches - mean**2)**0.5
    
    return mean, std

def fusion_online_mean_and_sd(loader):
    """Compute the mean and sd in an online fashion

        Var[x] = E[X^2] - E^2[X]
    """

    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    channels_sum2, channels_squared_sum2, num_batches2 = 0, 0, 0
    
    for iidx, content in enumerate(Bar(loader)):
        data, data2, _, _ = content
        
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
        
        channels_sum2 += torch.mean(data2, dim=[0, 2, 3])
        channels_squared_sum2 += torch.mean(data2**2, dim=[0,2,3])
        num_batches2 += 1
    
    mean = channels_sum/num_batches
    std = (channels_squared_sum/num_batches - mean**2)**0.5
    
    mean2 = channels_sum2/num_batches2
    std2 = (channels_squared_sum2/num_batches2 - mean2**2)**0.5
    
    return mean, std, mean2, std2