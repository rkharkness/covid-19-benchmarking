"""code adapted from: https://github.com/chirag126/CoroNet"""

# import required libraries
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import models, transforms
import torch.nn.functional as F

from torch.optim import Adam

import pandas as pd
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

import time
import csv
import argparse


class FPAE(nn.Module):
    def __init__(self):
        super(FPAE, self).__init__()
        self.bn0 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=24, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=24)
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=16, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(num_features=16)
        self.conv6 = nn.Conv2d(in_channels=16, out_channels=12, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(num_features=12)
        self.conv7 = nn.Conv2d(in_channels=12, out_channels=8, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(num_features=8)
        self.down = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=2)
        self.conv_smooth1 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.conv_smooth2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1)
        self.conv_smooth3 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.convtrans1 = nn.ConvTranspose2d(in_channels=8, out_channels=12, kernel_size=3, padding=1)
        self.convtrans2 = nn.ConvTranspose2d(in_channels=12, out_channels=16, kernel_size=3, padding=1)
        self.convtrans3 = nn.ConvTranspose2d(in_channels=16, out_channels=24, kernel_size=3, padding=1)
        self.convtrans4 = nn.ConvTranspose2d(in_channels=24, out_channels=32, kernel_size=3, padding=1)
        self.convtrans5 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.convtrans6 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, padding=1)
        self.convtrans7 = nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=3, padding=1)

    def forward(self, x):
        x_small = x.clone()
        x_clone = x_small.clone()

        # ============ Encoder ===========
        # ====== Bottom Up Layers =====
        x = self.bn0(x_small)
        res1_x = self.conv1(x)
        x = self.relu(res1_x)
        x = self.bn1(x)
        res2_x = self.conv2(x)
        x = self.relu(res2_x)
        x = self.bn2(x)
        res3_x = self.conv3(x)
        x = self.relu(res3_x)
        x = self.bn3(x)
        _, _, H1, W1 = x.size()

        ### ======= Branch network ======
        x_d1 = self.down(x)  # 128x128
        _, _, H2, W2 = x_d1.size()
        x_d2 = self.down(x_d1)  # 64x64
        _, _, H3, W3 = x_d2.size()
        x_d3 = self.down(x_d2)  # 32x32

        ### ======= First Branch =======
        res4_x = self.conv4(x)
        x = self.relu(res4_x)
        x = self.bn4(x)
        res5_x = self.conv5(x)
        x = self.relu(res5_x)
        x = self.bn5(x)
        res6_x = self.conv6(x)
        x = self.relu(res6_x)
        x = self.bn6(x)
        res7_x = self.conv7(x)
        x = self.relu(res7_x)
        x = self.bn7(x)

        ### ======= Second Branch ========
        x_d1 = self.conv4(x_d1)
        x_d1 = self.relu(x_d1)
        x_d1 = self.bn4(x_d1)
        x_d1 = self.conv5(x_d1)
        x_d1 = self.relu(x_d1)
        x_d1 = self.bn5(x_d1)
        x_d1 = self.conv6(x_d1)
        x_d1 = self.relu(x_d1)
        x_d1 = self.bn6(x_d1)
        x_d1 = self.conv7(x_d1)
        x_d1 = self.relu(x_d1)
        z1 = self.bn7(x_d1)
        x_d1 = self.upsample(z1, size=(H1, W1))

        ### ======= Third Branch ========
        x_d2 = self.conv4(x_d2)
        x_d2 = self.relu(x_d2)
        x_d2 = self.bn4(x_d2)
        x_d2 = self.conv5(x_d2)
        x_d2 = self.relu(x_d2)
        x_d2 = self.bn5(x_d2)
        x_d2 = self.conv6(x_d2)
        x_d2 = self.relu(x_d2)
        x_d2 = self.bn6(x_d2)
        x_d2 = self.conv7(x_d2)
        x_d2 = self.relu(x_d2)
        z2 = self.bn7(x_d2)
        x_d2 = self.upsample(z2, size=(H2, W2))
        x_d2 = self.upsample(x_d2, size=(H1, W1))

        ### ======= Fourth Branch ========
        x_d3 = self.conv4(x_d3)
        x_d3 = self.relu(x_d3)
        x_d3 = self.bn4(x_d3)
        x_d3 = self.conv5(x_d3)
        x_d3 = self.relu(x_d3)
        x_d3 = self.bn5(x_d3)
        x_d3 = self.conv6(x_d3)
        x_d3 = self.relu(x_d3)
        x_d3 = self.bn6(x_d3)
        x_d3 = self.conv7(x_d3)
        x_d3 = self.relu(x_d3)
        z3 = self.bn7(x_d3)
        x_d3 = self.upsample(z3, size=(H3, W3))
        x_d3 = self.upsample(x_d3, size=(H2, W2))
        x_d3 = self.upsample(x_d3, size=(H1, W1))

        ### ======= Concat maps ==========
        x = torch.cat((x, x_d1, x_d2, x_d3), 1)

        x = self.conv_smooth1(x)
        x = self.conv_smooth2(x)
        x = self.conv_smooth3(x)
       
        ### ============ Decoder ==========
        x = self.convtrans1(x)
        x = self.relu(x+res6_x)
        x = self.convtrans2(x)
        x = self.relu(x+res5_x)
        x = self.convtrans3(x)
        x = self.relu(x+res4_x)
        x = self.convtrans4(x)
        x = self.relu(x+res3_x)
        x = self.convtrans5(x)
        x = self.relu(x+res2_x)
        x = self.convtrans6(x)
        x = self.relu(x+res1_x)
        x = self.convtrans7(x)
        x = x + x_clone
        x = self.sigmoid(x)

        return [x, x_small, z3] # recon img, orig, latent

    def upsample(self, x, size):
        up = nn.Upsample(size=size, mode="bilinear")
        return up(x)

class Classifier(nn.Module):
    model_name = 'coronet'
    model_type = 'pytorch'
    optimizer = 'adam'
    loss_fn = {'ae': nn.MSELoss(), 'classifier':nn.BCELoss()}

    def __init__(self, trial=None):
        super(Classifier, self).__init__()

        self.num_classes = 1
        self.resnet = models.resnet18(pretrained=True)
        self.features_conv = nn.Sequential(*list(self.resnet.children())[:-2])

        self.avgpool = self.resnet.avgpool

        num_ftrs = self.resnet.fc.in_features      
        self.resnet.fc = nn.Linear(num_ftrs, self.num_classes)
        self.classifier = self.resnet.fc
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features_conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        x = self.sigmoid(x)
        return x

class CoroNet(nn.Module):
    supervised = False
    def __init__(self, trial=None, supervised=supervised):
        super(CoroNet, self).__init__()

        self.supervised = supervised

        self.fpae = FPAE()
        self.classifier = Classifier()

    def forward(self, x):
        if self.supervised:
            output = self.classifier(x)
        else:
            output = self.fpae(x) # return list of outputs

        return output 
        
            

    





