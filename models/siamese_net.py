# import required libraries
from curses.ascii import SI
import pandas as pd
import os
import cv2
import time
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torchsummary import summary

from torch.autograd import Variable

import pandas as pd
from tqdm import tqdm

import albumentations as A #need version 0.4.6
from albumentations.pytorch import ToTensorV2

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=1.25):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        # euclidean distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)
        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 48, 3)
        self.conv5 = nn.Conv2d(48, 32, 3)
        self.conv6 = nn.Conv2d(32, 16, 3)


        self.fc1 = nn.Linear(16*6*6, 32)
        self.fc2 = nn.Linear(32, 1)

        self.dropout = nn.Dropout(p=0.5)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = F.relu(self.conv6(x))
        x = x.view(-1, 16*6*6)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 8, 3),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, 3),
            nn.Conv2d(16, 32, 3),
            nn.Conv2d(32, 48, 3),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(48, 32, 3),
            nn.Conv2d(32, 16, 3)
          )


        self.fc1 = nn.Linear(16*6*6, 32)
        self.fc2 = nn.Linear(32, 1)

        self.dropout = nn.Dropout(p=0.5)
      
    def forward_once(self, x):
        output = self.cnn1(x)
        print(output.shape)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return torch.sigmoid(output)

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        dist = torch.abs(output1 - output2) #try fc layer -5120 to 1 instead
        dist = self.fc2(dist)
        return torch.sigmoid(dist)

    def build_model(self):
        model = SiameseNetwork()
        model = {'model':model, 'optimizer':self.optimizer, 'loss_fn':self.loss_fn, 'lr':self.lr,
        'model_name':self.model_name, 'model_type':self.model_type, 'supervised':self.supervised}
        return model

if __name__ == "__main__":
    siamese_net = SiameseNetwork()
    model = siamese_net.build_model()
    print(summary(model))
