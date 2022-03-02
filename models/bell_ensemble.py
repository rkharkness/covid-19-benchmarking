import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils import data
import numpy as np
import torchvision
from  numpy import exp,absolute
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import math


class BellEnsemble(nn.Module):
    def __init__(self) -> None:
        super(BellEnsemble, self).__init__()

    def vgg(self):
        model_vgg = torchvision.models.vgg16(pretrained=True)
        for param in model_vgg.parameters():
            param.requires_grad = True
        model_vgg.classifier[6] = nn.Linear(4096, 1)
        model_vgg = model_vgg.to('cuda')

        return model_vgg

    def alexnet(self):
        model_alexnet = torchvision.models.alexnet(pretrained=True)
        for param in model_alexnet.parameters():
            param.requires_grad = True
        model_alexnet.classifier[6] = nn.Linear(4096, 1)
        model_alexnet = model_vgg.to('cuda')

        return model_alexnet
    
    def resnet(self):
        model_resnet = torchvision.models.resnet18(pretrained=True)
        for param in model_resnet.parameters():
            param.requires_grad = True
        num_ftrs = model_resnet.fc.in_features
        model_resnet.fc = nn.Linear(num_ftrs, 1) 
        model_resnet = model_resnet.to('cuda')
        return model_resnet

    def dnet(self):
        model_dnet = torchvision.models.densenet161(pretrained=True)
        for param in model_dnet.parameters():
            param.requires_grad = True
        model_dnet.classifier = nn.Linear(2208, 1)
        model_dnet = model_dnet.to('cuda')
        return model_dnet



criterion = nn.CrossEntropyLoss()

optimizer_conv = optim.SGD(model_resnet.parameters(), lr=0.001, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=70, gamma=0.1)
model_resnet = train_model(model_resnet, criterion, optimizer_conv, exp_lr_scheduler,num_epochs=30)
torch.save(model_resnet, '/content/drive/My Drive/kaggle_resnet.pth')

optimizer_conv = optim.SGD(model_vgg.parameters(), lr=0.001, momentum=0.9)
model_vgg = train_model(model_vgg, criterion, optimizer_conv, exp_lr_scheduler,num_epochs=20)
torch.save(model_vgg, '/content/drive/My Drive/kaggle_vgg.pth')

optimizer_conv = optim.SGD(model_dnet.parameters(), lr=0.001, momentum=0.9)
model_dnet = train_model(model_dnet, criterion, optimizer_conv, exp_lr_scheduler,num_epochs=20)
torch.save(model_dnet, '/content/drive/My Drive/kaggle_dense.pth')

def ensemble (op_classifiers):
    cf = 0.0
    for c in op_classifiers:
        c = c.cpu().numpy()
        w = np.identity(c.shape[0])
        i = 0
        for x in c:
            w[i][i] = np.sum(exp((x-0.5)*(x-0.5)/0.5))
            i += 1
        cf += np.matmul(w,c)
    cf = (cf)/len(op_classifiers)
    return torch.from_numpy(cf)

soft = nn.Softmax(dim=1)
running_corrects = 0

for inputs, labels in dataloaders['val']:
    inputs = inputs.to(device)
    labels = labels.to(device)

    with torch.set_grad_enabled(False):
        outputs_resnet = soft(model_resnet(inputs))
        outputs_vgg = soft(model_vgg(inputs))
        outputs_dnet = soft(model_dnet(inputs))
        outputs_alexnet = soft(model_alexnet(inputs))
        outputs = ensemble({outputs_dnet,outputs_resnet,outputs_vgg})
        _, preds = torch.max(outputs, 1)
        
        running_corrects += torch.sum(preds == labels.data)
print(running_corrects/dataset_sizes['val'])