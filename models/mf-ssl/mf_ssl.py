import torch
import os
from PIL import Image

#from tkinter import Tcl
# import Vggbn_conv as Vgg
import pickle
import numpy as np
import pandas as pd
import PIL
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import pretrainedmodels
from torch.utils.tensorboard import SummaryWriter
from barbar import Bar
import torch.nn.functional as F

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

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Late_Fusion_Net(nn.Module):
    def __init__(self, NN, my_model1, my_model2, num_classes):
        super(Late_Fusion_Net, self).__init__()        
        self.NN = NN
        self.num_classes = num_classes

        if self.NN == 'Res50': 
                self.feature1 = nn.Sequential(*list(my_model1.children())[:-1])
                self.feature2 = nn.Sequential(*list(my_model2.children())[:-1])
                                
                self.classifier1 = my_model1.fc
                self.classifier2 = my_model2.fc
                
        elif self.NN == 'xception' or self.NN == 'inceptionresnetv2':
                self.feature1 = nn.Sequential(*list(my_model1.children())[:-1])
                self.feature2 = nn.Sequential(*list(my_model2.children())[:-1])
                                
                self.classifier1 = my_model1.last_linear
                self.classifier2 = my_model2.last_linear
                
                # self.classifier3 = nn.Linear(self.embedding_dim, self.num_classes)
        
                
        elif self.NN == 'efficientnet-b4':
                self.feature1 = my_model1
                self.feature2 = my_model2
                
                self.avgpool =  my_model1._avg_pooling
                self.dropout = my_model1._dropout
                self.swish = my_model1._swish
                
                # ori_model = EfficientNet.from_pretrained('efficientnet-b4')
                self.classifier1 = my_model1._fc
                self.classifier2 = my_model2._fc
                
                # self.classifier2 = nn.Linear(self.embedding_dim, num_classes)
                      
        # else:           
        elif self.NN == 'Alexnet' or self.NN == 'Vgg16bn':
            self.feature1 = my_model1.features
            self.feature2 = my_model2.features
            
            self.avgpool = my_model1.avgpool
            
            self.classifier1 = my_model1.classifier
            self.classifier2 = my_model2.classifier
            

        elif self.NN =='Vgg16bn_bn_conv':
            self.feature1 = my_model1.features
            self.feature2 = my_model2.features            
            
            self.classifier1 = my_model1.classifier
            self.classifier2 = my_model2.classifier

                                       
    def forward(self,x,y):       
        # if (self.NN == 'Alexnet' or self.NN == 'Vgg16' or self.NN == "Vgg16bn"
        #     or self.NN == 'Vgg19' or self.NN == 'Vgg19bn'):
        if self.NN == 'efficientnet-b4':
            x1 = self.feature1.extract_features(x)            
            x2 = self.feature2.extract_features(y)
        else:   
            x1 = self.feature1(x)            
            x2 = self.feature2(y)
        
        
        # if self.mtd == 'Sum':
        if (self.NN == 'Alexnet' or self.NN == 'Vgg16' or self.NN == "Vgg16bn"
            or self.NN == 'Vgg19' or self.NN == 'Vgg19bn'):
            
            x1 = self.avgpool(x1)
            x1 = x1.view(x1.size(0),-1)
            
            x2 = self.avgpool(x2)
            x2 = x2.view(x2.size(0),-1)
                        
            x1 = self.classifier1(x1) 
            x2 = self.classifier2(x2)            
        
        elif self.NN == 'Vgg16bn_bn_conv':
            
            x1 = self.classifier1(x1)
            x1 = F.avg_pool2d(x1, (x1.shape[-2], x1.shape[-1])).squeeze(2).squeeze(2)
            x2 = self.classifier2(x2)
            x2 = F.avg_pool2d(x2, (x2.shape[-2], x2.shape[-1])).squeeze(2).squeeze(2)             
        
        elif self.NN == 'xception':
            
            x1 = F.adaptive_avg_pool2d(x1, (1, 1))
            x1 = x1.view(x1.size(0), -1)
            
            x2 = F.adaptive_avg_pool2d(x2, (1, 1))
            x2 = x2.view(x2.size(0), -1)
        
            x1 = self.classifier1(x1)
            # x1 = F.softmax(x1)
            # print (x1)                
            x2 = self.classifier2(x2)
            # x2 = F.softmax(x2)
            # print (x2)              
        elif self.NN == 'inceptionresnetv2':
            
            # x1 = F.adaptive_avg_pool2d(x1, (8, 8))
            x1 = x1.view(x1.size(0), -1)
            
            # x2 = F.adaptive_avg_pool2d(x2, (8, 8))
            x2 = x2.view(x2.size(0), -1)
            
            x1 = self.classifier1(x1)                
            x2 = self.classifier2(x2)
            
        elif self.NN == 'efficientnet-b4':
            
            x1 = self.dropout(self.avgpool(x1))
            x1 = x1.view(x1.size(0), -1)
            x1 = self.swish(self.classifier1(x1))
            
            x2 = self.dropout(self.avgpool(x2))
            x2 = x2.view(x2.size(0), -1)
            x2 = self.swish(self.classifier2(x2))
                        
        else: # Res50 

            x1 = x1.view(x1.size(0),-1)                      
            x2 = x2.view(x2.size(0),-1)

            x1 = self.classifier1(x1) 
            x2 = self.classifier2(x2)  
            
        x3 = x1+x2
            
        return x3