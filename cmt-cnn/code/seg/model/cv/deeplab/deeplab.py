import torch
import torch.nn as nn
import torch.nn.functional as F

import os

from model.cv.deeplab.resnet import ResNet18_OS16, ResNet34_OS16, ResNet50_OS16, ResNet101_OS16, ResNet152_OS16, ResNet18_OS8, ResNet34_OS8
from model.cv.deeplab.aspp import ASPP, ASPP_Bottleneck

class DeepLabV3(nn.Module):
    def __init__(self, n_channels = 3, n_classes = 3):
        super(DeepLabV3, self).__init__()
        self.resnet = ResNet50_OS16()
        self.aspp = ASPP_Bottleneck(num_classes=n_classes)

    def forward(self, x):

        h = x.size()[2]
        w = x.size()[3]

        feature_map = self.resnet(x)

        output = self.aspp(feature_map)

        output = F.upsample(output, size=(h, w), mode="bilinear")

        return output
