import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model.loss import weighted_CE

from tools.accuracy_init import accuracy_function_dic
from model.cv.Unet_zoos import UNet_raw, UNet_Nested, UNet_Nested_dilated
from model.cv.FCN.fcn8s import FCN8s
from model.cv.segnet import SegNet
from model.cv.deeplab.deeplab import DeepLabV3
from model.loss import NLL_loss, EL_DiceLoss, weighted_CE, BCEWithLogitsLoss2d, DiceLoss


activation_list = {
    "sigmoid": nn.Sigmoid(),
    "softmax": nn.Softmax(dim = 1)
}

model_list = {
    "deeplabv3": DeepLabV3,
    "SegNet": SegNet,
    "FCN8s": FCN8s,
    "Dilated_U": UNet_Nested_dilated
}


class UNet(nn.Module):
    def __init__(self, config, gpu_list, role="main", *args, **params):
        super(UNet, self).__init__()
        self.role = role
        self.input_channel_num = config.getint("model", "input_channel_num")
        if self.role == "main":
            self.output_class_num = config.getint("model", "output_class_num")
            self.class_weights = [int(val) for val in config.get("model", "class_weights").split(",")]
        elif self.role == "stage_1":
            self.output_class_num = config.getint("model", "stage_1_output_class_num")
            self.class_weights = [int(val) for val in config.get("model", "stage_1_weights").split(",")]
        elif self.role == "stage_2":
            self.output_class_num = config.getint("model", "stage_2_output_class_num")
            self.class_weights = [int(val) for val in config.get("model", "stage_2_weights").split(",")]
        else:
            raise NotImplementedError

        self.class_weights = torch.from_numpy(np.array(self.class_weights)).cuda().float()
        self.use_weight_map = config.getboolean("data", "use_weight_map") if self.role == "stage_2" else False  #只考虑stage-2

        if self.role == "stage_2":
            module_name = config.get("model", "stage_2_model")
            self.net = model_list[module_name](n_channels=self.input_channel_num,
                                               n_classes=self.output_class_num)
        elif self.role == "stage_1":
            self.net = pre_UNet(n_channels=self.input_channel_num,
                             n_classes=self.output_class_num)
        elif self.role == "main":
            module_name = config.get("model", "module_name")
            self.net = model_list[module_name](n_channels=self.input_channel_num,
                                               n_classes=self.output_class_num)

        #activation final layer
        if self.role == "main":
            self.activation_type = config.get("model", "activation")
            self.activation = activation_list[config.get("model", "activation")]
        elif self.role == "stage_1":
            self.activation_type = config.get("model", "stage_1_activation")
            self.activation = activation_list[config.get("model", "stage_1_activation")]
        elif self.role == "stage_2":
            self.activation_type = config.get("model", "stage_2_activation")
            self.activation = activation_list[config.get("model", "stage_2_activation")]
        else:
            raise NotImplementedError

        self.criterion = self.get_criterion(config)

        if self.activation_type == "sigmoid":
            self.accuracy_function = accuracy_function_dic["IoU_sigmoid"]
        else:
            self.accuracy_function = accuracy_function_dic["IoU_softmax"]

    def get_criterion(self, config):
        if self.role == "main":
            return nn.BCELoss(reduction="sum")
        if self.role == "stage_1":
            return [nn.CrossEntropyLoss()]
        if self.activation_type == "softmax":
            return [NLL_loss(config, self.role),
                    ]
        else:
            return [BCEWithLogitsLoss2d(config),
                    ]

    def init_multi_gpu(self, device, config, *args, **params):
        self.net = nn.DataParallel(self.bert, device_ids=device)

    def forward(self, data, config, gpu_list, acc_result, mode):
        x = data['input']
        logits = self.net(x)
        activated_logits = self.activation(logits)
        if self.activation_type == "sigmoid":
            prediction = torch.ge(activated_logits, 0.5).long()
        else:
            prediction = torch.argmax(activated_logits, dim=1, keepdim=False)

        if self.role in ["stage_1", "stage_2"]:
            if mode in ["train", "valid"]:
                label = data["label"]
                loss = 0
                if self.use_weight_map:
                    loss += weighted_CE(config)(activated_logits, label, data["weight_map"])
                for each_criterion in self.criterion:
                    if self.role == "stage_1":
                        loss += nn.CrossEntropyLoss()(logits, label.long())
                    else:
                        loss += each_criterion(activated_logits.float(), label.float())
                acc_result = self.accuracy_function(prediction, label, acc_result, self.output_class_num)
                return {"logits": logits, "prediction": prediction, "loss": loss, "acc_result": acc_result}
            return {"logits": logits, "prediction": prediction}


        if "label" in data.keys():
            label = data["label"].float()
            loss = self.criterion(activated_logits, label)
            acc_result = self.accuracy_function(prediction, label, config, acc_result, self.output_class_num)
            return {"loss": loss, "acc_result": acc_result, "prediction": prediction}

        return {"prediction": prediction}


class pre_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(pre_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


