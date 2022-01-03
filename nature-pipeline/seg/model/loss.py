import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class MultiLabelSoftmaxLoss(nn.Module):
    def __init__(self, config):
        super(MultiLabelSoftmaxLoss, self).__init__()
        self.task_num = config.getint("model", "output_dim")
        self.criterion = []
        for a in range(0, self.task_num):
            try:
                ratio = config.getfloat("train", "loss_weight_%d" % a)
                self.criterion.append(
                    nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.0, ratio], dtype=np.float32)).cuda()))
            except Exception as e:
                self.criterion.append(nn.CrossEntropyLoss())

    def forward(self, outputs, labels):
        loss = 0
        for a in range(0, len(outputs[0])):
            o = outputs[:, a, :].view(outputs.size()[0], -1)
            loss += self.criterion[a](o, labels[:, a])

        return loss


class NLL_loss(nn.Module):
    def __init__(self, config, role="stage_2"):
        super(NLL_loss, self).__init__()
        self.role = role
        if role == "stage_2":
            self.weights = torch.Tensor([int(val) for val in config.get("model", "stage_2_weights").split(",")]).cuda()

    def forward(self, outputs, labels):
        labels = labels.long()
        outputs = torch.log(torch.clamp(outputs, 1e-6, 1 - 1e-6))
        if self.role != "stage_2":
            return nn.NLLLoss()(outputs, labels)
        return nn.NLLLoss(weight=self.weights, reduction="mean")(outputs, labels)


class weighted_CE(nn.Module):
    def __init__(self, config):
        super(weighted_CE, self).__init__()

    def forward(self, outputs, labels, weight_map):
        outputs = outputs.float()
        labels = labels.float()
        bs = outputs.shape[0]
        outputs = torch.clamp(outputs, 1e-6, 1 - 1e-6)
        loss = - labels * torch.log(outputs) - (1 - labels) * torch.log(1 - outputs)
        loss = torch.sum(loss, dim=1)
        loss = (loss * weight_map).view(bs, -1)
        loss = torch.sum(loss, dim=1)
        loss = torch.mean(loss)
        return loss


class BCEWithLogitsLoss2d(nn.Module):

    def __init__(self, config, size_average=True):
        super(BCEWithLogitsLoss2d, self).__init__()
        self.size_average = size_average
        self.weight = [float(val) for val in config.get("model", "stage_2_weights").split(",")]
        assert len(self.weight) == 5

    def forward(self, predict, target):
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 4
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(2), "{0} vs {1} ".format(predict.size(2), target.size(2))
        assert predict.size(3) == target.size(3), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        loss = 0
        for idx in range(predict.shape[1]):
            label_idx = target[:, idx, :, :]
            pred_idx = torch.clamp(predict[:, idx, :, :], 1e-6, 1 - 1e-6)
            loss_idx =  - self.weight[idx] * label_idx * torch.log(pred_idx) - (1 - label_idx) * torch.log(1 - pred_idx)
            loss += torch.mean(loss_idx)
        return loss / 5


class EL_DiceLoss(nn.Module):
    def __init__(self, config):
        super(EL_DiceLoss, self).__init__()
        self.smooth = 1
        self.class_num = config.getint("model", "stage_2_output_class_num")
        self.gamma = 0.5

    def forward(self, outputs, labels):
        softmax_label = False
        if len(labels.shape) == 3:
            softmax_label = True
        Dice = Variable(torch.Tensor([0]).float()).cuda()
        for i in range(1, self.class_num):
            input_i = outputs[:,i,:,:]
            if softmax_label:
                target_i = (labels == i).float()
            else:
                target_i = (labels[:, i, :, :] == 1).float()
            intersect = (input_i*target_i).sum()
            union = torch.sum(input_i) + torch.sum(target_i)
            if target_i.sum() == 0:
                dice = Variable(torch.Tensor([1]).float()).cuda()
            else:
                dice = (2 * intersect + self.smooth) / (union + self.smooth)
            Dice += (-torch.log(dice)) ** self.gamma
        dice_loss = Dice/(self.class_num - 1)
        return dice_loss[0]


class DiceLoss(nn.Module):
    def __init__(self, config):
        super(DiceLoss, self).__init__()
        self.smooth = 1
        self.class_num = config.getint("model", "stage_2_output_class_num")

    def forward(self, outputs, labels):
        softmax_label = False
        if len(labels) == 3: softmax_label = True
        Dice = Variable(torch.Tensor([0]).float()).cuda()
        for i in range(1, self.class_num):
            input_i = outputs[:,i,:,:]
            if softmax_label:
                target_i = (labels == i).float()
            else:
                target_i = (labels[:, i, :, :] == 1).float()
            intersect = (input_i*target_i).sum()
            union = torch.sum(input_i) + torch.sum(target_i)
            if target_i.sum() == 0:
                dice = Variable(torch.Tensor([1]).float()).cuda()
            else:
                dice = (2 * intersect + self.smooth) / (union + self.smooth)
            Dice += dice
        dice_loss = 1 - Dice/(self.class_num - 1)
        return dice_loss[0]


