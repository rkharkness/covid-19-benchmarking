import argparse
from statistics import mode
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import os
import copy
import torchvision
import torch.nn.functional as F
import random
import numpy as np

import argparse
from statistics import mode
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import os
import copy
import torchvision
import torch.nn.functional as F
import random
import numpy as np

import torchsummary

class LabelSmoothingLoss(nn.Module):
    """NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.3):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

EPSILON = 1e-10 #12

def batch_augment(images, attention_map, mode='mixup', theta=0.5, padding_ratio=0.1):
    batches, _, imgH, imgW = images.size()

    if mode == 'mixup':
        auged_images = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_c = random.uniform(*theta) * atten_map.max()
            else:
                theta_c = theta * atten_map.max()

            mixup_mask = F.interpolate(atten_map, size=(imgH, imgW), mode='bilinear') >= theta_c
            nonzero_indices = torch.nonzero(mixup_mask[0, 0, :,:], as_tuple =False)
            height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
            height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
            width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
            width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)
            upsampled_patch = F.interpolate(images[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                                    size=(imgH, imgW),  mode='bilinear')
            auged_image = images[batch_index:batch_index + 1,:,:,:]*0.6 + upsampled_patch*0.4
            # import matplotlib.pyplot as plt
            # plt.subplot(2,1,1)
            # plt.imshow(upsampled_patch[0][0].squeeze().cpu().numpy(), cmap='gray')
            # plt.subplot(2,1,2)
            # plt.imshow(auged_image[0][0].squeeze().cpu().numpy(), cmap='gray')
            # plt.show()


            auged_images.append(auged_image)
        auged_images = torch.cat(auged_images, dim=0)
        return auged_images

    elif mode == 'crop':
        crop_images = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_c = random.uniform(*theta) * atten_map.max()
            else:
                theta_c = theta * atten_map.max()

            crop_mask = F.interpolate(atten_map, size=(imgH, imgW), mode='bilinear') >= theta_c
            nonzero_indices = torch.nonzero(crop_mask[0, 0, :,:], as_tuple =False)
            height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
            height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
            width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
            width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)
            crop_images.append(F.interpolate(images[batch_index:batch_index + 1, :,
                                             height_min:height_max, width_min:width_max],
                                             size=(imgH, imgW),
                                             mode='bilinear'))

        crop_images = torch.cat(crop_images, dim=0)
        return crop_images


    elif mode == 'dim':
        drop_masks = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_d = random.uniform(*theta) * atten_map.max()
            else:
                theta_d = theta * atten_map.max()
            # drop_mask = F.upsample_bilinear(atten_map, size=(imgH, imgW)) < theta_d
            # drop_mask = drop_masks.float() * np.ones_like(drop_mask.numpy()) + 0.001
            # drop_masks.append(drop_mask)
            drop_masks.append(F.interpolate(atten_map, size=(imgH, imgW), mode='bilinear') < theta_d)
        drop_masks = torch.cat(drop_masks, dim=0)
        drop_images = images * (drop_masks.float() * torch.ones_like(drop_masks) + 0.001)
        return drop_images

    elif mode == 'patch':
        multi_image = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_c = random.uniform(*theta) * atten_map.max()
            else:
                theta_c = theta * atten_map.max()
            crop_mask = F.interpolate(atten_map, size=(imgH, imgW), mode='bilinear') >= theta_c
            nonzero_indices = torch.nonzero(crop_mask[0, 0, :,:], as_tuple =False)
            height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
            height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
            width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
            width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)
            patch = images.clone()[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max]
            auged_image = images.clone()[batch_index:batch_index + 1, :, ...]
            H_patch = random.randint(0, imgH-(height_max-height_min))
            W_patch = random.randint(0, imgW-(width_max-width_min))
            auged_image[:, :,H_patch:H_patch+(height_max-height_min), W_patch:W_patch+(width_max-width_min)] = patch
            multi_image.append(auged_image)

            # import matplotlib.pyplot as plt
            # plt.subplot(2,1,1)
            # plt.imshow(patch[0][0].squeeze().cpu().numpy(), cmap='gray')
            # plt.subplot(2,1,2)
            # plt.imshow(auged_image[0][0].squeeze().cpu().numpy(), cmap='gray')
            # plt.savefig("/content/fig.png")
            # plt.show()

        multi_images = torch.cat(multi_image, dim=0)
        return multi_images

    else:
        raise ValueError('Expected mode in [\'crop\', \'drop\'], but received unsupported augmentation method %s' % mode)


class WeightedBCE(nn.Module):
    def  __init__(self) -> None:
        super(WeightedBCE, self).__init__()
        self.weights = {'neg':1.32571275, 'pos':0.80276873}
    
    def forward(self, target, output):
        output = torch.clamp(output,min=1e-10,max=1-1e-10)  
        loss =  self.weights['pos'] * (target * torch.log(output)) + self.weights['neg'] * ((1 - target) * torch.log(1 - output))
        return torch.neg(torch.mean(loss))


class DblAttentionModule(nn.Module):

    def __init__(self, config):
        super(DblAttentionModule, self).__init__()
        self.pixel_shuffel_upsample = nn.PixelShuffle(2)
        self.pixel_shuffel_upsample2 = nn.PixelShuffle(2)
        self.ReLU = nn.ReLU(inplace=True)
        self.attention_texture = nn.Sequential(nn.Conv2d(1024, config['attention_map_num'], kernel_size=1, stride=1, bias=False),
                                             nn.BatchNorm2d(config['attention_map_num'], eps=0.001),
                                             nn.ReLU(inplace=True))
        self.attention_target = nn.Sequential(nn.Conv2d(2048, config['attention_map_num'], kernel_size=1, stride=1, bias=False),
                                             nn.BatchNorm2d(config['attention_map_num'], eps=0.001),
                                             nn.ReLU(inplace=True))
        self.avgpool = nn.AvgPool2d(kernel_size=2,stride=2)
    def forward(self, x2, x1):
        # print(x.size())
        target_map = self.attention_target(x1)  # 32 channels, size
        # up2 = self.pixel_shuffel_upsample(x1) # 512 chs, size*2
        texture_map = self.attention_texture(x2)
        # attention_output = texture_map + F.interpolate(target_map, scale_factor=2, mode='bilinear')
        attention_output = target_map + self.avgpool(texture_map)
        return attention_output

EPSILON = 1e-12
class BAP(nn.Module):
    def __init__(self, pool='GAP'):
        super(BAP, self).__init__()
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = None
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, features, attentions):
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()

        # match size
        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))

        # feature_matrix: (B, M, C) -> (B, M * C)
        if self.pool is None:
            feature_matrix = (torch.einsum('imjk,injk->imn', (attentions, features)) / float(H * W)).view(B, -1)
        else:
            feature_matrix = []
            for i in range(M):
                AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, -1)
                feature_matrix.append(AiF)
            feature_matrix = torch.cat(feature_matrix, dim=1)


        # sign-sqrt
        feature_matrix = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + EPSILON)

        # l2 normalization along dimension M and C
        feature_matrix = F.normalize(feature_matrix, dim=-1)
        return feature_matrix


class res50Encoder(nn.Module):
    def __init__(self, config):
        super(res50Encoder, self).__init__()

        # load backbone and optimize its architecture
        resnet = torchvision.models.resnet50(pretrained=True)
        self.fc = nn.Linear(2048*config['attention_map_num'], config['class_num'], bias=False)
        # self.fc_bone = nn.Linear(2048, config.class_num, bias=True)

        # features
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet_encoder = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                                         resnet.layer1, resnet.layer2)
        self.resnet_encoder3 = resnet.layer3
        self.resnet_encoder4 = resnet.layer4

        self.bap = BAP(pool='GAP')

        self.attention_module = DblAttentionModule(config)
        # self.attention_module = SimpleAttentionModule(config)
        # self.attention_module = TriAttentionModule(config)

        self.M = config['attention_map_num']
        self.GAP = nn.AdaptiveAvgPool2d((1,1))
        self.GMP = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x, training=True):
        batch_size = x.size(0)
        # 2&1atten
        features_1 = self.resnet_encoder3(self.resnet_encoder(x))
        features_2 = self.resnet_encoder4(features_1)
        attention_maps = self.attention_module(features_1, features_2) #2atten
        # attention_maps = self.attention_module(features_2) #1atten
        feature_matrix = self.bap(features_2, attention_maps)

        # 3atten
        # features_3 = self.resnet_encoder(x)
        # features_2 = self.resnet_encoder3(features_3)
        # features_1 = self.resnet_encoder4(features_2)
        # attention_maps = self.attention_module(features_3,features_2,features_1)
        # feature_matrix = self.bap(features_1, attention_maps)

        logits = self.fc(feature_matrix*100)
        # GAP/GMP experiments
        # logits_bone = self.fc_bone(torch.squeeze(self.GMP(features_2)))
        # attention map 4 augment
        if training:
            # Randomly choose one of attention maps Ak
            attention_map = []
            for i in range(batch_size):
                attention_weights = torch.sqrt(attention_maps[i].sum(dim=(1, 2)).detach() + EPSILON)
                attention_weights = F.normalize(attention_weights, p=1, dim=0)
                # print(attention_weights)
                k_index = np.random.choice(self.M, 3, p=attention_weights.cpu().numpy())
                attention_map.append(attention_maps[i, k_index, ...])
            attention_map = torch.stack(attention_map)  # (B, 3, H, W) -3 types of augs
        else:
            # Object Localization Am = mean(Ak)
            attention_map = torch.mean(attention_maps, dim=1, keepdim=True)  # (B, 1, H, W)


        return logits, features_1, attention_map


class MAG_SD:
    def __init__(self, config, lr=1e-3):
        self.config = config
        self.model_name = 'mag_sd'
        self.supervised = True
        self.model_type = 'pytorch'

        # path
        # dataset configuration
        self.class_num = config['class_num']


        # train configuration
        self.attention_map_num = config['attention_map_num']
        self.lr = lr
        self.device = torch.device('cuda')

        # test configuration

        # init_model
        ## the feature alignment module
        encoder = res50Encoder(config)

        attention_module = DblAttentionModule(config)

        self.encoder = encoder.to(self.device) #t#orch.nn.DataParallel(encoder).to(self.device)
        self.attention_module = attention_module.to(self.device) #torch.nn.DataParallel(attention_module).to(self.device)

        ## add all models to a list for esay using
        self.model_list = []
        self.model_list.append(self.encoder)
        self.model_list.append(self.attention_module)

        # init_criterions
        self.MSELoss = torch.nn.MSELoss()
        self.L1Loss = torch.nn.L1Loss(reduction='mean')
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.SoftMax = nn.Softmax(dim=1)

        # init_optimizer
        params = [{'params': self.encoder.parameters(), 'lr': self.lr},
                  {'params': self.attention_module.parameters(), 'lr': self.lr}]
        self.optimizer = optim.SGD(params=params, weight_decay=5e-4, momentum=0.9, nesterov=True)   

    def gen_refine_loss(self, pred0,pred1,pred2,pred3, targets):
        targets = self.onehot_2_label(targets)
        pred0_sm = self.SoftMax(pred0).detach()
        pred1_sm = self.SoftMax(pred1)
        pred2_sm = self.SoftMax(pred2)
        pred3_sm = self.SoftMax(pred3)

        for i, vector_pred0 in enumerate(pred0_sm):
            if torch.argmax(vector_pred0)!=targets[i] or torch.max(vector_pred0)<0.7:
                vector_alter = torch.ones_like(vector_pred0) * 0.01
                vector_alter[targets[i]] = 1 - len(vector_alter) * 0.01
                pred0_sm[i] = vector_alter
        loss0 = self.CrossEntropyLoss(pred0,targets)
        variance01 = self.L1Loss(pred0_sm, pred1_sm)
        variance02 = self.L1Loss(pred0_sm, pred2_sm)
        variance03 = self.L1Loss(pred0_sm, pred3_sm)
        loss_tensor = (variance01+variance02+variance03)+loss0
        # loss_tensor = (variance01+variance02) + loss0
        # print(loss0, (variance01+variance02+variance03))
        loss_tensor.to(self.device)
        return loss_tensor

    def compute_classification_loss(self, logits, one_hot_pids):
        # one-hot2label
        label_pids = self.onehot_2_label(one_hot_pids)
        #loss and top-3 acc
        loss_i = self.CrossEntropyLoss(logits, label_pids)
        acc = self.accuracy(logits, label_pids, topk=(1,))
        return acc, loss_i

    def onehot_2_label(self, one_hot):
        return torch.argmax(one_hot, -1)

    def accuracy(self, logits, target, topk=(1, 3) , cuda = True):
        maxk = max(topk)
        batch_size = target.size(0)
  
        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            if cuda == False:
              correct_k = correct_k.detach().cpu().numpy()
              
            res.append(correct_k / batch_size)
        return res

    ## set model as train mode
    def set_train(self):
        for ii, _ in enumerate(self.model_list):
            self.model_list[ii] = self.model_list[ii].train()

    ## set model as eval mode
    def set_eval(self):
        for ii, _ in enumerate(self.model_list):
            self.model_list[ii] = self.model_list[ii].eval()

    def save_model(self, k):
        # save model
        name_list = [f'mag_sd_encoder_{k}.pth', f'mag_sd_attention_{k}.pth']
        for ii, _ in enumerate(self.model_list):
            torch.save(self.model_list[ii].state_dict(), name_list[ii])

    ## resume model from resume_epoch
    def resume_model_from_path(self, path, resume_epoch):
        name_list = [f'mag_sd_encoder_{resume_epoch}.pth', f'mag_sd_attention_{resume_epoch}.pth']
        for ii, _ in enumerate(self.model_list):
            self.model_list[ii].load_state_dict(
                torch.load(os.path.join(path, name_list[ii])))
        #print('Time: {}, successfully resume model from {}'.format(time_now(), resume_epoch))

    def build_model(self):
        model = MAG_SD(self.config)
        model = {'model':model, 'optimizer':self.optimizer, 'lr':self.lr,
        'model_name':self.model_name, 'model_type':self.model_type, 'supervised':self.supervised}
        return model        

config = {'mode':'train', 'save_model_path': '/MULTIX/DATA/nccid/','class_num':2, 'attention_map_num':32,'image_size':480,'joint_training_steps':150}

if __name__ == "__main__":
    model = MAG_SD(config).build_model()
    print(torchsummary(model['model'],(3,480,480)))
