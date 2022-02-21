import argparse
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

            import matplotlib.pyplot as plt
            plt.subplot(2,1,1)
            plt.imshow(patch[0][0].squeeze().cpu().numpy(), cmap='gray')
            plt.subplot(2,1,2)
            plt.imshow(auged_image[0][0].squeeze().cpu().numpy(), cmap='gray')
            plt.show()

        multi_images = torch.cat(multi_image, dim=0)
        return multi_images

    else:
        raise ValueError('Expected mode in [\'crop\', \'drop\'], but received unsupported augmentation method %s' % mode)


class WeightedBCE():
    def  __init__(self) -> None:
        super().__init__()
        self.weights = {'neg':1.32571275, 'pos':0.80276873}
    
    def forward(self, target, output):
        output = torch.clamp(output,min=1e-10,max=1-1e-10)  
        loss =  self.weights['pos'] * (target * torch.log(output)) + self.weights['neg'] * ((1 - target) * torch.log(1 - output))
        return torch.neg(torch.mean(loss))

class MAGLoss():
    def __init__(self):
        self.bce = WeightedBCE()
        self.device = 'cuda'
    
    def accuracy(self, logits, target, topk=(1, 3)):
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k / batch_size)
        return res

    def gen_refined_loss(self, pred0, pred1, pred2, pred3, targets):
        pred0_sm = torch.sigmoid(pred0).detach()
        pred1_sm = torch.sigmoid(pred1)
        pred2_sm = torch.sigmoid(pred2)
        pred3_sm = torch.sigmoid(pred3)

        for i, vector_pred0 in enumerate(pred0_sm):
            if torch.argmax(vector_pred0)!=targets[i] or torch.max(vector_pred0)<0.7:
                vector_alter = torch.ones_like(vector_pred0) * 0.01
                vector_alter[targets[i]] = 1 - len(vector_alter) * 0.01
                pred0_sm[i] = vector_alter

        loss0 = self.bce(pred0,targets)
        variance01 = F.l1_loss(pred0_sm, pred1_sm)
        variance02 = F.l1_loss(pred0_sm, pred2_sm)
        variance03 = F.l1_loss(pred0_sm, pred3_sm)
        loss_tensor = (variance01+variance02+variance03)+loss0
        loss_tensor.to(self.device)
        
        return loss_tensor

    def compute_classification_loss(self, logits, inputs):
        # one-hot2label
      #  label_pids = self.onehot_2_label(one_hot_pids)
        #loss and top-3 acc
        loss_i = self.bce(logits, inputs)
        acc = self.accuracy(logits, inputs, topk=(1,))
        return acc, loss_i

    def forward(self, pred, batch_y):
        logit_raw, logit_mixup, logit_dim, logit_patch = pred

        acc_raw, loss_raw = self.compute_classification_loss(logit_raw, batch_y)
        acc_mixup, loss_mixup = self.compute_classification_loss(logit_mixup, batch_y)
        acc_dim, loss_dim = self.compute_classification_loss(logit_dim, batch_y)
        acc_patch, loss_patch = self.compute_classification_loss(logit_patch, batch_y)

        loss = self.gen_refine_loss(logit_raw, logit_mixup, logit_dim, logit_patch, batch_y)
        variance = loss - loss_raw

        return loss, ['acc_raw', 'loss_raw', 'acc_mixup','loss_mixup','acc_dim','loss_dim','acc_patch','loss_patch','loss_v', 'variance'], \
	       torch.Tensor([acc_raw[0], loss_raw.data, acc_mixup[0], loss_mixup.data, acc_dim[0], loss_dim.data, acc_patch[0], loss_patch.data, loss, variance])


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
        feature_matrix = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + 1e-12)

        # l2 normalization along dimension M and C
        feature_matrix = F.normalize(feature_matrix, dim=-1)
        return feature_matrix

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

class res50Encoder(nn.Module):
    def __init__(self, config):
        super(res50Encoder, self).__init__()

        # load backbone and optimize its architecture
        resnet = torchvision.models.resnet50(pretrained=True)
        self.fc = nn.Linear(2048*config['attention_map_num'], config['class_num'], bias=False)

        # features
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet_encoder = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                                         resnet.layer1, resnet.layer2)
        self.resnet_encoder3 = resnet.layer3
        self.resnet_encoder4 = resnet.layer4

        self.bap = BAP(pool='GAP')

        self.attention_module = DblAttentionModule(config)

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
                attention_weights = torch.sqrt(attention_maps[i].sum(dim=(1, 2)).detach() + 1e-12)
                attention_weights = F.normalize(attention_weights, p=1, dim=0)
                # print(attention_weights)
                k_index = np.random.choice(self.M, 3, p=attention_weights.cpu().numpy())
                attention_map.append(attention_maps[i, k_index, ...])
            attention_map = torch.stack(attention_map)  # (B, 3, H, W) -3 types of augs
        else:
            # Object Localization Am = mean(Ak)
            attention_map = torch.mean(attention_maps, dim=1, keepdim=True)  # (B, 1, H, W)


        return logits, features_1, attention_map

class MAG_SD(nn.Module):
    def __init__(self, config, batch_augment=batch_augment):
        super(MAG_SD, self).__init__()
        self.model_name = "mag_sd"
        self.model_type = "pytorch"
        self.supervised = True        
        self.loss_fn = MAGLoss()

        self.lr = 1e-4 

        self.config = config

        self.batch_augment = batch_augment

        # dataset configuration
        self.class_num = config['class_num']
        self.save_model_path = config['save_model_path']

        # train configuration
        self.attention_map_num = config['attention_map_num']
        self.device = torch.device('cuda')

        # init_model
        encoder = res50Encoder(config)
        attention_module = DblAttentionModule(config)

        self.encoder = torch.nn.DataParallel(encoder).to(self.device)
        self.attention_module = torch.nn.DataParallel(attention_module).to(self.device)

        ## add all models to a list for easy using
        self.model_list = []
        self.model_list.append(self.encoder)
        self.model_list.append(self.attention_module)

        # init_optimizer
        params = [{'params': self.encoder.parameters(), 'lr': self.lr},
                    {'params': self.attention_module.parameters(), 'lr': self.lr}]

        self.optimizer = optim.SGD(params=params, weight_decay=5e-4, momentum=0.9, nesterov=True)   


    def forward(self, x):
        logit_raw, feature_1, attention_map = self.encoder(x)

        ### batch augs
        # mixup
        mixup_images = self.batch_augment(x, attention_map[:, 0:1, :, :], mode='mixup', theta=(0.4, 0.6), padding_ratio=0.1)
        logit_mixup, _, _ = self.encoder(mixup_images)
        
        # # dropping
        drop_images = self.batch_augment(x, attention_map[:, 1:2, :, :], mode='dim', theta=(0.2, 0.5))
        logit_dim, _, _ = self.encoder(drop_images)
        #
        # ## patching
        patch_images = self.batch_augment(x, attention_map[:, 2:3, :, :], mode='patch', theta=(0.4, 0.6), padding_ratio=0.1)
        logit_patch, _, _= self.encoder(patch_images)

        return logit_raw, logit_mixup, logit_dim, logit_patch

    def build_model(self):
        model = MAG_SD(self.config)
        model = {'model':model, 'optimizer':self.optimizer, 'loss_fn':self.loss_fn, 'lr':self.lr,
        'model_name':self.model_name, 'model_type':self.model_type, 'supervised':self.supervised}
        return model

    def save_model(self):
        # save model
        name_list = ['mag_sd_encoder.pth', 'mag_sd_attention.pth']
        for ii, _ in enumerate(self.model_list):
            torch.save(self.model_list[ii].state_dict(), name_list[ii])

    ## set model as train mode
    def set_train(self):
        for ii, _ in enumerate(self.model_list):
            self.model_list[ii] = self.model_list[ii].train()

    ## set model as eval mode
    def set_eval(self):
        for ii, _ in enumerate(self.model_list):
            self.model_list[ii] = self.model_list[ii].eval()


config = {'mode':'train', 'save_model_path': '/MULTIX/DATA/nccid/','class_num':1, 'attention_map_num':32,'image_size':480,'joint_training_steps':150}

from torchinfo import summary
if __name__ == "__main__":
    mag_sd = MAG_SD(config)
    model = mag_sd.build_model()
    print(summary(model['model']))

