import argparse
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import os
import copy

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
        self.fc = nn.Linear(2048*config['attention_map_num'] config['class_num'], bias=False)

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

class mag_sd:
    def __init__(self, config, loader):
        self.config = config

        # paths
        self.loader = loader

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

        # init_criterions
        self.MSELoss = torch.nn.MSELoss()
        self.L1Loss = torch.nn.L1Loss(reduction='mean')
        self.CrossEntropyLoss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.SoftMax = nn.Softmax(dim=1)

        # init_optimizer
        params = [{'params': self.encoder.parameters()}, #'lr': self.base_learning_rate
                  {'params': self.attention_module.parameters()}] # 'lr': self.base_learning_rate

        self.optimizer = optim.SGD(params=params, weight_decay=5e-4, momentum=0.9, nesterov=True)   

    def gen_refine_loss(self, pred0, pred1, pred2, pred3, targets):
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

    def save_model(self, save_epoch):
        # save model
        for ii, _ in enumerate(self.model_list):
            torch.save(self.model_list[ii].state_dict(),
                       os.path.join(self.save_model_path, 'mag_sd-{}_{}.pkl'.format(ii, save_epoch)))

    ## set model as train mode
    def set_train(self):
        for ii, _ in enumerate(self.model_list):
            self.model_list[ii] = self.model_list[ii].train()

    ## set model as eval mode
    def set_eval(self):
        for ii, _ in enumerate(self.model_list):
            self.model_list[ii] = self.model_list[ii].eval()


config = {'mode':'train', 'save_model_path': '/MULTIX/DATA/HOME/','class_num':2, 'attention_map_num':32,'image_size':480,'joint_training_steps':150}

