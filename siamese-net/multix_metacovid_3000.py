# import required libraries
import pandas as pd
import os
import cv2
import time
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import models, transforms
from sklearn.metrics import precision_score, recall_score, f1_score, auc, precision_recall_curve, confusion_matrix
import torch.nn.functional as F

from torch.autograd import Variable


import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import albumentations as A #need version 0.4.6
from albumentations.pytorch import ToTensorV2


# read in csv of image info
total_data = pd.read_csv('/MULTIX/DATA/HOME/custom_data_ablation_3000.csv')
print(total_data.filename)

# divide full df into train, val and test dfs - can do with org img paths or multix
# using org multix paths here
train_df = total_data[total_data['split']=='train']
train_df = train_df.reset_index(drop=True)

val_df = total_data[total_data['split']=='val']
val_df = val_df.reset_index(drop=True)

test_df = total_data[total_data['split']=='test']
test_df = test_df.reset_index(drop=True)


import random

class ImbalancedSiameseNetworkDataset(Dataset):
    
    def __init__(self,trainImageFolderDataset,train, testImageFolderDataset=None, transform=None):
        # self.train = train
        self.trainImageFolderDataset = trainImageFolderDataset    
        self.transform = transform
        self.comparison_ds = trainImageFolderDataset

        if train == True:
          self.comparison_ds = trainImageFolderDataset
        else:
          self.comparison_ds = testImageFolderDataset
        
    def __getitem__(self,index):

        should_get_pneum = random.randint(0,1)

        if should_get_pneum:
          while True:
            img0_idx = np.random.choice(self.trainImageFolderDataset.index)
            img0_tuple = (self.trainImageFolderDataset.filename[img0_idx], self.trainImageFolderDataset.pneumonia_binary[img0_idx])
            if img0_tuple[1] == 1:
              break
        else:
          while True:
            img0_idx = np.random.choice(self.trainImageFolderDataset.index)
            img0_tuple = (self.trainImageFolderDataset.filename[img0_idx], self.trainImageFolderDataset.pneumonia_binary[img0_idx])
            if img0_tuple[1] == 0:
              break          

        #we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1)
        
        if should_get_same_class:
            while True:
                #keep looping till the same class image is found
                img1_idx = np.random.choice(self.comparison_ds.index) 
                img1_tuple = (self.comparison_ds.filename[img1_idx], self.comparison_ds.pneumonia_binary[img1_idx])
                
                if img0_tuple[1]==img1_tuple[1]:
                    break
        else:
            while True:
              img1_idx = np.random.choice(self.comparison_ds.index) 
              img1_tuple = (self.comparison_ds.filename[img1_idx], self.comparison_ds.pneumonia_binary[img1_idx])
              
              if img0_tuple[1]!=img1_tuple[1]:
                  break

        img0 = cv2.imread(img0_tuple[0])
        img1 = cv2.imread(img1_tuple[0])

        
        if self.transform is not None:
            img0 = self.transform(image=img0)["image"]
            img1 = self.transform(image=img1)["image"]
            
        return img0/255.0, img1/255.0 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32)), img0_tuple[1], img1_tuple[1]


    def __len__(self):
        return len(self.trainImageFolderDataset)

# define transforms
train_transform = A.Compose(
    [A.HorizontalFlip(p=0.2),
     # A.Rotate(p=0.2, limit=20),
     # A.ShiftScaleRotate(p=0.2, rotate_limit=20),
     ToTensorV2()]
)

test_transform = A.Compose(
    [
      ToTensorV2()
    ]
)

seed = 0
train_data = ImbalancedSiameseNetworkDataset(train_df, train=True, transform=train_transform)
train_loader = DataLoader(train_data, batch_size=1, num_workers=0,
                          pin_memory=True, shuffle=True, 
                          worker_init_fn=np.random.seed(seed))
all_labels = {'label':[],'img0_gt':[], 'img1_gt':[]}
for data in train_loader:
  img0, img1, label, gt0, gt1 = data
  all_labels['label'].append(np.array(label)[0][0])
  all_labels['img0_gt'].append(np.array(gt0)[0])
  all_labels['img1_gt'].append(np.array(gt1)[0])


label_df = pd.DataFrame(all_labels)

label_df.columns = ["label","img0_gt", "img1_gt"]
print(label_df)
print(label_df.groupby('label').count())
print(label_df.groupby('img0_gt').count())
print(label_df.groupby('img1_gt').count())


seed = 0
train_data = ImbalancedSiameseNetworkDataset(train_df, train=True, transform=train_transform)
train_loader = DataLoader(train_data, batch_size=4, num_workers=0,
                          pin_memory=True, shuffle=True, 
                          worker_init_fn=np.random.seed(seed))

val_data = ImbalancedSiameseNetworkDataset(train_df, train=False, testImageFolderDataset=val_df, transform=test_transform)
valid_loader = DataLoader(val_data, batch_size=4, num_workers=0,
                          pin_memory=True, shuffle=True, 
                          worker_init_fn=np.random.seed(seed))

test_data = ImbalancedSiameseNetworkDataset(train_df, train=False, testImageFolderDataset=test_df, transform=test_transform)
test_loader = DataLoader(test_data, batch_size=1, num_workers=0,
                          pin_memory=True, shuffle=True, 
                          worker_init_fn=np.random.seed(seed))

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            models.vgg16(pretrained=True).features, # pretrained on imagenet - finetuned?
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
        )

        self.fc1 = nn.Sequential(nn.Linear(51200, 5120),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(5120, 5120))
        
        self.fc2 = nn.Linear(5120,1)

        
    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return torch.sigmoid(output)

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        dist = torch.abs(output1 - output2) #try fc layer -5120 to 1 instead
        dist = self.fc2(dist)
        return torch.sigmoid(dist)

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=1.25):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, euclidean_distance, label):
        loss_contrastive = torch.mean((1-label) * (0.5 * torch.pow(euclidean_distance, 2)) +
                                      (label) * (0.5*torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)))

        return loss_contrastive

def save_best_model(model, epoch):
  print("saving model ...")
  torch.save(model.state_dict(), "/home/ubuntu/multix_metacovid_3000_{}.pth".format(epoch))


net = SiameseNetwork().cuda()

criterion = ContrastiveLoss()

optimizer = torch.optim.Adam(net.parameters(),lr = 1e-5, weight_decay=1e-3) # l2 reg
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,  mode='min', factor=0.9, patience=5, threshold=0.00001, 
                                                       threshold_mode='rel', cooldown=0, min_lr=1e-10, eps=1e-08, verbose=True)



def test_accuracy(data_loader, train):
    total = 0
    correct = 0
    for data in data_loader:
        img0, img1, label, gt0, gt1 = data
        img0, img1, label = Variable(img0).cuda(), Variable(img1).cuda(), Variable(label).cuda()   

        dist = net(img0,img1)

        # dist = torch.sigmoid(F.pairwise_distance(output1, output2))
        dist = dist.cpu()

        for j in range(dist.size()[0]):
            if ((dist.data.numpy()[j]>0.5)):
                if label.cpu().data.numpy()[j]==1:
                    correct +=1
                    total+=1
                else:
                    total+=1
            else:
                if label.cpu().data.numpy()[j]==0:
                    correct+=1
                    total+=1
                else:
                    total+=1
    
    if train==True:
      print('Accuracy of the network on the train images: %d %%' % (
          100 * correct / total))
    else:
      print('Accuracy of the network on the test images: %d %%' % (
          100 * correct / total))

def validate(epoch, valid_loader=valid_loader):
      val_losses = []
      with torch.no_grad():
	      for i, data in enumerate(valid_loader,0):
	          img0, img1 ,label, gt0, gt1 = data
	          img0, img1 , label = Variable(img0).cuda(), Variable(img1).cuda() , Variable(label).cuda()
	          dist = net(img0,img1)
	          # optimizer.zero_grad()
	          loss = criterion(dist,label)
	          val_losses.append(loss.item())
      
      print("Epoch: {} - Validation loss: {}".format(epoch,np.mean(val_losses)))
      return np.mean(val_losses)


def train(num_epoch, scheduler=scheduler, criterion=criterion, optimizer=optimizer, patience=25):
    best_val_loss = 1e10

    counter = []
    loss_history = [] 
    val_loss_history = []

    iteration_number= 0
    for epoch in range(0,num_epoch):
        losses = []
        for i, data in enumerate(train_loader,0):
            img0, img1 ,label, gt0, gt1 = data
            img0, img1 , label = Variable(img0).cuda(), Variable(img1).cuda() , Variable(label).cuda()
            dist = net(img0,img1)
            optimizer.zero_grad()
            loss = criterion(dist,label)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if i % 200 == 0 :
                print("Epoch {} - Loss: {}".format(epoch, np.mean(losses)))
                iteration_number += 200
                counter.append(iteration_number)
                loss_history.append(loss.item())

        print("Epoch: {} - Average loss: {}".format(epoch,np.mean(losses)))

        val_loss = validate(epoch)
        scheduler.step(val_loss)

        val_loss_history.append(val_loss)

        if epoch % 5 == 0:
          test_accuracy(train_loader, train=True)
          test_accuracy(valid_loader, train=False)

        if val_loss < best_val_loss:
          no_improvement = 0 
          best_val_loss = val_loss
          save_best_model(net, epoch)
          print("no improvements in validation loss - {}/{}".format(no_improvement, patience))

        elif val_loss > best_val_loss:
          no_improvement = no_improvement + 1
          print("no improvements in validation loss - {}/{}".format(no_improvement, patience))

        if no_improvement == patience:
          print("early stopped")
          break
    
    return counter, loss_history

counter, loss_history = train(500)

# class EvalSiameseNetworkDataset(Dataset):
    
#     def __init__(self,trainImageFolderDataset,testImageFolderDataset=None, way=3, transform=None):
#         self.trainImageFolderDataset = trainImageFolderDataset    
#         self.testImageFolderDataset = testImageFolderDataset    
#         self.transform = transform
#         self.way = way

        
#     def __getitem__(self,index):
#         img0_idx = np.random.choice(self.testImageFolderDataset.index)
#         img0_tuple = (self.testImageFolderDataset.full_path[img0_idx], self.testImageFolderDataset.covid_binary[img0_idx])
#         img0 = cv2.imread(img0_tuple[0])
#         img0 = cv2.resize(img0, (512,512))
                
#         if self.transform is not None:
#             img0 = self.transform(image=img0)["image"]
#         img0 = img0/255.0
        
#         all_img0 = []
#         all_img0_tuple = []
#         for i in range(0,self.way):
#           all_img0.append(img0)
#           all_img0_tuple.append(img0_tuple[1])
        
#         all_img0 = torch.stack(all_img0)

#         img1_idx = np.random.choice(self.trainImageFolderDataset.index, self.way)

#         all_img1 = []
#         all_img1_tuple = []
#         for i in img1_idx:
#           img1_tuple = (self.trainImageFolderDataset.full_path[i], self.trainImageFolderDataset.covid_binary[i])

#           all_img1_tuple.append(img1_tuple[1])

#           img1 = cv2.imread(img1_tuple[0])
#           img1 = cv2.resize(img1, (512,512))

#           if self.transform is not None:
#             img1 = self.transform(image=img1)["image"]
#           img1 = img1/255.0

#           all_img1.append(img1)
        
#         all_img1 = torch.stack(all_img1)

#         target_list = []
#         for i in range(0,self.way):
#           target = torch.from_numpy(np.array([int(all_img1_tuple[i]!=all_img0_tuple[i])],dtype=np.float32))
#           target_list.append(target)
#         target_list = torch.cat(target_list)
            
#         return all_img0, all_img1, target_list, all_img0_tuple, all_img1_tuple


#     def __len__(self):
#         return len(self.testImageFolderDataset)

# test_transform = A.Compose(
#     [
#       ToTensorV2()
#     ]
# )

# seed = 0

# test_data = EvalSiameseNetworkDataset(train_df, testImageFolderDataset=test_df[:4000], transform=test_transform)
# test_loader = DataLoader(test_data, batch_size=1, num_workers=0,
#                           pin_memory=True, shuffle=True, 
#                           worker_init_fn=np.random.seed(seed))

# all_img0, all_img1, target_list, img0_tuple, img1_tuple= next(iter(test_loader))

# all_img0 = all_img0.permute(1,0,2,3,4)
# print(all_img0.shape)


# net = SiameseNetwork().cuda()
# net.load_state_dict(torch.load('/MULTIX/DATA/HOME/multix_covid2_balanced_metacovid_81.pth'))

# from collections import Counter

# def most_common(lst):
#     data = Counter(lst)
#     return data.most_common(1)[0][0]

# def imshow(img,gt0b,gt1b,text=None,should_save=False):
#     npimg = img.numpy()
#     plt.axis("off")
#     if text:
#         plt.text(275, 8, text, style='italic',fontweight='bold',
#             bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
#         plt.text(200,550, gt0b)
#         plt.text(700,550, gt1b)

#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.savefig('/home/ubuntu/covid_dissimilarity.png')
#     plt.show() 

# import torchvision
# def dissimilarity(test_loader, sample_num):
#     dataiter = iter(test_loader)
#     with torch.no_grad():
#       for i in range(sample_num):
#           x0,_,_,_,_ = next(dataiter)
#           x0 = x0.permute(1,0,2,3,4)
#           _,x1,label2,gt0,gt1 = next(dataiter)
#           x1 = x1.permute(1,0,2,3,4)
#           for j in range(len(x1)):
#             concatenated = torch.cat((x0[j],x1[j]),0)
#             dist = net(Variable(x0[j]).cuda(),Variable(x1[j]).cuda())
#             dist = dist.cpu().detach().numpy()[0][0]
#             gt0b = gt0[j].cpu().detach().numpy()
#             gt1b = gt1[j].cpu().detach().numpy()

#             imshow(torchvision.utils.make_grid(concatenated),gt0b,gt1b,'Dissimilarity: {:1f}'.format(dist))

# sample_num = 5
# dissimilarity(test_loader, sample_num)

# from tqdm import tqdm

# ##  === Testing the Classification Model ===
# def multi_evaluate(loader):
#   correct = 0
#   total = 0.0
#   all_gt_label = []
#   all_pr_label = []
#   sim_label = []
#   with torch.no_grad():
#       for data in tqdm(loader):
#           all_img0, all_img1 ,all_label, all_img0_gt, all_img1_gt = data
#           all_img0, all_img1 , all_label = Variable(all_img0).cuda(), Variable(all_img1).cuda() , Variable(all_label).cuda()
#           all_label = all_label.permute(1,0)
#           all_img0, all_img1 = all_img0.permute(1,0,2,3,4), all_img1.permute(1,0,2,3,4)
          
#           gt_label = []
#           pr_label = []

#           for i in range(all_img1.size(0)):
#               img0 = all_img0[i]
#               img1 = all_img1[i]
#               label = all_label[i]

#               dist = net(img0, img1)
#               pred = (dist>0.5).float()
#               # correct += (pred==label).sum().item()
              
#               if pred == 0: # predicts as same if 0
#                   pred_label = all_img0_gt[i]
              
#               elif pred == 1:
#                   poss_pred = [torch.tensor(0.),torch.tensor(1.)]
#                   poss_pred.remove(all_img0_gt[i])
#                   pred_label = poss_pred.pop()

#               gt_label.append(all_img0_gt[i])
#               pr_label.append(pred_label)

#           pr_label = most_common(pr_label)
#           correct += (pr_label==all_img0_gt[i]).float().sum().item()

#           all_pr_label.append(pr_label)
#           all_gt_label.append(all_img0_gt[i])
      
  
#   print('Accuracy of the model: {} %%'.format((100 * correct / len(loader.dataset))))

#   return all_gt_label, all_pr_label

# gt, pr = multi_evaluate(test_loader)

# from sklearn.metrics import precision_score, recall_score, f1_score, auc, precision_recall_curve, confusion_matrix, classification_report

# print(classification_report(np.array(gt), np.array(pr)))

# cm = confusion_matrix(np.array(gt), np.array(pr))

# total = sum(sum(cm))
# acc = (cm[0, 0] + cm[1, 1]) / total
# sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
# specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
# precision = cm[0, 0] / (cm[0, 0] + cm[1, 0])
# recall = cm[1, 1] / (cm[0, 1] + cm[1, 1])

# print(cm)
# print("acc: {:.4f}".format(acc))
# print("sensitivity: {:.4f}".format(sensitivity))
# print("specificity: {:.4f}".format(specificity))
# print("precision: {:.4f}".format(precision))
# print("recall: {:.4f}".format(recall))



# test_report = classification_report(gt , pr, output_dict=True)
# pd.DataFrame(test_report).to_csv('/home/ubuntu/siam_covid_test_report_covidx.csv', index=False)
# print(test_report)

# ## === Evaluation metrics ###
# mapping = ['negative', 'positive']
# recall = recall_score(gt, pr, average='weighted')
# class_wise_recall = recall_score(gt, pr, average=None)
# print(f'Sensitivity of each class:\n{mapping[0]} = {class_wise_recall[0]:.4f} | {mapping[1]} = {class_wise_recall[1]:.4f}\n')
#       # | {mapping[2]} = {class_wise_recall[2]:.4f}\n')
# precision = precision_score(gt, pr, average='weighted')
# class_wise_precision = precision_score(gt, pr, average=None)
# print(f'PPV of each class:\n{mapping[0]} = {class_wise_precision[0]:.4f} | {mapping[1]} = {class_wise_precision[1]:.4f}\n')
#       # | {mapping[2]} = {class_wise_precision[2]:.4f}\n')

# # fpr, tpr, roc_thresh = sklearn.metrics.roc_curve(gt, pr)

# from sklearn.metrics import roc_curve
# fpr_keras, tpr_keras, thresholds_keras = roc_curve(gt, pr)
# from sklearn.metrics import auc
# auc_keras = auc(fpr_keras, tpr_keras)


# plt.figure(1)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr_keras, tpr_keras, label='siamese (area = {:.3f})'.format(auc_keras))
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC curve')
# plt.legend(loc='best')
# plt.savefig("/home/ubuntu/siam_covid_covidx_test_roc.png")

# plt.show()
# # Zoom in view of the upper left corner.
# # plt.figure(2)
# # plt.xlim(0, 0.4)
# # plt.ylim(0.65, 1)
# # plt.plot([0, 1], [0, 1], 'k--')
# # plt.plot(fpr_keras, tpr_keras, label='coronet (area = {:.3f})'.format(auc_keras))
# # plt.xlabel('False positive rate')
# # plt.ylabel('True positive rate')
# # plt.title('ROC curve (zoomed in at top left)')
# # plt.legend(loc='best')
# # plt.savefig("/content/gdrive/MyDrive/siam_pne_covidx_test_roc_zoom.png")
# # plt.show()

# # roc_df = pd.DataFrame()
# # roc_df['fpr'] = fpr
# # roc_df['tpr'] = tpr
# # roc_df['roc_thresh'] = roc_thresh

# # roc_df.to_csv(f'/content/gdrive/MyDrive/coronet_roc_df_{seed}.csv')

# # plt.figure(1)
# # plt.plot([0, 1], [0, 1], 'k--')
# # plt.plot(fpr_keras, tpr_keras, label='coronet (area = {:.3f})'.format(auc_keras))
# # plt.xlabel('False positive rate')
# # plt.ylabel('True positive rate')
# # plt.title('ROC curve')
# # plt.legend(loc='best')
# # plt.show()

# # roc_display = sklearn.metrics.RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
# # num_classes=2
# # lr_precision, lr_recall, lr_auc, lr_thresh = [], [], [], []
# # for mm in range(num_classes):
# #     pre, rec, thresh= precision_recall_curve([(f==mm)*1 for f in gt], [(f==mm)*1 for f in pr])
# #     lr_precision.append(pre)
# #     lr_thresh.append(thresh)
# #     lr_recall.append(rec)
# #     lr_auc.append(auc(rec, pre))

# # lr_f1 = f1_score(gt, pr, average='weighted')
# # print('## === Dataset Evaluation Metrics ===')
# # print(f'AUC: {np.mean(lr_auc):0.4f}\nAccuracy: {correct / len(test_loader.dataset):0.4f}\nPrecision: {precision:0.4f}\nRecall: {recall:0.4f} \nF1-score: {lr_f1:0.4f}')

# # class_wise_metrics_df = pd.DataFrame()
# # class_wise_metrics_df['class'] = range(num_classes)
# # class_wise_metrics_df['auc'] = lr_auc
# # class_wise_metrics_df['class_wise_precision'] = class_wise_precision
# # class_wise_metrics_df['lr_precision'] = lr_precision
# # class_wise_metrics_df['lr_recall'] = lr_recall
# # class_wise_metrics_df['lr_thresh'] = lr_thresh

# # class_wise_metrics_df.to_csv(f'/home/ubuntu/siam_pneumonia_test_class_wise_metrics_df_{seed}.csv')

# # overall_metrics_df = pd.DataFrame()
# # overall_metrics_df['accuracy'] = [correct / len(test_loader.dataset)]
# # overall_metrics_df['auc'] = np.mean(lr_auc)
# # overall_metrics_df['precision'] = precision
# # overall_metrics_df['recall'] = recall
# # overall_metrics_df['f1'] = lr_f1

# # overall_metrics_df.to_csv(f'/home/ubuntu/siam_pneumonia_test_overall_metrics_df_{seed}.csv')

# # print(f'Confusion Matrix:\n {confusion_matrix(gt, pr)}')
# # cm = confusion_matrix(gt, pr)
# # cm.to_csv(f'coronet_cm_{seed}.csv')

# def plot_confusion_matrix(cm, target_names, save_path, title='Confusion matrix', cmap=None, normalize=True):

#     import matplotlib.pyplot as plt
#     import numpy as np
#     import itertools

#     accuracy = np.trace(cm) / float(np.sum(cm))
#     misclass = 1 - accuracy

#     if cmap is None:
#         cmap = plt.get_cmap('Blues')

#     plt.figure(figsize=(8, 6))
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()

#     if target_names is not None:
#         tick_marks = np.arange(len(target_names))
#         plt.xticks(tick_marks, target_names, rotation=45)
#         plt.yticks(tick_marks, target_names)

#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


#     thresh = cm.max() / 1.5 if normalize else cm.max() / 2
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         if normalize:
#             plt.text(j, i, "{:0.4f}".format(cm[i, j]),fontsize=20,
#                      horizontalalignment="center",
#                      color="white" if cm[i, j] > thresh else "black")
#         else:
#             plt.text(j, i, "{:,}".format(cm[i, j]),fontsize=20,
#                      horizontalalignment="center",
#                      color="white" if cm[i, j] > thresh else "black")


#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.savefig(save_path)

# plot_confusion_matrix(cm, normalize=False, target_names=['Negative', 'Positive'], title="Confusion Matrix",
#                       save_path="/home/ubuntu/siam_covid_cm_plot_test_covidx.png")








