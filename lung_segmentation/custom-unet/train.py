from  torch.utils.data import Dataset
from torchvision.utils import make_grid
import cv2

from models import VGGNestedUNet
from seg_tools import draw_bbox
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2

import torch
from sklearn.model_selection import KFold
import torch.optim as optim
from tqdm import tqdm

from seg_tools import bbox

############################# Define Loss & Metrics ###########################

def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def dice_coef(output, target):
    smooth = 1e-5
    # output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    output = torch.sigmoid(output).view(-1)
    target = target.view(-1)
    # target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

def dice_coef_loss(output, target):
	return 1. - dice_coef(output, target)


############################# Dataloaders ###########################
class CustomDataloader(Dataset):

  def __init__(self, df, dilate=False, test_data='cohen', bbox_training=True ,transforms=None):
    self.transforms = transforms
    self.df = df
    self.dilate = dilate
    self.test_data = test_data
    self.bbox_training = bbox_training

  def __getitem__(self, index):

    if self.test_data == 'segmentation':
      cxr_img = self.df.image[index]
      cxr_img = cxr_img.split('/')
      cxr_img = "/MULTIX/DATA/HOME/" + '/'.join(cxr_img[4:])

      if self.dilate==True:
        mask_img = self.df.dilate[index]
      else:
        mask_img = self.df['mask'][index]
      mask_img = mask_img.split('/')
      mask_img = "/MULTIX/DATA/HOME/" + '/'.join(mask_img[4:])  

    image = cv2.imread(cxr_img,1)
    image = cv2.resize(image, (480,480))

    masks = cv2.imread(mask_img,0)
    masks = cv2.resize(masks, (480,480))

    if self.bbox_training == True:
      bounding_box = bbox(masks)
      bounding_mask = draw_bbox(bounding_box)
      masks = [bounding_mask]

    else:
      masks = [masks]

    if self.transforms is not None:
        augmented = self.transforms(image=image, masks=masks)
        image = augmented['image']
        masks = augmented['masks']

    masks = torch.tensor(masks)
    return torch.as_tensor(image, dtype=torch.float32)/255.0, torch.as_tensor(masks, dtype=torch.int32)
    
  def __len__(self):
    return len(self.df)


############################# Define Loss & Metrics ###########################

def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def dice_coef(output, target):
    smooth = 1e-5
    # output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    output = torch.sigmoid(output).view(-1)
    target = target.view(-1)
    # target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

def dice_coef_loss(output, target):
	return 1. - dice_coef(output, target)


def create_dataloader(bs, dataframe, custom_dataloader, test_data, random_seed, num_workers, train=False, shuffle=False, pin_memory=False):
	means = [0.485, 0.456, 0.406]
	stdevs = [0.229, 0.224, 0.225]
	
	if train == False:
		transform = A.Compose([
        # A.Resize(480,480,p=1),
        # A.Normalize(),
        # ToTensorV2(transpose_mask=True)
        ])

	else:
		transform = A.Compose([
			# A.Resize(480,480,p=1),
    A.VerticalFlip(p=0.5),              
    A.RandomRotate90(p=0.5),
    A.OneOf([
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
        A.GridDistortion(p=0.5),
        A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)                  
        ], p=0.5),
    A.CLAHE(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),    
    A.RandomGamma(p=0.5),
    # ToTensorV2(transpose_mask=True)
    ])

	data = custom_dataloader(
		dataframe,
    test_data=test_data,
		transforms=transform)

	dataloader = torch.utils.data.DataLoader(data, bs,
	                                          shuffle=True,
                                            num_workers=num_workers,
                                            pin_memory=pin_memory
	                                          )
	return dataloader

def save_best_model(model, filepath): 
  torch.save(model.state_dict(), filepath)

def validate(model, loss_fn, val_loader):
  model.eval()
  val_loss = []
  val_iou = []

  with torch.no_grad():
	  for inputs, masks in val_loader:
	    masks = masks.to(device)
	    inputs = inputs.to(device)
	    inputs = inputs.permute(0,3,1,2)

	    if model.deep_supervision_status==True:
	      outputs = model(inputs)
	      loss = 0
	      for output in outputs:
	        loss += loss_fn(output, masks)
	        loss /= len(outputs)
	        iou = iou_score(outputs[-1], masks)
	    else:
	      outputs = model(inputs)
	      loss = loss_fn(outputs, masks)
	      iou = iou_score(outputs, masks)

	    val_loss.append(loss.item())
	    val_iou.append(iou.item())
		
	  val_avg_loss = np.mean(val_loss)
	  val_avg_iou = np.mean(val_iou)
  
  model.train()

  return val_avg_loss, val_avg_iou

def train(model, train_loader, val_loader, optimizer, loss_fn, k, scheduler, patience=20, num_epochs=200):
	"""Code adapted from: https://github.com/4uiiurz1/pytorch-nested-unet"""
	best_loss = 1e10

	for epoch in range(num_epochs):

		epoch_loss = []
		epoch_iou = []

		progress_bar = tqdm(total=len(train_loader))

		for inputs, masks in train_loader:
			progress_bar.update(1)
			masks = masks.to(device)

			inputs = inputs.to(device)
			inputs = inputs.permute(0,3,1,2)

			if model.deep_supervision_status==True:
        #print('ds - status: ', model.deep_supervision_status)
				outputs = model(inputs)
				loss = 0
				for output in outputs:
					loss += loss_fn(output, masks)

				loss /= len(outputs)
				iou = iou_score(outputs[-1], masks)

			else:
				outputs = model(inputs)
				loss = loss_fn(outputs, masks)
				iou = iou_score(outputs, masks)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		epoch_iou.append(iou.item())
		epoch_loss.append(loss.item())

		val_loss, val_iou = validate(model, loss_fn, val_loader)
		scheduler.step(val_loss)

		if val_loss < best_loss:
			best_loss = val_loss
			no_improvements = 0
			save_best_model(model, f'/MULTIX/DATA/HOME/vgg_nested_unet_bbox{k}.pth')
			print(f"No improvements for {no_improvements} epochs")

		else:
			no_improvements += 1
			print(f"No improvements for {no_improvements} epochs")
			if patience == no_improvements:
				print("Early stopped !")
				break
		
		print(f"Epoch: {epoch} - Train loss: {np.mean(epoch_loss)}, Train iou: {np.mean(epoch_iou)} - Val loss: {val_loss}, Val iou: {val_iou}")

	return model


if __name__ == "__main__":

    full_data = pd.read_csv("/MULTIX/DATA/HOME/LungSegmentation_JSRT/lung_segmentation_data.csv")
    train_df = full_data[full_data['split']=='train']
    train_df=train_df.reset_index(drop=True)
    
    seed = 0
    np.random.seed(seed)
    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
    
    fold_no = 1
    for train_idx, val_idx in kfold.split(train_df):
      train_data = train_df.iloc[train_idx]
      train_data = train_data.reset_index(drop=True)
      
      val_data = train_df.iloc[val_idx]
      val_data = val_data.reset_index(drop=True)
      
      train_loader = create_dataloader(bs=4, custom_dataloader=CustomDataloader, dataframe=train_data, test_data='segmentation', train=True, random_seed=0, num_workers=2)
      
      device = torch.device('cuda')
      model = VGGNestedUNet(num_classes=1, deep_supervision_status=True)
      model = model.to(device)
      
      batch_x, batch_y = next(iter(train_loader))
      grid_img = make_grid(batch_x.permute(0,3,1,2), nrow=4)
      plt.imshow(grid_img.permute(1, 2, 0))
      plt.savefig('/MULTIX/DATA/HOME/lung_segmentation/batch_cxr.png')
      
      print(np.max(np.array(batch_y)))
      batch_y = batch_y * 255
      grid_img = make_grid(batch_y, nrow=4)
      plt.imshow(grid_img.permute(1, 2, 0))
      plt.savefig('/MULTIX/DATA/HOME/lung_segmentation/batch_masks.png')
      
      optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
      scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft,  mode='min', factor=0.9, patience=5, threshold=1e-10, 
                                                           threshold_mode='rel', cooldown=0, min_lr=1e-10, eps=1e-08, verbose=True)
      val_loader = create_dataloader(bs=4, custom_dataloader=CustomDataloader, dataframe=val_data, test_data='segmentation', random_seed=0, num_workers=2)
      
      model = train(model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer_ft, loss_fn=dice_coef_loss, k=fold_no, patience=20, num_epochs=200, scheduler=scheduler)
      
      fold_no = fold_no + 1


