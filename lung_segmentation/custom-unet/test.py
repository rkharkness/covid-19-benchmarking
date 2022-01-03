from train import CustomDataloader, iou_score, dice_coef, dice_coef_loss, create_dataloader

from  torch.utils.data import Dataset
from torchvision.utils import make_grid
import cv2

from models import ResNetUNet, VGGUNet, VGGNestedUNet
from seg_tools import mask_from_bbox, mask_from_img, reverse_transform, masks_to_colorimg, plot_side_by_side, segment_mask, visualize_bbox, bbox
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2

from functools import reduce
import torch

from sklearn.model_selection import KFold
from sklearn.metrics import jaccard_score

import torch.optim as optim
from tqdm import tqdm

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

class LTHTDataloader(Dataset):

	def __init__(self, df, test_data, transforms=None):
		self.transforms = transforms
		self.df = df
		self.test_data = test_data

	def __getitem__(self, index):
		cxr_img = self.df.structured_path[index]
		image = cv2.imread(cxr_img,1)
		image = cv2.resize(image, (480,480))

		img_class_str = self.df.finding[index]
		
		if self.transforms is not None:
			augmented = self.transforms(image=image)
			image = augmented['image']

		return torch.as_tensor(image, dtype=torch.float32)/255.0, img_class_str

	def __len__(self):
		return len(self.df)

def tester(model, test_loader, test_data, model_num):

		progress_bar = tqdm(total=len(test_loader))

		i = 1

		epoch_loss = []
		epoch_iou = []
		epoch_jaccard = []

		for data in test_loader:
			progress_bar.update(1)

			if test_data != 'ltht':
				inputs, masks = data
				masks = masks.to(device)

				inputs = inputs.to(device)
				inputs = inputs.permute(0,3,1,2)

				pred = model(inputs)
				loss = dice_coef_loss(pred, masks)
				iou = iou_score(pred, masks)
				jaccard = jaccard_score(pred,masks)

				epoch_iou.append(iou.item())
				epoch_loss.append(loss.item())
				epoch_jaccard.append(jaccard.item())
			
			else:
				inputs, img_class = data
				inputs = inputs.to(device)
				inputs = inputs.permute(0,3,1,2)
				pred = model(inputs)
				pass

			pred = pred.data.cpu().numpy()
			input_images_rgb = [reverse_transform(x) for x in inputs.cpu()]
				
			pred_rgb = [masks_to_colorimg(x) for x in pred]
			pred_mask = [mask_from_img(x) for x in pred]
			pred_box = [bbox(x) for x in pred]
				
			img_box = [visualize_bbox(b,x) for b, x in zip(input_images_rgb, pred_box)]
			bbox_mask = [mask_from_bbox(b,x) for b, x in zip(input_images_rgb, pred_box)]
			seg_mask = [segment_mask(b,x) for b, x in zip(input_images_rgb, pred_mask)]
			
			if i % 5 == 0:
				plot_side_by_side([input_images_rgb, pred_rgb, seg_mask, img_box, bbox_mask], i, model_num, img_class=img_class)

			i = i + 1

		print(f"vgg_nested_unet{model_num} - loss: {np.mean(epoch_loss)} iou: {np.mean(epoch_iou)} jaccard: {np.mean(epoch_jaccard)}")

		

if __name__ == "__main__":

		segmentation_data = pd.read_csv("/MULTIX/DATA/HOME/LungSegmentation_JSRT/lung_segmentation_data.csv")
		seg_test_df = segmentation_data[segmentation_data['split']=='test']
		seg_test_df = seg_test_df.reset_index(drop=True)

		#cohen_data = pd.read_csv('/MULTIX/DATA/HOME/covid-19-chest-xray-segmentations-dataset-master/structure.csv')

		ltht_data = pd.read_csv('/MULTIX/DATA/INPUT/seg_binary_data.csv')
		test_df = ltht_data
		test_loader = create_dataloader(bs=4, dataframe=test_df, custom_dataloader=LTHTDataloader, test_data='ltht', train=False, random_seed=0, num_workers=2)

		device = torch.device('cuda')
		model = VGGNestedUNet(num_classes=1)
		model = model.to(device)

		for model_num in range(1,6):
			model.load_state_dict(torch.load(f"/MULTIX/DATA/HOME/vgg_nested_unet_bbox{model_num}.pth"))
			tester(model, test_loader, test_data = 'ltht', model_num=model_num)
			


