from train import iou_score, dice_coef, dice_coef_loss, create_dataloader

from  torch.utils.data import Dataset
from torchvision.utils import make_grid
import cv2

from models import ResNetUNet, VGGUNet, VGGNestedUNet
from seg_tools import reverse_transform, masks_to_colorimg, plot_side_by_side, get_box_from_mask, visualize_bbox, plot_img_array, bbox
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2

from functools import reduce
import torch
from sklearn.model_selection import KFold
import torch.optim as optim
from tqdm import tqdm

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def tester(model, test_loader, model_num):

		progress_bar = tqdm(total=len(test_loader))

		i = 1

		epoch_loss = []
		epoch_iou = []

		for inputs, masks in test_loader:
			progress_bar.update(1)
			masks = masks.to(device)

			inputs = inputs.to(device)
			inputs = inputs.permute(0,3,1,2)

			pred = model(inputs)

			loss = dice_coef_loss(pred, masks)
			iou = iou_score(pred, masks)

			pred = pred.data.cpu().numpy()
			input_images_rgb = [reverse_transform(x) for x in inputs.cpu()]
			pred_rgb = [masks_to_colorimg(x) for x in pred]
			pred_box = [bbox(x) for x in pred]
			print(pred_box)
			pred_box = [visualize_bbox(b,x) for b, x in zip(input_images_rgb, pred_box)]

			plot_side_by_side([input_images_rgb, pred_rgb, pred_box], i, model_num)

			epoch_iou.append(iou.item())
			epoch_loss.append(loss.item())

			i = i + 1

		print(f"vgg_nested_unet{model_num} - loss: {np.mean(epoch_loss)} iou: {np.mean(epoch_iou)}")

		

if __name__ == "__main__":

		full_data = pd.read_csv("/MULTIX/DATA/HOME/LungSegmentation_JSRT/lung_segmentation_data.csv")
		test_df = full_data[full_data['split']=='test']
		test_df = test_df.reset_index(drop=True)

		test_loader = create_dataloader(bs=4, dataframe=test_df, train=False, random_seed=0, num_workers=2)

		device = torch.device('cuda')
		model = VGGNestedUNet(num_classes=1)
		model = model.to(device)

		for model_num in range(1,4):
			model.load_state_dict(torch.load(f"/MULTIX/DATA/HOME/vgg_nested_unet_{model_num}.pth"))

			tester(model, test_loader, model_num=model_num)


