import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from  torch.utils.data import Dataset
from torchvision.utils import make_grid
import cv2
import torch
import albumentations as A
from models import ResNetUNet, VGGUNet, VGGNestedUNet


def create_dataloader(bs, dataframe, custom_dataloader, pin_memory=False):
    means = [0.485, 0.456, 0.406]
    stdevs = [0.229, 0.224, 0.225]
    
    transform = A.Compose([
        A.VerticalFlip(p=0.5),              
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
       # A.Affine(translate_percent=10,p=0.5),
        A.CLAHE(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),    
        A.RandomGamma(p=0.5),
    ])
    
    data = custom_dataloader(
		dataframe,
		transforms=transform)
        
    dataloader = torch.utils.data.DataLoader(data, bs,
	                                        shuffle=True,
                                            pin_memory=pin_memory
    )
    return dataloader


class LTHTDataloader(Dataset):

	def __init__(self, df, transforms=None):
		self.transforms = transforms
		self.df = df
		#self.test_data = test_data

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


ltht_data = pd.read_csv('/MULTIX/DATA/INPUT/seg_binary_data.csv')

data_loader = create_dataloader(bs=12, custom_dataloader=LTHTDataloader, dataframe=ltht_data)

model = VGGNestedUNet(num_classes=1)

batch_x, batch_y = next(iter(data_loader))
grid_img = make_grid(batch_x.permute(0,3,1,2), nrow=4)
plt.imshow(grid_img.permute(1, 2, 0))
plt.tight_layout()
plt.savefig('/MULTIX/DATA/HOME/covid-19-benchmarking/data/alb_batch_cxr.png')