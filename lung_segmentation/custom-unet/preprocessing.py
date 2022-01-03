import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

import matplotlib.patches as patches

from datetime import datetime
from glob import glob
from tqdm import tqdm

import pandas as pd

    # set width and height 
def save_raw_img(filepath, h=2048, w=2048):
  shape = (h,w)
  dtype = np.dtype('>i2')
  path, img_name = filepath.split('/')[0:-1], filepath.split('/')[-1]
  path = '/'.join(path)
  output_filename = str(img_name.split('.')[0]) + '.png'
  output_filepath = os.path.join(path, output_filename)

  img = open(filepath, 'rb')
  data = np.fromfile(img, dtype).reshape(shape)

  plt.imshow(data, cmap=plt.get_cmap('gray'))
  plt.savefig(output_filepath)

  return data


def read_gif(gif_file):
    obj = cv2.VideoCapture(gif_file)
    ret, image = obj.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print('read gif -', image.shape)
    obj.release()

    return image


def create_dirs(source_dir):

  seg_dir = os.path.join(source_dir, "segmentation")
  seg_test_dir = os.path.join(seg_dir, "test")
  seg_train_dir = os.path.join(seg_dir, "train")
  seg_aug_dir = os.path.join(seg_train_dir, "augmentation")
  seg_image_dir = os.path.join(seg_train_dir, "image")
  seg_mask_dir = os.path.join(seg_train_dir, "mask")
  seg_dilate_dir = os.path.join(seg_train_dir, "dilate")
  seg_test_image_dir = os.path.join(seg_test_dir, "image")
  seg_test_mask_dir = os.path.join(seg_test_dir, "mask")
  seg_test_dilate_dir = os.path.join(seg_test_dir, "dilate")
  
  if not os.path.exists(seg_dir):
    os.makedirs(seg_dir)
  
  if not os.path.exists(seg_test_dir):
    os.makedirs(seg_test_dir)
  
  if not os.path.exists(seg_train_dir):
    os.makedirs(seg_train_dir)
  
  if not os.path.exists(seg_image_dir):
    os.makedirs(seg_image_dir)
  
  if not os.path.exists(seg_aug_dir):
    os.makedirs(seg_aug_dir)
  
  if not os.path.exists(seg_mask_dir):
    os.makedirs(seg_mask_dir)
  
  if not os.path.exists(seg_dilate_dir):
    os.makedirs(seg_dilate_dir)

  if not os.path.exists(seg_test_image_dir):
    os.makedirs(seg_test_image_dir)
  
  if not os.path.exists(seg_test_mask_dir):
    os.makedirs(seg_test_mask_dir)
  
  if not os.path.exists(seg_test_dilate_dir):
    os.makedirs(seg_test_dilate_dir)

  return seg_dir,seg_test_dir,seg_train_dir,seg_aug_dir,seg_image_dir,seg_mask_dir,seg_dilate_dir


def preprocess_images(shen_folder, mont_folder, jsrt_folder, seg_image_dir, seg_mask_dir, seg_dilate_dir, seg_test_dir):

  shen_mask_folder = os.path.join(shen_folder, "mask")
  mont_seg_folder = os.path.join(mont_folder, "ManualMask")
  jsrt_img_folder = os.path.join(jsrt_folder, "All247images")

  mont_img_folder = os.path.join(mont_folder, "CXR_png")
  shen_img_folder = os.path.join(shen_folder, "ChinaSet_AllFiles/CXR_png") #/content/drive/My Drive/LungSegmentation/ChinaSet_AllFiles/ChinaSet_AllFiles/ChinaSet_AllFiles/CXR_png

  mont_lmask_folder = os.path.join(mont_seg_folder, "leftMask")
  mont_rmask_folder = os.path.join(mont_seg_folder, "rightMask")

  jsrt_lmask_folder_train = os.path.join(jsrt_folder, 'scratch/fold1/masks/left lung')

  #print('exists', os.path.isdir('/MULTIX/DATA/HOME/LungSegmentation_JSRT/scratch/fold1/masks/left lung/'))

  jsrt_lmask_folder_train = '/MULTIX/DATA/HOME/LungSegmentation_JSRT/scratch/fold1/masks/left lung/'
  jsrt_lmask_folder_test = '/MULTIX/DATA/HOME/LungSegmentation_JSRT/scratch/fold2/masks/left lung/'
  
  jsrt_rmask_folder_train = '/MULTIX/DATA/HOME/LungSegmentation_JSRT/scratch/fold1/masks/right lung/'
  jsrt_rmask_folder_test = '/MULTIX/DATA/HOME/LungSegmentation_JSRT/scratch/fold2/masks/right lung/'

  jsrt_lmask_folders = [jsrt_lmask_folder_train, jsrt_lmask_folder_test]
  jsrt_rmask_folders = [jsrt_rmask_folder_train, jsrt_rmask_folder_test]
  
  jsrt_lmask_files = [glob(os.path.join(i, '*.gif')) for i in jsrt_lmask_folders]
  jsrt_rmask_files = [glob(os.path.join(i, '*.gif')) for i in jsrt_rmask_folders]

  jsrt_lmask_files = [item for sublist in jsrt_lmask_files for item in sublist]
  jsrt_rmask_files = [item for sublist in jsrt_rmask_files for item in sublist]


  # Combine left and right lung masks for montgomery dataset
  mont_lmask_files = glob(os.path.join(mont_lmask_folder, '*.png'))
  mont_lmask_files = sorted(mont_lmask_files)

  mont_test = mont_lmask_files[0:50]
  mont_train = mont_lmask_files[50:]

  shen_mask_files = glob(os.path.join(shen_mask_folder, '*.png'))
  shen_mask_files = sorted(shen_mask_files)
  shen_test = shen_mask_files[0:50]
  shen_train = shen_mask_files[50:]

  dilate_kernel = np.ones((15,15), np.uint8) # pixels to pad

  write_images(mont_lmask_files,mont_rmask_folder,mont_img_folder,dilate_kernel,seg_image_dir,seg_mask_dir,
              mont_train,seg_dilate_dir,seg_test_dir,shen_img_folder,shen_mask_files,shen_train,jsrt_img_folder,jsrt_lmask_files,jsrt_rmask_folders)

  return mont_train, mont_test, shen_train, shen_test

def write_images(mont_lmask_files,mont_rmask_folder,mont_img_folder,dilate_kernel,
                 seg_image_dir,seg_mask_dir,mont_train,seg_dilate_dir,seg_test_dir,
                 shen_img_folder,shen_mask_files,shen_train,jsrt_image_folder,jsrt_lmask_files,jsrt_rmask_folder):
  
  # create dir to save to
  seg_test_image_dir = os.path.join(seg_test_dir, "image")
  seg_test_mask_dir = os.path.join(seg_test_dir, "mask")
  seg_test_dilate_dir = os.path.join(seg_test_dir, "dilate")

  # deal with jsrt images
  for lmask_file in tqdm(jsrt_lmask_files):
    base_file = os.path.basename(lmask_file)
    base = base_file.split('.')[0]
    img_file = os.path.join(jsrt_image_folder, base + '.IMG')

    path = lmask_file.split('/')[:-2]
    path = '/'.join(path)
    rmask_file = os.path.join(path, 'right lung', base_file)

    l_img = read_gif(lmask_file)
    r_img = read_gif(rmask_file)
    image = save_raw_img(img_file)
    image = cv2.imread(os.path.join(jsrt_image_folder, base + '.png'))

    image = cv2.resize(image, (480, 480))
    l_img = cv2.resize(l_img, (480, 480))
    r_img = cv2.resize(r_img, (480, 480))

    mask = np.maximum(l_img,r_img)
    #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    print("size of mask = ", mask.shape)
    mask_dilate = cv2.dilate(mask, dilate_kernel, iterations=1)
    # print("size of dilated mask = ", mask.shape)
    save_file = base + '.png'

    if 'fold1' in rmask_file:
      cv2.imwrite(os.path.join(seg_image_dir, save_file), image)
      cv2.imwrite(os.path.join(seg_mask_dir, save_file), mask)
      cv2.imwrite(os.path.join(seg_dilate_dir, save_file), mask_dilate)
    else:
    #  filename, fileext = os.path.splitext(save_file)
      cv2.imwrite(os.path.join(seg_test_image_dir, save_file), image)
      cv2.imwrite(os.path.join(seg_test_mask_dir, save_file), mask)
      cv2.imwrite(os.path.join(seg_test_dilate_dir, save_file), mask_dilate)

  for lmask_file in tqdm(mont_lmask_files):
    base_file = os.path.basename(lmask_file)
    img_file = os.path.join(mont_img_folder, base_file)
    rmask_file = os.path.join(mont_rmask_folder, base_file)

    l_img = cv2.imread(lmask_file, cv2.IMREAD_GRAYSCALE)
    r_img = cv2.imread(rmask_file, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(img_file)

    image = cv2.resize(image, (480, 480))
    l_img = cv2.resize(l_img, (480, 480))
    r_img = cv2.resize(r_img, (480, 480))

    mask = np.maximum(l_img,r_img)
    #print("size of mask = ", mask.shape)
    mask_dilate = cv2.dilate(mask, dilate_kernel, iterations=1)
    #print("size of dilated mask = ", mask.shape)
    if lmask_file in mont_train:
      cv2.imwrite(os.path.join(seg_image_dir, base_file), image)
      cv2.imwrite(os.path.join(seg_mask_dir, base_file), mask)
      cv2.imwrite(os.path.join(seg_dilate_dir, base_file), mask_dilate)
    else:
    #  filename, fileext = os.path.splitext(base_file)
      cv2.imwrite(os.path.join(seg_test_image_dir, base_file), image)
      cv2.imwrite(os.path.join(seg_test_mask_dir, base_file), mask)
      cv2.imwrite(os.path.join(seg_test_dilate_dir, base_file), mask_dilate)

  for mask_file in tqdm(shen_mask_files):
    base_file = os.path.basename(mask_file).replace("_mask", "")
    img_file = os.path.join(shen_img_folder, base_file)

    m_img = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(img_file)

    image = cv2.resize(image, (480, 480))
    mask = cv2.resize(m_img, (480, 480))
    mask_dilate = cv2.dilate(mask, dilate_kernel, iterations=1)
    if mask_file in shen_train:
      cv2.imwrite(os.path.join(seg_image_dir, base_file), image)
      cv2.imwrite(os.path.join(seg_mask_dir, base_file), mask)
      cv2.imwrite(os.path.join(seg_dilate_dir, base_file), mask_dilate)
    else:
     # filename, fileext = os.path.splitext(base_file)
      cv2.imwrite(os.path.join(seg_test_image_dir, base_file), image)
      cv2.imwrite(os.path.join(seg_test_mask_dir, base_file), mask)
      cv2.imwrite(os.path.join(seg_test_dilate_dir, base_file), mask_dilate)


  

if __name__ == "__main__":
  root_dir = "/MULTIX/DATA/HOME/LungSegmentation_JSRT/"

  seg_dir,seg_test_dir,seg_train_dir,seg_aug_dir,seg_image_dir,seg_mask_dir,seg_dilate_dir = create_dirs(root_dir)

  seg_test_image_dir = os.path.join(seg_test_dir, "image")
  seg_test_mask_dir = os.path.join(seg_test_dir, "mask")
  seg_test_dilate_dir = os.path.join(seg_test_dir, "dilate")

  shen_folder = root_dir + "Shenzhen-CXR"
  mont_folder = root_dir + "MontgomerySet"

  jsrt_folder = root_dir
  
  mont_train, mont_test, shen_train, shen_test = preprocess_images(shen_folder, mont_folder, jsrt_folder, seg_image_dir, seg_mask_dir, seg_dilate_dir, seg_test_dir)

  train_dilate_path = [os.path.abspath(os.path.join(root_dir + "segmentation/train/dilate", p)) for p in os.listdir(root_dir + "segmentation/train/dilate")]
  train_image_path = [os.path.abspath(os.path.join(root_dir + "segmentation/train/image", p)) for p in os.listdir(root_dir + "segmentation/train/image")]
  train_mask_path = [os.path.abspath(os.path.join(root_dir + "segmentation/train/mask", p)) for p in os.listdir(root_dir + "segmentation/train/mask")]

  train_df = pd.DataFrame([train_dilate_path, train_image_path, train_mask_path])
  train_df = train_df.T
  train_df.columns = ["dilate","image","mask"]
  train_df['split'] = "train"

  print(f"Size of train dataset {len(train_df)}")

  test_dilate_path = [os.path.abspath(os.path.join(root_dir + "segmentation/test/dilate", p)) for p in os.listdir(root_dir + "segmentation/test/dilate")]
  test_image_path = [os.path.abspath(os.path.join(root_dir + "segmentation/test/image", p)) for p in os.listdir(root_dir + "segmentation/test/image")]
  test_mask_path = [os.path.abspath(os.path.join(root_dir + "segmentation/test/mask", p)) for p in os.listdir(root_dir + "segmentation/test/mask")]

  test_df = pd.DataFrame([test_dilate_path, test_image_path, test_mask_path])
  test_df = test_df.T
  test_df.columns = ["dilate","image","mask"]
  test_df['split'] = "test"

  print(f"Size of test dataset {len(test_df)}")

  full_df = pd.concat([train_df,test_df])

  print(f"Size of full dataset {len(full_df)}")

  full_df.to_csv("/MULTIX/DATA/HOME/lung_segmentation_data/lung_segmentation_data.csv")