import pandas as pd
import numpy as np
import pydicom
from pydicom.pixel_data_handlers import apply_modality_lut, apply_voi_lut
import os
import cv2
from tqdm import tqdm
import pathlib

import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool
import argparse

def dicom_fn(dcm_file, apply_voi=True):
    # Load DICOM image
    ds = pydicom.dcmread(dcm_file)

    # Apply transformations if required
    if ds.pixel_array.dtype != np.uint8:

        # Apply LUT transforms
        arr = apply_modality_lut(ds.pixel_array, ds)
        
        if arr.dtype == np.float64 and ds.RescaleSlope == 1 and ds.RescaleIntercept == 0:
            arr = arr.astype(np.uint16)
        
        if apply_voi:
            arr = apply_voi_lut(arr, ds)
        
        else:
            arr = ds.pixel_array

        arr = arr.astype(np.float64)

        # Normalize to [0, 1]
        arr = (arr - arr.min())/arr.ptp()

        # Invert MONOCHROME1 images
        if ds.PhotometricInterpretation == 'MONOCHROME1':
            arr = 1. - arr

        # Convert to uint8
        image = np.uint8(255.*arr)

    else:
        # Invert MONOCHROME1 images
        if ds.PhotometricInterpretation == 'MONOCHROME1':
            image = 255 - ds.pixel_array
        else:
            image = ds.pixel_array

    image = cv2.resize(image, (480,480), interpolation = cv2.INTER_AREA) # best for downsampling

    return image
    
def keep_dicom(dcm_file):
    ds = pydicom.dcmread(dcm_file)
    return ds

def process_path(row):
    path = row['path'].split('/')[6:]
    path = ('_').join(path)
   # path = path[:-4] + '.png'
    return path 

def main(df, root_dir='nccid_dcm'):
    df['cxr_path'] = None
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        ds = keep_dicom(row['path'])
        path = process_path(row)
        structured_path = f'/MULTIX/DATA/INPUT_NCCID/{root_dir}/{path}'
        ds.save_as(structured_path)
        df.loc[idx, 'cxr_path'] = structured_path

    return df
  
#def main(df, root_dir='nccid_png'):
   # exceptions = 0
  #  df['cxr_path'] = None
  #  for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
      #  try:
    #        img = dicom_fn(row['path'])
    #        path = process_path(row)
    #        structured_path = f'/MULTIX/DATA/INPUT_NCCID/{root_dir}/{path}'
  #          cv2.imwrite(structured_path, img)
   #         df.loc[idx, 'cxr_path'] = structured_path

    #    except:
   # #        print(f"apply_voi error: {row['path']} - an exception occurred")
            
     #       try:
    ##            img = dicom_fn(row['path'], apply_voi=False)
    #            path, status = process_path(row)
    #            structured_path = f'/MULTIX/DATA/INPUT_NCCID/{root_dir}/{path}'
    #            cv2.imwrite(structured_path, img)
    #            df.loc[idx, 'cxr_path'] = structured_path

    #        except:
   #             print(f"unable to process image: {row['path']} - an exception occurred")
   #             exceptions += 1
    #print(exceptions)
    return df
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training Script')
    parser.add_argument('--data_csv', default='/MULTIX/DATA/INPUT_NCCID/nccid_preprocessed.csv', type=str, help='Path to data file')
    parser.add_argument('--save_dir', type=str, help='Name of dir to save images to')

    args = parser.parse_args()

    df = pd.read_csv(args.data_csv)
    df = df[~pd.isnull(df['path'])]
    print(df['xray_status'].value_counts(dropna=False))  
    # create dir for writing out image
#    pathlib.Path(f'/MULTIX/DATA/INPUT_NCCID/{args.save_dir}').mkdir(parents=True) 

    cores=mp.cpu_count()
    df_split = np.array_split(df, cores, axis=0)

    # create the multiprocessing pool
    pool = Pool(cores)

    # process the DataFrame by mapping function to each df across the pool
    df_out = pd.concat(pool.map(main, df_split))
    df_out = pd.DataFrame(df_out)

    df_out.to_csv(args.data_csv)
    # close down the pool and join
    pool.close()
    pool.join()
    pool.clear()

