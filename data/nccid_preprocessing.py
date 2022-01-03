import pandas as pd
import numpy as np
import pydicom
from pydicom.pixel_data_handlers import apply_modality_lut, apply_voi_lut
import os
import cv2
from tqdm import tqdm
import pathlib

def dicom_fn(dcm_file):
    # Load DICOM image
    ds = pydicom.dcmread(dcm_file)

    # Apply transformations if required
    if ds.pixel_array.dtype != np.uint8:
        # Apply LUT transforms
        arr = apply_modality_lut(ds.pixel_array, ds)
        if arr.dtype == np.float64 and ds.RescaleSlope == 1 and ds.RescaleIntercept == 0:
            arr = arr.astype(np.uint16)
        arr = apply_voi_lut(arr, ds)
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

df = pd.read_csv('/MULTIX/DATA/HOME/covid-19-benchmarking/data/nccid_prepared.csv')

mapping = {0.0:'positive',1.0:'negative'}
df['xray_status'] = df['xray_status'].map(mapping)
print(df['xray_status'].value_counts(dropna=False))
# create dir for writing out image
for i in df['xray_status'].unique():
    if i != np.nan:
        pathlib.Path(f'/MULTIX/DATA/INPUT_NCCID/nccid/{i}').mkdir(parents=True) 

for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    img = dicom_fn(row['path'])
    path = row['path'].split('/')[6:]
    path = ('_').join(path)
    path = path[:-4] + '.png'
    status = row['xray_status']
    structured_path = f'/MULTIX/DATA/INPUT_NCCID/nccid/{status}/{path}'
    df.loc[idx, 'structured_path'] = structured_path
    # save as gray