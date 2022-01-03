# prep for use of nccid data
import pandas as pd
import numpy as np
import datetime
import os
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.multi import MultiIndex
from tqdm import tqdm
import pydicom as dicom
import csv
import seaborn as sns
import matplotlib.pyplot as plt

# match cxr path to cxr metadata
def xray_covid_status(df, before, after):
    '''inputs:
            df: dataframe
            days_around:  +/- days around to give window
        outputs: 
            dataframe with cxr outcome column'''
    df['xray_status'] = None
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        if row['filename_covid_status'] == True:
            if row["date_of_positive_covid_swab"] is not None:
                try:
                    date = row["date_of_positive_covid_swab"]
                    days_after = pd.date_range(start = date, periods=after, freq='D')
                    days_before = pd.date_range(end = date, periods=before, freq='D')
                    window1 = days_before.append(days_after)
                except:
                    print(f"POSITIVE SWAB: Window creation failure - due to {date}")       
            else:
                window1 = DatetimeIndex([])

            if row["1st_rt-pcr_result"]=="Positive":
                if row["date_of_result_of_1st_rt-pcr"] is not None:
                    try:
                        date = row["date_of_result_of_1st_rt-pcr"]
                        days_after = pd.date_range(start = date, periods=after, freq='D')
                        days_before = pd.date_range(end = date, periods=before, freq='D')
                        window2 = days_before.append(days_after)
                    except:
                        print(f"1ST RT-PCR: Window creation failure - due to {date}") 
                else:
                    window2 = DatetimeIndex([])
            else:
                window2 = DatetimeIndex([])
        
            if row["2nd_rt-pcr_result"]== "Positive":
                if row["2nd_rt-pcr_result"] is not None:
                    try:
                        date = row["date_of_result_of_2nd_rt-pcr"]
                        days_after = pd.date_range(start = date, periods=after, freq='D')
                        days_before = pd.date_range(end = date, periods=before, freq='D')
                        window3 = days_before.append(days_after)  
                    except:
                       print(f"2ND RT-PCR: Window creation failure - due to {date}")   
                else:
                    window3 = DatetimeIndex([])         
            else:
                window3 = DatetimeIndex([])      

            window = DatetimeIndex([])
            window_a = window.append(window1)
            window_b = window_a.append(window2)
            window_c = window_b.append(window3)

            xray_date = pd.to_datetime(row["AcquisitionDate"], format='%Y%m%d')

            if len(window_c) > 0:
                if xray_date in window_c:
                    df.loc[idx,'xray_status'] = 1.0 # inside window
                elif xray_date > max(window_c):
                    df.loc[idx,'xray_status'] = "AW" # after window
                elif xray_date < min(window_c):
                    df.loc[idx,'xray_status'] = "BW" # before window
        else:
            df.loc[idx,'xray_status'] = 0.0

    return df

# image data
xray = pd.read_csv('/MULTIX/DATA/INPUT_NCCID/training/xray_id.csv')

# clinical data
patients = pd.read_csv('/MULTIX/DATA/INPUT_NCCID/training/patients.csv', parse_dates=['date_of_positive_covid_swab','date_of_result_of_1st_rt-pcr',
'date_of_result_of_2nd_rt-pcr'])
total_data = pd.merge(xray, patients, on=['Pseudonym'], how="left")

date_cols = ['date_of_positive_covid_swab','date_of_result_of_1st_rt-pcr','date_of_result_of_2nd_rt-pcr']
total_data[date_cols] = total_data[date_cols].apply(pd.to_datetime)

total_data = total_data[~pd.isnull(total_data['AcquisitionDate'])] # remove cxrs without acquisition dates

total_data = total_data.replace({pd.NaT: np.nan})
total_data = total_data.replace({np.nan:None}) # convert all nan values to None



#df = xray_covid_status(total_data, 14, 28) # get swab date-specific cxr labels
#df.to_csv('/MULTIX/DATA/INPUT_NCCID/total_nccid.csv') # save df with cxr labels

df = pd.read_csv('/MULTIX/DATA/INPUT_NCCID/total_nccid.csv')

print(len(df))
# remove lateral cxrs
#df = df[df['Laterality']==None]

print(df['ViewPosition'].value_counts())
#views = ["Abdomen, AP", "ABDOMEN AP", "ADDOMEN, AP", "Abdomen AP", "Abdomen", "Lateral","LL", "Lateral L", "Lateral R","LATERAL", "Lateral Decub", "Supine", "RENAL", "T Abdomen Supine", "X Abdomen Supine", "Trolley Abdomen AP", "T Abdomen AP"]

views = ["AP","PA","CHEST AP", "CHEST PA", None]
descriptions = ['Chest PA', 'Chest AP', 'PA', 'AP', 'W Chest PA', 'X Chest AP', 
'Chest', 'PA Paed Fuji', 'Chest LAT', 'CHEST PA', 'AP Chest', 
'CHEST AP', 'PA Chest', 'PA CHEST', 'AP CHEST', 'Chest PA Grid', 'AP chest',
np.nan, 'PA NO GRID', 'CR (AP)', 'AP Wireless', 'PA Non Grid', 
'SAVED IMAGES', 'Virtual Grid AP', 'AP Landscape', 'AP NG', 'PA GRID', 
'Chest AP (M2 C PKG)', 'Chest Mobile', 'PA chest', 'Mobile AP', '1 YR Chest AP', '6 YR Chest AP', 'Renal', 'NGT AP', 
'AP Skyplate']

#
#['Chest PA', 'Chest AP', 'PA', 'AP', 'W Chest PA', 'X Chest AP', 'Chest', 'PA Paed Fuji', 'Chest LAT', 
#'Abdomen', 'CHEST PA', 'AP Chest', 'CHEST AP', 'PA Chest', 'PA CHEST', 'AP CHEST', 'Chest PA Grid', 'AP chest',
# np.nan, 'PA NO GRID', 'CR (AP)', 'Trolley Abdomen AP', 'AP Wireless', 'PA Non Grid', 'SAVED IMAGES', 
# 'Virtual Grid AP', 'LATERAL', 'AP Landscape', 'AP NG', 'PA GRID', 'Chest AP (M2 C PKG)', 'Chest Mobile', 
# 'PA chest', 'Mobile AP', 'Supine', 'AP Abdomen', 'Abdomen AP', '1 YR Chest AP', '6 YR Chest AP', 'Renal', 
# 'NGT AP', 'AP Skyplate']

# deal with before window cases
for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    if row['xray_status'] == 'BW':
        date = pd.to_datetime(row["AcquisitionDate"], format='%Y%m%d')
        if date.year <= 2019:
            if date.month < 11:
                df.loc[idx, 'xray_status'] = "0.0"

df = df[df['xray_status']!='BW']
df = df[df['xray_status']!='AW']

df['xray_status'] = df['xray_status'].astype(float)

print(df['xray_status'].value_counts())


# create training, val, test populations - ensure no patient overlap
patient_groups = df.groupby(['Pseudonym'], as_index=False)
group_size = patient_groups.size()
#group_size_val = group_size['size'].values

size_max = 10

# find overrepresented patients
#weighty_patients = [i for i in patient_groups['Pseudonym'] if len(i) > 8]
weighty_patients = group_size[group_size['size'] > size_max]

# get test group - incl. excess imgs from overrep patients
test_patients = patient_groups.sample(frac=0.1, random_state=1)
# remove excess imgs from overrep patients
def df_sampling(x, n):
    if x[0] in group_size['Pseudonym'].values:
        return x.sample(n).astype(np.int)
    else:
        return x

test_patients = test_patients.apply(lambda x: df_sampling(x, size_max), axis=1) # sample median number of cxrs from overrep patients
test_pseudonym = test_patients['Pseudonym'].unique()


# get train + val groups
training_bank = df[~df['Pseudonym'].isin(test_pseudonym)]

training_bank = training_bank.apply(lambda x: df_sampling(x, size_max), axis=1) # will find that some cxrs not incl in some kfold

val_cv = []
for k in range(1,6):
    print(f"Identifying splits for kfold: {k}")
    df_col = f"kfold_{k}"
    # prevent overrepresentation of severely ill patients
    if k == 1:
        val_patients = training_bank.sample(frac=0.20) # get val split - sample from training bank (without replacement)
    else:
        val_patients = val_bank.sample(frac=0.20)
        
    val_pseudonyms = set(val_patients['Pseudonym'].values)
    training_patients = training_bank[~training_bank['Pseudonym'].isin(val_pseudonyms)]

    val_cv.append([*val_pseudonyms,])
    val_cv = [item for items in val_cv for item in items]

    val_bank = training_bank[~training_bank['Pseudonym'].isin(val_cv)]

    for idx, row in tqdm(df.iterrows(), total= df.shape[0]):
        if row['path'] in test_patients['path'].values:
            df.loc[idx, df_col] = 'test'
        elif row['path'] in val_patients['path'].values:
            df.loc[idx, df_col] = 'val'
        elif row['path'] in training_patients['path'].values:
            df.loc[idx, df_col] = 'train'

    # pseudonym | accession | kfold_1 | ... | kfold_5
    # ===============================================
    # covid1000 | 103855432 |  test   | ... |  test
    # covid2000 | 534375980 |  train  | ... |  val
    # covid2001 | 547980570 |   val   | ... |  train



# sample accoridng ot time windows - vary the labels
# post anal - neg --> pos? 

# check no overlap in patients between data splits
print(df['kfold_1'].value_counts())
print(df['kfold_2'].value_counts())
print(df['kfold_3'].value_counts())
print(df['kfold_4'].value_counts())
print(df['kfold_5'].value_counts())

for i in range(1,6):
    test_df = df[df[f'kfold_{i}']=='test']
    val_df = df[df[f'kfold_{i}']=='val']
    train_df = df[df[f'kfold_{i}']=='train']
    print(f"kfold_{i}: Overlap between patients in test and train : {len(set(test_df['Pseudonym']).intersection(train_df['Pseudonym']))}")
    print(f"kfold_{i}: Overlap between patients in test and val : {len(set(test_df['Pseudonym']).intersection(val_df['Pseudonym']))}")
    print(f"kfold_{i}: Overlap between patients in val and train : {len(set(val_df['Pseudonym']).intersection(train_df['Pseudonym']))}")

overlaps = []
for i in val_df['Pseudonym'].unique():
 	if i in train_df['Pseudonym'].unique():
 		overlaps.append(i)
print(f"Overlap in train and val for kfold_1: {len(overlaps)}")

# find overlaps in validation patients across the cv
val_patients = []
for i in range(1,6):
    cols = f"kfold_{i}"
    val = df[df[cols]=='val']
    patients = set(val['Pseudonym'].values)
    val_patients.append(patients)

for i in range(4):
    overlap = [x for x in val_patients[i] if x in val_patients[i+1]]
    print(f"Overlap in val between kfold {i} and {i+1}: {len(overlap)}")


test_df = df[df[f'kfold_{i}']=='test']
val_df = df[df[f'kfold_{i}']=='val']
train_df = df[df[f'kfold_{i}']=='train']

print("TEST")
print(test_df['xray_status'].value_counts(dropna=False))
print("VAL")
print(val_df['xray_status'].value_counts(dropna=False))
print("TRAIN")
print(train_df['xray_status'].value_counts(dropna=False))

if len(overlaps) == 0:
    df.to_csv('/MULTIX/DATA/HOME/covid-19-benchmarking/data/nccid_prepared.csv')