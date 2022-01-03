# nccid analysis

import pandas as pd
import numpy as np
import datetime
import os
from pandas.core.indexes.multi import MultiIndex
from tqdm import tqdm
import pydicom as dicom
import csv
import seaborn as sns
import matplotlib.pyplot as plt


def collect_cxr_paths(dir):
    '''collect all paths in dir
    input: root dir
    return: dict of paths {pseudonym: [paths in subdir]}
    
    ========================
    NCCID DATA STRUCTURE
    ========================
    /MULTIX/DATA/INPUT_NCCID
     --- PSEUDONYM
         --- SUBDIR
             --- DCOM FILE
             '''

    pseudonym_path_dict = {}
    assert os.path.exists(dir)
    print('collecting paths...')
    for path, subdirs, files in os.walk(dir):
        for name in files:
            if name.endswith('.dcm'):
                pseudonym = path.split('/')[6]
                id = collect_dcom_data(os.path.join(path, name))
                if pseudonym in pseudonym_path_dict:
                    pseudonym_path_dict[pseudonym].append(os.path.join(path, name))
                else:
                    pseudonym_path_dict[pseudonym] = [os.path.join(path, name)]
    
    return pseudonym_path_dict

def collect_dcom_data(path):
    ds = dicom.read_file(path)
    id = ds.StudyInstanceUID
    return id


# clinical data
patients = pd.read_csv('/MULTIX/DATA/INPUT_NCCID/training/patients.csv')
patients = patients.fillna("NA")
gr = patients.groupby('Pseudonym')
print('NUMBER OF DIFFERENT PATIENTS: ', len(gr))
print(len(patients))

pcr_df = patients[['Pseudonym','1st RT-PCR result','1st_rt-pcr_result','2nd_rt-pcr_result','final_covid_status']]
print(pcr_df.head(10))

# image data
xray = pd.read_csv('/MULTIX/DATA/INPUT_NCCID/training/xray_id.csv')

## get chest x-ray image paths

#pseudonym_path_dict = collect_cxr_paths('/MULTIX/DATA/INPUT_NCCID/training/xray')
#path_df = pd.DataFrame.from_dict(pseudonym_path_dict,orient='index').transpose()
#path_df.to_csv('/MULTIX/DATA/INPUT_NCCID/training/paths.csv')

# deal with paths csv
#paths = pd.read_csv('/MULTIX/DATA/INPUT_NCCID/training/paths.csv', dtype=str)
#print(paths.head(5))

#paths = csv.reader(open('/MULTIX/DATA/INPUT_NCCID/training/paths.csv', 'r'))
#paths_dict = {}

#paths = paths.reset_index(drop=True)
#paths = paths.drop(columns = 'Unnamed: 0')
#print(paths.head(5))

def get_studyid(row):
    if pd.isna(row):
        pass
    else:
        id = row.split('/')[-1]
        id = id.split('.')[0:-1] #remove .dcm suffix
        id = '.'.join(id)
        return id

#print(xray.columns)
#xray['path'] = None
#for col in paths:
 #   if col == 'Unnamed: 0':
 #       pass
 #   else:
  #      for idx, row in paths[col].iteritems():
  #          id = get_studyid(row)
   #         if id is not None:
   #             for idx, xrow in tqdm(xray.iterrows(), total=xray.shape[0]):
    #                if str(id) == str(xrow['SOPInstanceUID']):
     #                   xray.loc[idx,'path'] = row
     #                   break


#xray.to_csv('/MULTIX/DATA/INPUT_NCCID/training/xray_id.csv')
# find number of unique paths in cxr_id - gives number of images
print(len(xray['path'].unique()))
print(len(set(xray['path'].values)))

# find frequency of cxrs per patient
print("FIND FREQ OF CXRS PER PATIENT")
counts = xray['Pseudonym'].value_counts().to_dict()
vals = np.array(counts.values())
unique, counts = np.unique(vals, return_counts=True)
print(unique[0])
u, val = np.unique(unique[0], return_counts=True)
print(np.asarray((u, val)).T)

# number of unique patients
print(len(patients['Pseudonym']))
print(len(patients['Pseudonym'].unique())) # check no repeats in patient dataframe

# merge xray and clinical data - get repeats of patients (according to the number of cxrs)
total_data = pd.merge(xray, patients, on=['Pseudonym'], how="left")
total_data.fillna("NA")
#total_data['overall_rtpcr'] = None
#for idx, row  in total_data.iterrows():
  #  rt_pcr = row[['1st_rt-pcr_result','2nd_rt-pcr_result','final_covid_status']].values
  #  if 'Positive' in str(rt_pcr):
  #      total_data[idx, 'overall_rtpcr'] = 'Positive'
 #   elif 'Negative' in str(rt_pcr):
  #      total_data[idx, 'overall_rtpcr'] = 'Negative'
  #  else:
   #     total_data[idx, 'overall_rtpcr'] = 'Unknown'


# number of cxrs according to diff variables
def by_var(df, var):
    print(str(i))
    print(df[var].value_counts(dropna=False))

# total data analysis
#for i in ['sex_update','ethnicity', 'filename_covid_status', 'smoking_status', 'ViewPosition']:
#    by_var(total_data, i)
#    plt.figure()
 #   total_data[i] = total_data[i].fillna("NA")
 #   sns.countplot(x=i, data=total_data)
 #   plt.xticks(rotation=45)
 #   plt.savefig(f'/MULTIX/DATA/HOME/covid-19-benchmarking/data/nccid_analysis/{i}_merged.png')
 #   plt.show()
#plt.show()

# patient data analysis
for i in ['sex_update','ethnicity', 'filename_covid_status','smoking_status']:
    by_var(patients, i)
    plt.figure()
    patients[i] = patients[i].fillna("NA")
    sns.countplot(x=i, data=patients)
    plt.savefig(f'/MULTIX/DATA/HOME/covid-19-benchmarking/data/nccid_analysis/{i}_patient.png')
    plt.xticks(rotation=45)
    plt.show()

patients = patients[patients['age_update'] != 'NA']
patients['age_update'] = patients['age_update'].astype(np.float)

#ax = patients.hist(column='Age', bins=25, grid=False, figsize=(12,8), rwidth=0.9)
#plt.savefig(f'/MULTIX/DATA/HOME/covid-19-benchmarking/data/nccid_analysis/age_hist_patient.png')
#plt.show()

plt.figure()

covid_neg = patients[patients['filename_covid_status']!=True]
covid_pos = patients[patients['filename_covid_status']==True]

x = covid_pos['age_update']
y = covid_neg['age_update']
print(covid_neg)

bins = np.linspace(np.min(patients['age_update'].values), np.max(patients['age_update'].values), 100)

plt.hist(x, bins, alpha=0.5, label='covid-positive')
plt.hist(y, bins, alpha=0.5, label='covid-negative')
plt.legend(loc='upper right')
plt.savefig('/MULTIX/DATA/HOME/covid-19-benchmarking/data/nccid_analysis/age_hist_grouped_patient.png')
plt.show()


pd.crosstab(total_data['ViewPosition'].fillna("NA"), total_data['filename_covid_status']).plot.bar()
plt.xticks(rotation=45)
plt.savefig(f'/MULTIX/DATA/HOME/covid-19-benchmarking/data/nccid_analysis/tot_view_bycovid.png')

views = xray['ViewPosition'].value_counts(dropna=False).to_dict()
print(views)
print(views.keys())




