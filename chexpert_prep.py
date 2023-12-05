import pandas as pd
import numpy as np

data_path = "/MULTIX/DATA/HOME/"
full_train_df = pd.read_csv(data_path + 'CheXpert-v1.0-small/train.csv')
full_valid_df = pd.read_csv(data_path + 'CheXpert-v1.0-small/valid.csv')

#full_train_df['Path']
paths = [i.split('/')[1:] for i in full_train_df['Path'].values]
paths = ['/'.join(i) for i in paths]

full_train_df['Path'] = ['CheXpert-v1.0-small/' + i for i in paths]
print(full_train_df['Path'])
#chexpert_targets = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion', 'No Finding']

full_train_df.iloc[:,6:] = full_train_df.iloc[:,6:].fillna(0).replace(-1.0,0)
full_valid_df.iloc[:,6:] = full_valid_df.iloc[:,6:].fillna(0).replace(-1.0,0)

full_train_df['outcome'] = np.array(full_train_df.iloc[:,:].sum(axis=1) == 0).astype(int)
full_valid_df['outcome'] = np.array(full_valid_df.iloc[:,:].sum(axis=1) == 0).astype(int)

full_train_df = full_train_df.reset_index(drop=True)
full_valid_df = full_valid_df.reset_index(drop=True)
            
full_train_df['split'] = 'train'
full_valid_df['split'] = 'val'           

full_train_df=full_train_df.loc[full_train_df['Frontal/Lateral']=='Frontal']
full_valid_df=full_valid_df.loc[full_valid_df['Frontal/Lateral']=='Frontal']

full_train_df['patient'] = full_train_df.Path.str.split('/',3,True)[2]
full_train_df  ['study'] = full_train_df.Path.str.split('/',4,True)[3]

full_valid_df['patient'] = full_valid_df.Path.str.split('/',3,True)[2]
full_valid_df  ['study'] = full_valid_df.Path.str.split('/',4,True)[3]

full_df = pd.concat([full_train_df, full_valid_df])
full_df.head()

# full_df['feature_string'] = full_df.apply(feature_string,axis = 1).fillna('')
# full_df['feature_string'] =full_df['feature_string'] .apply(lambda x:x.split(";"))
# full_df.head()

sample_perc = 0.00
train_only_df = full_df[full_df['split']=='train']
valid_only_df = full_df[full_df['split']=='val']
unique_patients = train_only_df.patient.unique()
mask = np.random.rand(len(unique_patients)) <= sample_perc
sample_patients = unique_patients[mask]

dev_df = train_only_df[full_train_df.patient.isin(sample_patients)]
train_df = train_only_df[~full_train_df.patient.isin(sample_patients)]
full_df = pd.concat([train_df, valid_only_df])

#full_df['outcomes'] = full_df[['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']].values

full_path = ["/MULTIX/DATA/HOME/" + i for i in full_df['Path'].values]
full_df['Path'] = full_path

full_df.to_csv('/MULTIX/DATA/HOME/covid-19-benchmarking/chexpert.csv')

