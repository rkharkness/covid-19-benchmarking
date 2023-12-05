import pandas as pd 
import numpy as np

def find_view(a,b):
    print(a,b)
    if 'PA' in a.upper():
        return 'PA'
    elif 'AP' in a.upper():
        return 'AP' #a.upper() =='CHEST':
    elif b.upper() == 'PA':
        return 'PA'
    elif b.upper() == 'AP':
        return 'AP'
#                         return 'AP'
    else:
        return None

df = pd.read_csv('/MULTIX/DATA/HOME/covid-19-benchmarking/data/nccid_preprocessed14_21.csv')
df = df[df['kfold_1']!='test']
df['SeriesDescription'] = df['SeriesDescription'].fillna('nan')
df['ViewPosition'] = df['ViewPosition'].fillna('nan')
df['view'] = df[['SeriesDescription','ViewPosition']].apply(lambda x: find_view(*x), axis=1)

print(df[['xray_status','view']].value_counts())
