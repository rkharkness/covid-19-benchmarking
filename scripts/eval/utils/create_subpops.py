import pandas as pd
import numpy as np

def create_subpopulations(df, test):
    df['PseudoPatientID'] = df['PseudoPatientID_x']
    demo_df = pd.read_csv("/MULTIX/DATA/INPUT/LTH20029_CovidX/CovidX_ClinicalData/LTH20029_CovidX_Comorbidities.txt", sep="|")
    demo_df = demo_df[~pd.isnull(demo_df['ComorbidityName'])]
    demo_df = demo_df.groupby('PseudoPatientID')['ComorbidityName'].apply(list).reset_index(name='Comorbidities')
    demo_df = pd.merge(demo_df, df, on='PseudoPatientID',how='right')

    demo_df = demo_df[~pd.isnull(demo_df['Comorbidities'])]

    all_comorbidities = list(demo_df['Comorbidities'])
    comorbidities = [x for xs in all_comorbidities for x in xs]

    comorbidities = np.unique(np.array(comorbidities), axis=0)
    comorbidity_map = {"Chest X-ray Comorbidity":["Cancer","Malignancy","MalignancyFinal","MI","HPTN","Cardiomyopathy","CCF","CAD","Obese","Obesity","COPD","Asthma", "Resp","Restrictive"], "No Chest X-ray Comorbidity":["PAD","Thromb","Stroke","TIA","Demyelination","Parkinsons","Stroke","Paraplegia","Spinal","Dementia","Gout","PUD","Malabsorption","RA","Pancreatitis","Rheum","AS","T1Diabetes","T2Diabetes","OtherDiabetes","AnyDiabetes","Diabetic_Bloods", "Neuromuscular","MND","HIV","HIV_Result","HIVFinal","Liver"]}

    reverse_comorbidity_map = {x:k for k,v in comorbidity_map.items() for x in v}
    comorbidities = pd.Series(comorbidities).map(reverse_comorbidity_map)

    comorbidities_dict = {k: [] for k in comorbidities}
    for idx, row in demo_df.iterrows():
        for idx, c in enumerate(row['Comorbidities']):
            if c in comorbidity_map['Chest X-ray Comorbidity']:
                appearance = True
            else:
                appearance = False

        if appearance == False:
            comorbidities_dict['No Chest X-ray Comorbidity'].append(row)
        else:
            comorbidities_dict['Chest X-ray Comorbidity'].append(row)

    comorbidities_dict = {k: pd.DataFrame(v) for k, v in comorbidities_dict.items()}
    for k, v in comorbidities_dict.items():
        print(k, len(v))

    df['Age'] = [2022 - i for i in df['YearofBirth'].values]
    df['AgeGroup'] = df['Age']//25

    age_mapping = {0:" Age Group: 0-24", 1:"Age Group: 25-49", 2:"Age Group: 50-74",
        3:"Age Group: 75-99", 4:"Age Group: 100-125", 5:"Age Group: 100-125"}   
    df['AgeGroup'] = df['AgeGroup'].map(age_mapping)

    age_df = df.dropna(subset=['AgeGroup'])
    age_dict= {k: list(v) for k, v in age_df.groupby('AgeGroup')}
    for k, v in age_dict.items():
        age_dict[k] = age_df[age_df['AgeGroup']==k]

    df = df.reset_index(drop=True)
    ethnicity_df = df.dropna(subset=['EthnicCategoryCode'])
    ethnic_map = {'A': 'White', 'B': 'White', 'C': 'White', 'D': 'Multiple', 'E': 'Multiple',
    'F': 'Multiple', 'G': 'Multiple', 'H': 'Asian', 'J': 'Asian', 'K': 'Asian',
    'L': 'Asian', 'M': 'Black', 'N': 'Black', 'P': 'Black',
    'R':'Asian', 'S': 'Other', 'Z':'Unknown', 'G': 'Other'}

    ethnicity_df['MultiEthnicCategoryCode'] = ethnicity_df['EthnicCategoryCode'].map(ethnic_map)

    ethnicity_dict= {k: list(v) for k, v in ethnicity_df.groupby('MultiEthnicCategoryCode')}
    for k, v in ethnicity_dict.items():
        ethnicity_dict[k] = ethnicity_df[ethnicity_df['MultiEthnicCategoryCode']==k]

    ethnicity_dict = {k:v for k,v in ethnicity_dict.items() if len(v) > 20}

    bin_ethnic_map = {'A': 'White', 'B': 'White', 'C': 'White', 'D': 'Rest', 'E': 'Rest',
    'F': 'Rest', 'G': 'Rest', 'H': 'Rest', 'J': 'Rest', 'K': 'Rest','L': 'Rest', 'M': 'Rest', 'N': 'Rest', 'P': 'Rest',
    'R':'Rest', 'S': 'Rest', 'Z':'Rest'}
    ethnicity_df['BinEthnicCategoryCode'] = ethnicity_df['EthnicCategoryCode'].map(bin_ethnic_map)
    bin_ethnicity_dict= {k: list(v) for k, v in ethnicity_df.groupby('BinEthnicCategoryCode')}
    for k, v in ethnicity_dict.items():
        bin_ethnicity_dict[k] = ethnicity_df[ethnicity_df['BinEthnicCategoryCode']==k]

    ethnicity_dict = {k:v for k,v in ethnicity_dict.items() if len(v) > 20}
    gender_map = {'Male':'M', 'Female':'F', 'M':'M','F':'F'}

    df['Gender'] = df['Gender'].map(gender_map)

    gender_df = df.dropna(subset=['Gender'])
    gender_dict= {k: list(v) for k, v in gender_df.groupby('Gender')}
    for k, v in gender_dict.items():
        gender_dict[k] = gender_df[gender_df['Gender']==k]

    age_dict = {k: v for k, v in age_dict.items() if len(v) > 10}

    smoked_df = df.dropna(subset=['SmokedLastThreeMonths'])
    smoked_map = {'used': 'Former', 'ex':'Former', 'smoked':'Current', 'never':'Never'}
    smoked_df['SmokingStatus'] = smoked_df['SmokedLastThreeMonths'].map(smoked_map)
    smoked_dict= {k: list(v) for k, v in smoked_df.groupby('SmokingStatus')}

    for k, v in smoked_dict.items():
        smoked_dict[k] = smoked_df[smoked_df['SmokingStatus']==k]

    del comorbidities_dict[np.nan]
    subgroup_dicts = {'Gender':gender_dict}
    subgroup_dicts = {'Comorbidity':comorbidities_dict, 'Ethnicity':ethnicity_dict, 'Smoker':smoked_dict, 'Age':age_dict, 'Gender':gender_dict}


    for k,sub in subgroup_dicts.items():
        for sub_k, sub_v in sub.items():
            print(sub_k)
            print(sub_v['finding'].value_counts())

    return subgroup_dicts