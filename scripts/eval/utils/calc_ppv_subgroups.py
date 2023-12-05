import json 
import pandas as pd 
import numpy as np 
import argparse

def create_subpopulations(df, test):
    #def create_subgroups(df):
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
    print(df['Age'].value_counts())
    df['AgeGroup'] = df['Age']//25
    #age_mapping = {0:" Age Group: 0-24", 1:"Age Group: 25-49", 2:"Age Group: 50-74", \
     #   3:"Age Group: 75-99", 4:"Age Group: 100-124", 5:"Age Group: 125+"}
    age_mapping = {0:" Age Group: 0-24", 1:"Age Group: 25-49", 2:"Age Group: 50-74", 
        3:"Age Group: 75-99", 4:"Age Group: 100-125", 5:"Age Group: 100-125"}    
    df['AgeGroup'] = df['AgeGroup'].map(age_mapping)
    print(df['AgeGroup'].value_counts())
    
    age_df = df.dropna(subset=['AgeGroup'])
    age_dict= {k: list(v) for k, v in age_df.groupby('AgeGroup')}
    for k, v in age_dict.items():
        age_dict[k] = age_df[age_df['AgeGroup']==k]
        
    sorted_age_dict = sorted(age_dict.keys(), key=lambda x: int(x.split('-')[-1]))
    df = df.reset_index(drop=True)
    ethnicity_df = df.dropna(subset=['EthnicCategoryCode'])
    #ethnicity_df = ethnicity_df[ethnicity_df['EthnicCategoryCode']!='Z']
  #  ethnicity_df = ethnicity_df[ethnicity_df['EthnicCategoryCode']!='S']
   # ethnicity_df = ethnicity_df[ethnicity_df['EthnicCategoryCode']!='G']
    
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
    print(df['Gender'].value_counts())
    gender_map = {'Male':'M', 'Female':'F'}
    
    df['Gender'] = df['Gender'].map(gender_map)
    print(df['Gender'].value_counts())
    
    gender_df = df.dropna(subset=['Gender'])
    gender_dict= {k: list(v) for k, v in gender_df.groupby('Gender')}
    for k, v in gender_dict.items():
        gender_dict[k] = gender_df[gender_df['Gender']==k]             
    
    age_dict = {k: v for k, v in age_dict.items() if len(v) > 10}  

    smoked_df = df.dropna(subset=['SmokedLastThreeMonths'])
    print(smoked_df['SmokedLastThreeMonths'].value_counts())
    smoked_map = {'used': 'Former', 'ex':'Former', 'smoked':'Current', 'never':'Never'}
    smoked_df['SmokingStatus'] = smoked_df['SmokedLastThreeMonths'].map(smoked_map)
    smoked_dict= {k: list(v) for k, v in smoked_df.groupby('SmokingStatus')}

    for k, v in smoked_dict.items():
        smoked_dict[k] = smoked_df[smoked_df['SmokingStatus']==k]   
    
    del comorbidities_dict[np.nan] 
    subgroup_dicts = {'Comorbidity':comorbidities_dict, 'Ethnicity':ethnicity_dict, 'Smoker':smoked_dict, 'Age':age_dict, 'Gender':gender_dict}
#    subgroup_dicts = {'Age':age_dict}
    return subgroup_dicts

class PPV:
    def __init__(self, data, test, results):
        self.data =  data
        self.test = test
        self.results_path = results
        self.results = self.load_results(results)
        
        self.groups = list(data.keys())
        #self.thresholds = self.load_results(thresholds)
        self.prevalence = self.calc_prevalence()

    def load_results(self, results):
        file = open(results)
        results_dict = json.load(file)

       # results_dict = results_dict['All']
        print('loading results with keys ...')
#        print(results_dict.keys())
        return results_dict

    def calc_prevalence(self):
        popsize_dict = {'F':1246, 'M':1658, 'Age Group: 0-24':1076,'Age Group: 100+': 109, \
       'Age Group: 50-74':3764, 'Age Group: 25-49':1497, 'Age Group: 75-99':4758, 'Current':958, 'Former':1548, 'Never':3742, \
       'Asian':726, 'Black':302, 'Multiple':72, 'Other':160, 'White':6737, 'Unknown':0}
        
        prevalence_dict = {}

        for g in self.groups:
            data_g = self.data[g]
            n = len(self.data[g])
            print(g, n)
            if self.test == 'nccid' or self.test =='covidgr' or self.test == 'ltht_pneumonia' or self.test == 'ltht_no_pneumonia':
                pos_cases = len(self.data[g][self.data[g]['xray_status']==1.0])
            elif self.test == 'ltht':
                pos_cases = len(self.data[g][self.data[g]['FinalPCR']==1.0])
            prevalence_dict[g] = pos_cases/n

   #    assert prevalence < 1
        return prevalence_dict
    
    def get_best_threshold_idx(self, k, g):
        roc_thresholds = self.results[g]['roc_thresholds'][k]
        best_threshold = 0.5
        roc_thresholds = np.around(roc_thresholds,1)
        print(roc_thresholds)
        try:   
     #[xv if c else yv for c, xv, yv in zip(x == y, roc_thresholds, best_threshold)]
            best_threshold_idx = np.argwhere(roc_thresholds == best_threshold)
            best_threshold_idx = best_threshold_idx[0][0]
        except:
            try:
                best_threshold = 0.4
                best_threshold_idx = np.argwhere(roc_thresholds == best_threshold)
                best_threshold_idx = best_threshold_idx[0][0]
            except:
                best_threshold = 0.7
                best_threshold_idx = np.argwhere(roc_thresholds == best_threshold)
                best_threshold_idx = best_threshold_idx[0][0]
        print(best_threshold_idx)
        return best_threshold_idx

    def get_tpr(self, best_threshold_idx, k, g):
       tprs = self.results[g]['tpr'][k]
       return tprs[best_threshold_idx]

    def get_fpr(self, best_threshold_idx, k, g):
       fprs = self.results[g]['fpr'][k]
       return fprs[best_threshold_idx]

    def get_recall(self, k, g):
       recall = self.results[g]['recall'][k]
       return recall

    def calc_ppv(self):
        groups_ppv = {k:[] for k in self.groups}
        for k in range(5):
            for g in self.groups:
                best_threshold_idx = self.get_best_threshold_idx(k,g)
                fpr = self.get_fpr(best_threshold_idx, k, g)
                tnr = 1. - fpr

                tpr = self.get_tpr(best_threshold_idx, k, g)
                recall = self.get_recall(k, g)
                ppv = (recall * self.prevalence[g]) / ((recall * self.prevalence[g]) + ((1. - tnr) * (1. - self.prevalence[g])))
                groups_ppv[g].append(ppv)

        return groups_ppv
   
    def write_to_results(self):
        self.groups_ppv = self.calc_ppv() 
        
        for g in self.groups:
            self.results[g]['ppv'] = self.groups_ppv[g]


       #self.results['All']['ppv'] = ppv_list
        print(self.results)
      # print(f"saving to {self.results_path} ...")
        new_path = self.results_path[:-5] + '_ppv.json'
        print(new_path)
        with open(new_path, 'w') as f:
            json.dump(self.results, f)
    # save when sure
        
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPV RECOVERY')
    parser.add_argument('--data_csv')
    parser.add_argument('--results_csv', default='/MULTIX/DATA/HOME/covid-19-benchmarking/data/nccid_preprocessed14_21.csv', type=str, help='Path to data file')
    parser.add_argument('--test', default=None, type=str, help='Choose on of: [ltht, nccid_test, nccid_val, nccid_leeds, chexpert, custom]')
    parser.add_argument('--threshold_data', type=str, help='Path to threshold data, generated by nccid_test i.e. root/model_name_performance_nccid_test_df.csv')
    parser.add_argument('--subpop', type=bool)
    parser.add_argument('--group', type=str)

    args = parser.parse_args()
   
    df = pd.read_csv(args.data_csv)
    if args.test == 'ltht_no_pneumonia':
        df = df[df['finding']!=np.nan]
        df = df[df['finding']!='pneumonia']
        df = df.dropna(subset=['finding'])

        mapping = {'covid-19':1, 'negative':0}

        df['xray_status'] = df['finding'].map(mapping)
        df['cxr_path'] = df['paths']
        df['kfold_1'] = 'test'
        data = df[df['kfold_1'] == 'test']
        if args.subpop == True:
            data = create_subpopulations(df, 'ltht')
            data = data[args.group]        

    elif args.test == 'ltht_pneumonia':
        df = df[df['finding']!=np.nan]
        mapping = {'covid-19':1, 'pneumonia':1, 'negative':0}
        df = df.dropna(subset=['finding'])
        df['xray_status'] = df['finding'].map(mapping)
        df['cxr_path'] = df['paths']
        df['kfold_1'] = 'test'

        data = df[df['kfold_1']=='test']
        
    elif args.test == 'nccid':
        df = df[df['xray_status']!=np.nan]
        df = df.dropna(subset=['xray_status'])
        data = df[df['kfold_1'] == 'test']

    elif args.test == 'ltht':
        data = df
        if args.subpop == True:
            data = create_subpopulations(df, 'ltht')
            data = data[args.group]
    calc_ppv = PPV(data, args.test, args.results_csv)
    calc_ppv.write_to_results()
