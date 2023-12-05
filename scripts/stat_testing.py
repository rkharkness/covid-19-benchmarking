import argparse
import numpy as np
from scipy import stats
import json
#from scipy.stats import f_oneway

def kruskal_wallis(file, var):
    data = json.load(file)
    print(var)
    var_list = [np.array(data[i][var]) for i in data.keys()]
    
    subpops = data.keys()
    print(stats.kruskal(*var_list))

def anova(file, var):
    data = json.load(file)
    print(var)
    var_list = [np.array(data[i][var]) for i in data.keys()]
    
    subpops = data.keys()
    print(stats.f_oneway(*var_list))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training Script') 
    parser.add_argument('--results_csv', default=None, type=str, help='Path to data file')
    parser.add_argument('--subpop', default=False, type=bool, help='One of: Ethnicity, Comorbidity, Smoker, Gender, Age')
    args = parser.parse_args()
    vars = ['acc','roc_auc','f1','precision','recall']
    
    for i in vars:
        with open(args.results_csv) as f:
            kruskal_wallis(f, i)

    
    for i in vars:
        with open(args.results_csv) as f:
            anova(f, i)
