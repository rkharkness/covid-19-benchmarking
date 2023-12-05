import json 
import pandas as pd

import numpy as np

import argparse

from cycler import cycler

import json

import numpy as np 
import os

import glob 
from scipy import stats 
import matplotlib.pyplot as plt 
from matplotlib import rc

# activate latex text rendering
rc('text', usetex=True)
plt.style.use(['science','nature','scatter'])
#plt.gca().set_prop_cycle(None)

plt.rcParams['axes.prop_cycle'] = cycler('color', plt.get_cmap('tab20').colors)


class PPV:
    def __init__(self, data, test, results):
        self.data =  data
        self.test = test
        self.results_path = results
        self.results = results
        #self.results = self.load_results(results)
        #self.thresholds = self.load_results(thresholds)
     #   self.prevalence = self.calc_prevalence()

    def load_results(self, results):
        file = open(results)
        results_dict = json.load(file)

       # results_dict = results_dict['All']
        print('loading results with keys ...')
#        print(results_dict.keys())
        return results_dict

    def calc_prevalence(self):
       # popsize_dict = {'F':1246, 'M':1658, 'Age Group: 0-24':1076,'Age Group: 100+': 109, \
       #'Age Group: 50-74':3764, 'Age Group: 25-49':1497, 'Age Group: 75-99':4758, 'Current':958, 'Former':1548, 'Never':3742, \
       #'Asian':726, 'Black':302, 'Multiple':72, 'Other':160, 'White':6737, 'Unknown':0}
       n = len(self.data)
       if self.test == 'nccid' or self.test =='covidgr' or self.test == 'ltht_pneumonia' or self.test == 'ltht_no_pneumonia':
           pos_cases = len(self.data[self.data['xray_status']==1.0])
       elif self.test == 'ltht':
           pos_cases = len(self.data[self.data['FinalPCR']==1.0])
       prevalence = pos_cases/n

       assert prevalence < 1
       return prevalence
    
    def get_best_threshold_idx(self, k, group):
#       best_threshold = self.thresholds['best_threshold'][k]
 #      print(best_threshold)

#       print(roc_thresholds)
      # roc_thresholds = self.results['All']['roc_thresholds'][k]
       print(self.results[group].keys())
       roc_thresholds = self.results[group]['roc_thresholds'][k]

 #      print(roc_thresholds)
       best_threshold = 0.5 #np.around(best_threshold,3)
 #      print(best_threshold)
       roc_thresholds = np.around(roc_thresholds,1)

     #  print(roc_thresholds)
       best_threshold_idx = np.where(roc_thresholds == best_threshold)
#       print(best_threshold_idx[0][0],'b')
       return best_threshold_idx[0][0]

    def get_tpr(self, best_threshold_idx, k,g):
       tprs = self.results[g]['tpr'][k]
       return tprs[best_threshold_idx]

    def get_fpr(self, best_threshold_idx, k,g):
       print('all fprs',self.results[g]['fpr'])
       fprs = self.results[g]['fpr'][k]
       return fprs[best_threshold_idx]

    def get_recall(self, k):
       recall = self.results['All']['recall'][k]
       return recall

    def calc_ppv(self,k):
       best_threshold_idx = self.get_best_threshold_idx(k)
       fpr = self.get_fpr(best_threshold_idx, k)
       #fpr = self.results['All']['fpr']
       #print(fpr)
       tnr = 1. - fpr

       tpr = self.get_tpr(best_threshold_idx, k)
       recall = self.get_recall(k)
       ppv = (recall * self.prevalence) / ((recall * self.prevalence) + ((1. - tnr) * (1. - self.prevalence)))
       return ppv
   
    def write_to_results(self):
       ppv_list = []

       for i in range(5):

#          print(i)
          ppv_list.append(self.calc_ppv(i))
#          print(i)

       self.results['All']['ppv'] = ppv_list
       #print(self.results)
       print(f"saving to {self.results_path} ...")
       with open(self.results_path, 'w') as f:
            json.dump(self.results, f)
    # save when sure
    
# give list of data
def plot_fpr_tpr(data, test, results, popsize_dict, fig, ax, iter, group, g, color_idx):
    ppv = PPV(data, test, results)
    plt.rcParams['axes.prop_cycle'] = cycler('color', plt.get_cmap('tab20').colors)

    color_cycle = plt.get_cmap('tab20').colors
    fpr_list = []
    tpr_list = []
    for k in range(5):
        best_idx = ppv.get_best_threshold_idx(k, g)
        fpr_list.append(ppv.get_fpr(best_idx,k,g))
        tpr_list.append(ppv.get_tpr(best_idx,k,g))
    
 #   print('group', g)
    mean_tpr = np.mean(tpr_list)
    std_tpr = np.std(tpr_list)
    mean_fpr = np.mean(fpr_list)
    std_fpr = np.std(fpr_list)

    width = 0.55  # the width of the bars
    multiplier = group

    x = iter
    n = popsize_dict[g]
    rects = ax[0].barh(x * width+group, 1-mean_tpr, width, xerr=std_tpr, capsize=2, align='center', color=color_cycle[color_idx])
  #  print(tpr_list,'tl')


    rects = ax[1].barh(x * width+group, mean_fpr, width, xerr=std_fpr, capsize=2, align='center', color=color_cycle[color_idx])

    format_map = {"All":"All","F":"Female", "M":"Male","Chest X-ray Comorbidity":"Yes", "No Chest X-ray Comorbidity":"No","Age Group: 100-125":"100+ years"," Age Group: 0-24": "0-24 years", "Age Group: 50-74":"50-74 years", 'Age Group: 75-99':"75-99 years", 'Age Group: 25-49':'25-49 years'}

    if g in format_map.keys():
        subpop = format_map[g]
    else:
        subpop = g
    ax[1].bar_label(rects, labels=[r'\hspace{{25cm}} \textbf{{{}}}: n = {}'.format(subpop,n)], fontsize=6.2, padding=3)
    
    return fig, ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPV RECOVERY')
    parser.add_argument('--data_csv')
    parser.add_argument('--results_csv', default='/MULTIX/DATA/HOME/covid-19-benchmarking/data/nccid_preprocessed14_21.csv', type=str, help='Path to data file')
    parser.add_argument('--test', default=None, type=str, help='Choose on of: [ltht, nccid_test, nccid_val, nccid_leeds, chexpert, custom]')
    parser.add_argument('--threshold_data', type=str, help='Path to threshold data, generated by nccid_test i.e. root/model_name_performance_nccid_test_df.csv')
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

    elif args.test=='covidgr':
        data = df

  #  calc_ppv = PPV(data, args.test, args.results_csv)
 #   calc_ppv.write_to_results()

    barplot = True
    popsize_dict = {'F':1246, 'M':1658,' Age Group: 0-24':1076,'Age Group: 100-125': 109, \
        'Age Group: 50-74':3764, 'Age Group: 25-49':1497, 'Age Group: 75-99':4758, 'Current':958, 'Former':1548, 'Never':3742, \
        'Asian':726, 'Black':302, 'Multiple':72, 'Other':160, 'White':6737, 'No Chest X-ray Comorbidity': 1412, 'Chest X-ray Comorbidity':1214}
    gender_map = {'F':'Female','M':'Male'}

    ltht_pneum = {'No Chest X-ray Comorbidity': 929, 'Chest X-ray Comorbidity': 711, 'Asian': 267, 'Black': 147, 'Multiple': 28, 'Other': 42, 'Unknown': 170, 'White': 2312, 'Current': 298, 'Former': 615, 'Never': 1370, ' Age Group: 0-24': 176, 'Age Group: 100-125': 49, 'Age Group: 25-49': 386, 'Age Group: 50-74': 1373, 'Age Group: 75-99': 1964, 'F': 1691, 'M': 2256}
    ltht_no_pneum = {'No Chest X-ray Comorbidity': 683, 'Chest X-ray Comorbidity': 477, 'Asian': 141, 'Black': 110, 'Multiple': 21, 'Other': 29, 'Unknown': 105, 'White': 959, 'Current': 42, 'Former': 143, 'Never': 551, ' Age Group: 0-24': 77, 'Age Group: 25-49': 116, 'Age Group: 50-74': 633, 'Age Group: 75-99': 615, 'F': 532, 'M': 919}
    ltht = {'No Chest X-ray Comorbidity': 1412, 'Chest X-ray Comorbidity': 1214, 'Asian': 726, 'Black': 302, 'Multiple': 72, 'Other': 160, 'Unknown': 464, 'White': 6737, 'Current': 958, 'Former': 1548, 'Never': 3742, ' Age Group: 0-24': 1076, 'Age Group: 100-125': 109, 'Age Group: 25-49': 1497, 'Age Group: 50-74': 3764, 'Age Group: 75-99': 4758, 'F': 5108, 'M': 6095}

    if args.test == 'ltht':
        popsize_dict =  ltht
    if args.test == 'ltht_pneumonia':
        popsize_dict = ltht_pneum
    if args.test == 'ltht_no_pneumonia':
        popsize_dict = ltht_no_pneum

    def barplot_fn(model, popsize_dict, comp=False):
        results_list = glob.glob(f"/MULTIX/DATA/HOME/covid-19-benchmarking/results/{model}/{args.test}/*.json")
        results_list = [r for r in results_list if "ppv" not in r]
        results_list = [r for r in results_list if 'All' not in r]      
#        results_list = [r for r in results_list if 'Comorbidity' not in r]
        fig1, ax1 = plt.subplots(figsize=(6,4), sharey=True, ncols=2) 

        iter = 0
        if args.test == 'ltht':
            if comp == True:
                results_list = [r for r in results_list if 'Comorbidity' not in r]
                group_names = ["Age Group","Ethnicity", "Sex", "Smoker"]
                y_ticks = [0.8,3.7,6.1,8.4]
            else:
                group_names = ["Age Group","Comorbidity","Ethnicity", "Sex", "Smoker"]
                y_ticks = [1.0, 4, 7., 10, 12.25]
        else:
            results_list = [r for r in results_list if 'Comorbidity' not in r]
            group_names = ["Age Group","Ethnicity", "Sex", "Smoker"]
            y_ticks = [0.8,3.7,6.1,8.4]           
        
        for n, res in enumerate(results_list):

            with open(res) as f:
                results_data = json.load(f)
                if 'Age' in res:
                    sorted_results_data = sorted(results_data.keys(), key=lambda x: int(x.split('-')[-1]))
                    new_results_data = {}
                    for i in sorted_results_data:
                        new_results_data[i] = results_data[i]
                    results_data = new_results_data
                if 'Unknown' in results_data: del results_data['Unknown']
                if args.test == 'ltht_no_pneumonia':
                    print(args.test)
                    if 'Age Group: 100-125' in results_data: del results_data['Age Group: 100-125']
                    if 'No Chest X-ray Comorbidity' in results_data: del results_data['No Chest X-ray Comorbidity']
                    if 'Chest X-ray Comorbidity' in results_data: del results_data['Chest X-ray Comorbidity']
                    if 'Black' in results_data: del results_data['Black']
                    if 'Other' in results_data: del results_data['Other']
                if args.test == 'ltht':
                    if comp == True:
                        if 'Age Group: 100-125' in results_data: del results_data['Age Group: 100-125']
                        if 'No Chest X-ray Comorbidity' in results_data: del results_data['No Chest X-ray Comorbidity']
                        if 'Chest X-ray Comorbidity' in results_data: del results_data['Chest X-ray Comorbidity']
                        if 'Black' in results_data: del results_data['Black']
                        if 'Other' in results_data: del results_data['Other']
            ax1[0].set_yticks(y_ticks, group_names, rotation=45)   
            #ax1.set_xlim(0.5, 1.0)
            ax1[0].set_xlabel('False Negative Rate')
            ax1[1].set_xlabel('False Positive Rate')
            ax1[0].yaxis.tick_left()
            ax1[1].set_xlim(0,1.)
            ax1[0].set_xlim(0,1.)
            ax1[1].yaxis.tick_right()

            for idx, i in enumerate(results_data.keys()):
                    fig1, ax1 = plot_fpr_tpr(res, 'ltht', results_data, popsize_dict,fig=fig1, ax=ax1, iter=iter, group=n, g=i, color_idx=idx)
                    iter +=1

            ax1[0].invert_xaxis()

            plt.subplots_adjust(wspace=0.0, top=0.85, bottom=0.1, left=0.18, right=0.95)
        
        if comp == True:
            filename = f'{model}_subpop_fnr_fpr_bar_comp_new_{args.test}.pdf'
        else:
            filename = f'{model}_subpop_fnr_fpr_bar_new_{args.test}.pdf'

        fig1.savefig(f'/MULTIX/DATA/HOME/covid-19-benchmarking/results/{model}/{args.test}/{filename}')

    if barplot == True:
        for args.test in ['ltht_no_pneumonia']:
            for c in [False, True]:
                for model in ['xvitcos', 'ssl_am','coronet_tfl']:
                    print(popsize_dict)
#                    break
                    barplot_fn(model, popsize_dict, c)
