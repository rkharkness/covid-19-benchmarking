import json

import matplotlib.pyplot as plt 
import matplotlib.markers as markers

import numpy as np 
import glob 
from scipy import stats 
from sklearn.metrics import auc
import scienceplots

ltht_pneum = {'No Chest X-ray Comorbidity': 929, 'Chest X-ray Comorbidity': 711, 'Asian': 267, 'Black': 147, 'Multiple': 28, 'Other': 42, 'Unknown': 170, 'White': 2312, 'Current': 298, 'Former': 615, 'Never': 1370, ' Age Group: 0-24': 176, 'Age Group: 100-125': 49, 'Age Group: 25-49': 386, 'Age Group: 50-74': 1373, 'Age Group: 75-99': 1964, 'F': 1691, 'M': 2256}
ltht_no_pneum = {'No Chest X-ray Comorbidity': 683, 'Chest X-ray Comorbidity': 477, 'Asian': 141, 'Black': 110, 'Multiple': 21, 'Other': 29, 'Unknown': 105, 'White': 959, 'Current': 42, 'Former': 143, 'Never': 551, ' Age Group: 0-24': 77, 'Age Group: 25-49': 116, 'Age Group: 50-74': 633, 'Age Group: 75-99': 615, 'F': 532, 'M': 919}
ltht = {'No Chest X-ray Comorbidity': 1412, 'Chest X-ray Comorbidity': 1214, 'Asian': 726, 'Black': 302, 'Multiple': 72, 'Other': 160, 'Unknown': 464, 'White': 6737, 'Current': 958, 'Former': 1548, 'Never': 3742, ' Age Group: 0-24': 1076, 'Age Group: 100-125': 109, 'Age Group: 25-49': 1497, 'Age Group: 50-74': 3764, 'Age Group: 75-99': 4758, 'F': 5108, 'M': 6095}

def ci(data):
    # mean  +/- t*s/sqrt n
    return stats.t.interval(alpha=0.95, df = len(data)-1, loc= np.mean(data), scale=stats.sem(data))


def model_prs(model_name, dataset_name):
        pop = ['Smoker', 'Gender','Ethnicity','Age','Comorbidity']
        model_name_map = {'CAPSNET':'CAPSNET','XVITCOS_SEG':'XVITCOS (ROI)','SSL_AM_SEG':'SSL-AM (ROI)','CORONET_TFL_SEG':'XCEPTION NET (ROI)', 'FUSENET':'FUSENET','MAG_SD':'MAG-SD', 'CORONET':'CORONET','ECOVNET':'ECOVNET','XVITCOS':'XVITCOS','CORONET_TFL':'XCEPTION NET','RES_ATTN':'RES. ATTN.', 'COVIDNET':'COVIDNET', 'SSL_AM':'SSL-AM'}             
        dataset_name_map = {'ltht_pneumonia':'LTHT (P)', 'ltht':'LTHT', 'ltht_no_pneumonia':'LTHT (NP)'}
        subpop_key_map = {"F":"Female","M":"Male"}
       # model_name = model_name_map[model_name.upper()]
        dataset_name_label = dataset_name_map[dataset_name]
        if dataset_name == 'ltht': popsize_dict = ltht
        if dataset_name == 'ltht_pneumonia': popsize_dict = ltht_pneum
        if dataset_name == 'ltht_no_pneumonia': popsize_dict = ltht_no_pneum

        with plt.style.context(['science', 'nature']):
            f, ax = plt.subplots(1, 5, figsize=(14,3.2), gridspec_kw={'width_ratios': [1.3, 1.3, 1.3, 1.3, 1.3]})
#            counter = 0
            for p, a in zip(pop, ax):
                colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

                with open(f'/MULTIX/DATA/HOME/covid-19-benchmarking/results/{model_name}/{dataset_name}/{model_name}_performance_{dataset_name}_{p}.json') as results_file:
                    r = json.load(results_file)
                    if dataset_name == 'ltht_pneumonia':
                        if 'Unknown' in r: del r['Unknown']
                        if 'Age Group: 100-125' in r: del r['Age Group: 100-125']
                        if 'No Chest X-ray Comorbidity' in r: del r['No Chest X-ray Comorbidity']
                        if 'Chest X-ray Comorbidity' in r: del r['Chest X-ray Comorbidity']
                        if 'Black' in r: del r['Black']
                        if 'Other' in r: del r['Other']
                keys = r.keys()
                counter = 0
                for k in keys:
                    r_k = r[k] 
                        
                    mean_x = np.linspace(0, 1, 100)
                    x = r_k['recall_curve']
                    aucs = r_k['prec_auc']
                    y = r_k['interp_rec']
                    
                    mean_y = np.mean(y, axis=0)
                    mean_auc = auc(mean_x, mean_y)
                    std_auc = np.std(aucs)
                    ci = 1.96 * np.std(y) / np.sqrt(len(y)) 
 
                    tprs_upper = np.minimum(mean_y + ci, 1)
                    tprs_lower = np.maximum(mean_y - ci, 0)

                    n = popsize_dict[k]

                    legend = None
                    if p == "Gender":
                        k = subpop_key_map[k]

                    if p == "Age":
                        print(p,'p')
                        k = k[11:] + " years"

                    if str(mean_auc) != 'nan':
                        legend = '%s (n=%s): \n AUC = %0.2f $\pm$ %0.2f' % (k, n, mean_auc, std_auc)           

                    if k!='Unknown':
                        a.plot(mean_x, mean_y, lw=1, alpha=.8, color=colors[counter], label=legend)
                        a.fill_between(mean_x, tprs_lower, tprs_upper,alpha=.1, color=colors[counter])
                        a.set_title(r'\textbf{{{}}}'.format(p), fontsize=10, weight='bold', fontdict=dict(weight='bold'), wrap=True)
                        counter +=1
                        if str(mean_auc) != 'nan':
                            print('nan')
                            a.legend()

            ax[0].set_yticklabels([])
            ax[0].set_ylabel('True Positive Rate', fontsize=10)
            ax[1].set_yticklabels([])
            ax[2].set_yticklabels([])
            ax[3].set_yticklabels([])
            ax[4].set_yticklabels([])    
            #0.513
            f.text(0.405, -0.005, 'False Positive Rate', ha='center', fontsize=10)
      
            f.tight_layout()
            f.savefig(f"/MULTIX/DATA/HOME/covid-19-benchmarking/results/{model_name}_{dataset_name}_subpop_pr_new.pdf",dpi=300, bbox_inches='tight')   

def model_rocs(model_name, dataset_name):
        pop = ['Smoker', 'Gender','Ethnicity','Age','Comorbidity']
        model_name_map = {'CAPSNET':'CAPSNET','XVITCOS_SEG':'XVITCOS (ROI)','SSL_AM_SEG':'SSL-AM (ROI)','CORONET_TFL_SEG':'XCEPTION NET (ROI)', 'FUSENET':'FUSENET','MAG_SD':'MAG-SD', 'CORONET':'CORONET','ECOVNET':'ECOVNET','XVITCOS':'XVITCOS','CORONET_TFL':'XCEPTION NET','RES_ATTN':'RES. ATTN.', 'COVIDNET':'COVIDNET', 'SSL_AM':'SSL-AM'}             
        dataset_name_map = {'ltht_pneumonia':'LTHT (P)', 'ltht':'LTHT', 'ltht_no_pneumonia':'LTHT (NP)'}
        subpop_key_map = {"F":"Female","M":"Male"}
       # model_name = model_name_map[model_name.upper()]
        dataset_name_label = dataset_name_map[dataset_name]
        if dataset_name == 'ltht': popsize_dict = ltht
        if dataset_name == 'ltht_pneumonia': popsize_dict = ltht_pneum
        if dataset_name == 'ltht_no_pneumonia': popsize_dict = ltht_no_pneum
        with plt.style.context(['science', 'nature']):
            f, ax = plt.subplots(1, 5, figsize=(14,3.2), gridspec_kw={'width_ratios': [1.3, 1.3, 1.3, 1.3, 1.3]})
#            counter = 0
            for p, a in zip(pop, ax):
                colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

                with open(f'/MULTIX/DATA/HOME/covid-19-benchmarking/results/{model_name}/{dataset_name}/{model_name}_performance_{dataset_name}_{p}.json') as results_file:
                    r = json.load(results_file)
                    if dataset_name == 'ltht_pneumonia' or dataset_name == 'ltht' or 'ltht_no_pneumonia':
                        if 'Unknown' in r: del r['Unknown']
                        if 'Age Group: 100-125' in r: del r['Age Group: 100-125']
                        if 'No Chest X-ray Comorbidity' in r: del r['No Chest X-ray Comorbidity']
                        if 'Chest X-ray Comorbidity' in r: del r['Chest X-ray Comorbidity']
                        if 'Black' in r: del r['Black']
                        if 'Other' in r: del r['Other']

                keys = r.keys()
                counter = 0
                for k in keys:
                    r_k = r[k] 
                        
                    mean_x = np.linspace(0, 1, 100)
                    x = r_k['fpr']
                    aucs = r_k['roc_auc']
                    y = r_k['interp_tpr']
            
                    mean_y = np.mean(y, axis=0)
                    mean_auc = auc(mean_x, mean_y)
                    std_auc = np.std(aucs)
                    ci = 1.96 * np.std(y) / np.sqrt(len(y)) 
 
                    tprs_upper = np.minimum(mean_y + ci, 1)
                    tprs_lower = np.maximum(mean_y - ci, 0)

                    n = popsize_dict[k]
                    legend = None

                    if p == "Gender":
                        k = subpop_key_map[k]
                    if p == "Age":
                        k = k[11:] + " years"
                    if str(mean_auc) != 'nan':
                        legend = '%s (n=%s): \n AUC = %0.2f $\pm$ %0.2f' % (k, n, mean_auc, std_auc)          
                    if k!='Unknown':
                        a.plot(mean_x, mean_y, lw=1, alpha=.8, color=colors[counter], label=legend)
                        a.fill_between(mean_x, tprs_lower, tprs_upper,alpha=.1, color=colors[counter])
                        a.set_title(r'\textbf{{{}}}'.format(p), fontsize=10, weight='bold', fontdict=dict(weight='bold'), wrap=True)
                        counter +=1
                        if str(mean_auc) != 'nan':
                            print('nan')
                            a.legend(loc='lower right')

           # ax[0].set_yticklabels([])
            ax[0].set_ylabel('True Positive Rate', fontsize=10)
            ax[1].set_yticklabels([])
            ax[2].set_yticklabels([])
            ax[3].set_yticklabels([])
            ax[4].set_yticklabels([])
        
            f.text(0.405, -0.005, 'False Positive Rate', ha='center', fontsize=10)     
                
            f.tight_layout()
            f.savefig(f"/MULTIX/DATA/HOME/covid-19-benchmarking/results/{model_name}_{dataset_name}_subpop_roc_comp_new2.pdf",dpi=300, bbox_inches='tight')   

data =  ['ltht', 'ltht_no_pneumonia','ltht_pneumonia']

for i in data:
    model_prs('coronet_tfl',i)
    model_rocs('coronet_tfl',i)
 #   model_prs('coronet_tfl_seg', i)
#    model_rocs('coronet_tfl_seg', i)
    model_prs('xvitcos', i)
    model_rocs('xvitcos',i)
  #  model_prs('xvitcos_seg',i)
  #  model_rocs('xvitcos_seg',i)
    model_prs('ssl_am' ,i)
    model_rocs('ssl_am',i)
  #  model_prs('ssl_am_seg', i)
  #  model_rocs('ssl_am_seg', i)
