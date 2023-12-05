import json

import matplotlib.pyplot as plt 
import matplotlib.markers as markers

import numpy as np 
import glob 
from scipy import stats 
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score

def ci(data):
    # mean  +/- t*s/sqrt n
    return stats.t.interval(alpha=0.95, df = len(data)-1, loc= np.mean(data), scale=stats.sem(data))

def model_prs(model_name):
    results = glob.glob(f'/MULTIX/DATA/HOME/covid-19-benchmarking/results/{model_name}/*/*All.json')
    results = [i for i in results if i.split('/')[7] not in ['nccid_val','covid_qu_ex']]
    results = [i for i in results if i.split('/')[6] not in ['xvitcos_seg','ssl_am_seg','coronet_tfl_seg']]   
    model_name_map = {'CAPSNET':'CAPSNET','XVITCOS_SEG':'XVITCOS (ROI)','SSL_AM_SEG':'SSL-AM (ROI)','CORONET_TFL_SEG':'XCEPTION NET (ROI)', 'FUSENET':'FUSENET','MAG_SD':'MAG-SD', 'CORONET':'CORONET','ECOVNET':'ECOVNET','XVITCOS':'XVITCOS','CORONET_TFL':'XCEPTION NET','RES_ATTN':'RES. ATTN.', 'COVIDNET':'COVIDNET', 'SSL_AM':'SSL-AM'}             
    dataset_name = {'nccid_val':'NCCID Val','covidgr':'COVIDGR', 'nccid_test':'NCCID TEST','ltht_pneumonia':'LTHT (P)', 'ltht':'LTHT', 'ltht_no_pneumonia':'LTHT (NP)', 'covid_qu_ex':'COVID-QU-Ex'}
    
    test = 'All'
    
    data = [i.split('/')[7] for i in results]
    data = list(set(data))
    data = sorted(data) #, reverse=True)
    data = ['nccid_test','ltht','ltht_no_pneumonia','ltht_pneumonia','covidgr']
#    data = [None] + data
    with plt.style.context(['science', 'nature']):
        f, ax = plt.subplots(1, 5, figsize=(14,3.2), gridspec_kw={'width_ratios': [1.3, 1.3, 1.3, 1.3, 1.3]})
        counter = 0
        for d, a in zip(data, ax):
          with open(f'/MULTIX/DATA/HOME/covid-19-benchmarking/results/{model_name}/{d}/{model_name}_performance_{d}_All.json') as results_file:
            r = json.load(results_file)
            r = r[test]
            
            data_name = dataset_name[d]
            
            mean_x = np.linspace(0, 1, 100)
            x = r['recall_curve']
            aucs = r['prec_auc']
            y = r['interp_rec']
            
            mean_y = np.mean(y, axis=0)
        #    y_ci = ci(y)
            #print(y_ci)
       #     lower_ci = np.nan_to_num(y_ci[0])
       #     upper_ci = np.nan_to_num(y_ci[1])
   #         print(lower_ci)
            mean_auc = auc(mean_x, mean_y)
            std_auc = np.std(aucs)
            ci = 1.96 * np.std(y) / np.sqrt(len(y)) 
#            print(confidence_interval, 'ci')
            tprs_upper = np.minimum(mean_y + ci, 1)
            tprs_lower = np.maximum(mean_y - ci, 0)
 #           tprs_upper = upper_ci
#            tprs_lower = lower_ci
            print(d)
            if d == 'nccid_test':
              ax[1].plot(mean_x, mean_y, lw=1, alpha=.8)
              ax[1].fill_between(mean_x, tprs_lower, tprs_upper,alpha=.1)
              ax[2].plot(mean_x, mean_y, lw=1, alpha=.8)
              ax[2].fill_between(mean_x, tprs_lower, tprs_upper,alpha=.1)
              ax[3].plot(mean_x, mean_y, lw=1, alpha=.8)
              ax[3].fill_between(mean_x, tprs_lower, tprs_upper,alpha=.1)
              ax[4].plot(mean_x, mean_y, lw=1, alpha=.8)
              ax[4].fill_between(mean_x, tprs_lower, tprs_upper,alpha=.1)
              
            ax[0].plot(mean_x, mean_y, label=data_name, lw=1, alpha=.8)   #'%s: \n AUC = %0.2f $\pm$ %0.2f' % (data_name, mean_auc, std_auc),
            ax[0].fill_between(mean_x, tprs_lower, tprs_upper,alpha=.1)
            ax[0].set_title(r'\textbf{{{}}}'.format(model_name_map[model_name.upper()]), rotation='vertical', x=-.25,y=0.34, fontsize=12, fontdict=dict(weight='bold'), wrap=True) #,fontsize=8)
            ax[0].set_xlabel('Recall',fontsize=10)
            ax[0].set_ylabel('Precision',fontsize=10)
            #ax[0].legend(fontsize=5, loc='lower right')
            
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            
            a.plot(mean_x, mean_y, lw=1, alpha=.8, color=colors[counter])
            a.fill_between(mean_x, tprs_lower, tprs_upper,alpha=.1, color=colors[counter])
            a.set_title(r'\textbf{{{}}}'.format(data_name), fontsize=10, weight='bold', fontdict=dict(weight='bold'), wrap=True)
       #     a.set_xlabel('False Positive Rate',fontsize=7.5)

            ax[1].set_yticklabels([])
            ax[2].set_yticklabels([])
            ax[3].set_yticklabels([])
            ax[4].set_yticklabels([])

         #   f.xlabel('False Positive Rate', fontsize=7.5)
            counter = counter + 1
            handles, labels = ax[0].get_legend_handles_labels() 
        

        f.text(0.513, 0.039, 'Recall', ha='center', fontsize=10)
        f.legend(handles, labels, bbox_to_anchor=(0.715, 0.00), ncol=5, fontsize=9)
#            a.set_ylabel('True Positive Rate') #,fontsize=7)            
            
        f.tight_layout()
        f.savefig(f"/MULTIX/DATA/HOME/covid-19-benchmarking/results/{model_name}_pr_new.pdf",dpi=300, bbox_inches='tight')   

def model_rocs(model_name):
    results = glob.glob(f'/MULTIX/DATA/HOME/covid-19-benchmarking/results/{model_name}/*/*All.json')
    results = [i for i in results if i.split('/')[7] not in ['nccid_val','covid_qu_ex']]
    results = [i for i in results if i.split('/')[6] not in ['xvitcos_seg','ssl_am_seg','coronet_tfl_seg']]   
    model_name_map = {'CAPSNET':'CAPSNET','XVITCOS_SEG':'XVITCOS (ROI)','SSL_AM_SEG':'SSL-AM (ROI)','CORONET_TFL_SEG':'XCEPTION NET (ROI)', 'FUSENET':'FUSENET','MAG_SD':'MAG-SD', 'CORONET':'CORONET','ECOVNET':'ECOVNET','XVITCOS':'XVITCOS','CORONET_TFL':'XCEPTION NET','RES_ATTN':'RES. ATTN.', 'COVIDNET':'COVIDNET', 'SSL_AM':'SSL-AM'}             
    dataset_name = {'nccid_val':'NCCID Val','covidgr':'COVIDGR', 'nccid_test':'NCCID TEST','ltht_pneumonia':'LTHT (P)', 'ltht':'LTHT', 'ltht_no_pneumonia':'LTHT (NP)', 'covid_qu_ex':'COVID-QU-Ex'}
    
    test = 'All'
    
    data = [i.split('/')[7] for i in results]
    data = list(set(data))
    data = sorted(data) #, reverse=True)
    data = ['nccid_test','ltht','ltht_no_pneumonia','ltht_pneumonia','covidgr']
#    data = [None] + data
    with plt.style.context(['science', 'nature']):
        f, ax = plt.subplots(1, 5, figsize=(14,3.2), gridspec_kw={'width_ratios': [1.3, 1.3, 1.3, 1.3, 1.3]})
        counter = 0
        for d, a in zip(data, ax):
          with open(f'/MULTIX/DATA/HOME/covid-19-benchmarking/results/{model_name}/{d}/{model_name}_performance_{d}_All.json') as results_file:
            r = json.load(results_file)
            r = r[test]
            
            data_name = dataset_name[d]
            
            mean_x = np.linspace(0, 1, 100)
            x = r['fpr']
            aucs = r['roc_auc']
            y = r['interp_tpr']
            
            mean_y = np.mean(y, axis=0)
        #    y_ci = ci(y)
            #print(y_ci)
       #     lower_ci = np.nan_to_num(y_ci[0])
       #     upper_ci = np.nan_to_num(y_ci[1])
   #         print(lower_ci)
            mean_auc = auc(mean_x, mean_y)
            std_auc = np.std(aucs)
            ci = 1.96 * np.std(y) / np.sqrt(len(y)) 
#            print(confidence_interval, 'ci')
            tprs_upper = np.minimum(mean_y + ci, 1)
            tprs_lower = np.maximum(mean_y - ci, 0)
 #           tprs_upper = upper_ci
#            tprs_lower = lower_ci
            print(d)
            if d == 'nccid_test':
              ax[1].plot(mean_x, mean_y, lw=1, alpha=.8)
              ax[1].fill_between(mean_x, tprs_lower, tprs_upper,alpha=.1)
              ax[2].plot(mean_x, mean_y, lw=1, alpha=.8)
              ax[2].fill_between(mean_x, tprs_lower, tprs_upper,alpha=.1)
              ax[3].plot(mean_x, mean_y, lw=1, alpha=.8)
              ax[3].fill_between(mean_x, tprs_lower, tprs_upper,alpha=.1)
              ax[4].plot(mean_x, mean_y, lw=1, alpha=.8)
              ax[4].fill_between(mean_x, tprs_lower, tprs_upper,alpha=.1)
              
            ax[0].plot(mean_x, mean_y, label=data_name, lw=1, alpha=.8)   #'%s: \n AUC = %0.2f $\pm$ %0.2f' % (data_name, mean_auc, std_auc),
            ax[0].fill_between(mean_x, tprs_lower, tprs_upper,alpha=.1)
            ax[0].set_title(r'\textbf{{{}}}'.format(model_name_map[model_name.upper()]), rotation='vertical', x=-.25,y=0.34, fontsize=12, fontdict=dict(weight='bold'), wrap=True) #,fontsize=8)
            ax[0].set_xlabel('False Positive Rate',fontsize=10)
            ax[0].set_ylabel('True Positive Rate',fontsize=10)
            #ax[0].legend(fontsize=5, loc='lower right')
            
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            
            a.plot(mean_x, mean_y, lw=1, alpha=.8, color=colors[counter])
            a.fill_between(mean_x, tprs_lower, tprs_upper,alpha=.1, color=colors[counter])
            a.set_title(r'\textbf{{{}}}'.format(data_name), fontsize=10, weight='bold', fontdict=dict(weight='bold'), wrap=True)
       #     a.set_xlabel('False Positive Rate',fontsize=7.5)

            ax[1].set_yticklabels([])
            ax[2].set_yticklabels([])
            ax[3].set_yticklabels([])
            ax[4].set_yticklabels([])

         #   f.xlabel('False Positive Rate', fontsize=7.5)
            counter = counter + 1
            handles, labels = ax[0].get_legend_handles_labels() 
        

        f.text(0.513, 0.039, 'False Positive Rate', ha='center', fontsize=10)
        f.legend(handles, labels, bbox_to_anchor=(0.715, 0.00), ncol=5, fontsize=9)
#            a.set_ylabel('True Positive Rate') #,fontsize=7)            
            
        f.tight_layout()
        f.savefig(f"/MULTIX/DATA/HOME/covid-19-benchmarking/results/{model_name}_roc_new.pdf",dpi=300, bbox_inches='tight')   

def data_rocs(data_names=['nccid_test','ltht','ltht_no_pneumonia','ltht_pneumonia','covidgr'], model_names=[['xvitcos', 'xvitcos_seg']]):
#    results = glob.glob(f'/MULTIX/DATA/HOME/covid-19-benchmarking/results/*/{data_name}/*All.json')
#    results = [i for i in results if i.split('/')[7] in data_names]

    model_name_map = {'XVITCOS':'XVITCOS','SSL_AM':'SSL_AM','CORONET_TFL':'XCEPTION NET','XVITCOS_SEG':'XVITCOS (ROI)','SSL_AM_SEG':'SSL-AM (ROI)','CORONET_TFL_SEG':'XCEPTION NET (ROI)', 'FUSENET':'FUSENET'}
    dataset_name = {'nccid_val':'NCCID Val','covidgr':'COVIDGR', 'nccid_test':'NCCID TEST','ltht_pneumonia':'LTHT (P)', 'ltht':'LTHT', 'ltht_no_pneumonia':'LTHT (NP)'}
    print(model_names)    
    test = 'All'
    markers = [None, '^']
  #  if seg_eval == True:
#    model_names = [['xvitcos', 'xvitcos_seg']] #,['ssl_am','ssl_am_seg'],['coronet_tfl','coronet_tfl']]
#    data = ['nccid_test','ltht','ltht_no_pneumonia','ltht_pneumonia','covidgr']

    with plt.style.context(['science', 'nature']):
        f, ax = plt.subplots(1, 5, figsize=(14,3.2), gridspec_kw={'width_ratios': [1.3, 1.3, 1.3, 1.3, 1.3]})
        #for model in model_names:
        counter = 0
        for i in range(2):
##            counter= 0
            for d, a in zip(data_names, ax):
                with open(f'/MULTIX/DATA/HOME/covid-19-benchmarking/results/{model_names[i]}/{d}/{model_names[i]}_performance_{d}_All.json') as results_file:
                    r = json.load(results_file)
                    r = r[test]
            
                    data_name = dataset_name[d]
            
                    mean_x = np.linspace(0, 1, 100)
                    x = r['fpr']
                    aucs = r['roc_auc']
                    y = r['interp_tpr']
            
                    mean_y = np.mean(y, axis=0)
     
                    mean_auc = auc(mean_x, mean_y)
                    std_auc = np.std(aucs)
                    ci = 1.96 * np.std(y) / np.sqrt(len(y)) 

                    tprs_upper = np.minimum(mean_y + ci, 1)
                    tprs_lower = np.maximum(mean_y - ci, 0)
                
            #    if d == 'nccid_test':
             #       ax[1].plot(mean_x, mean_y, lw=1, alpha=.8, marker=markers[i], markevery=10)
              #      ax[1].fill_between(mean_x, tprs_lower, tprs_upper,alpha=.1)
               #    ax[2].plot(mean_x, mean_y, lw=1, alpha=.8,  marker=markers[i], markevery=10)
                #    ax[2].fill_between(mean_x, tprs_lower, tprs_upper,alpha=.1)
#                    ax[3].plot(mean_x, mean_y, lw=1, alpha=.8, marker=markers[i])
 #                   ax[3].fill_between(mean_x, tprs_lower, tprs_upper,alpha=.1)
        #            ax[4].plot(mean_x, mean_y, lw=1, alpha=.8)
       #             ax[4].fill_between(mean_x, tprs_lower, tprs_upper,alpha=.1)
              
                #ax[0].plot(mean_x, mean_y, label=data_name, lw=1, alpha=.8, marker=markers[i], markevery=10)                
                #ax[0].fill_between(mean_x, tprs_lower, tprs_upper,alpha=.1)
                #ax[0].set_title(r'\textbf{{{}}}'.format(model_name_map[model[i].upper()]), rotation='vertical', x=-.3,y=0.35, fontsize=8, fontdict=dict(weight='bold'))
                #ax[0].set_xlabel('False Positive Rate',fontsize=7.5)
                ax[0].set_ylabel('True Positive Rate',fontsize=10)
#                ax[0].set_xlabel('False Positive Rate', fontsize=7.5)
 #               ax[1].set_xlabel('False Positive Rate', fontsize=7.5)  
                ax[2].set_xlabel('False Positive Rate', fontsize=10)  
  #              ax[3].set_xlabel('False Positive Rate', fontsize=7.5)  
   #             ax[4].set_xlabel('False Positive Rate', fontsize=7.5)  
  
                colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            
                a.plot(mean_x, mean_y, lw=1, alpha=.8, color=colors[counter], marker=markers[i], markevery=10, label=model_name_map[model_names[i].upper()])
                a.fill_between(mean_x, tprs_lower, tprs_upper,alpha=.1, color=colors[counter])
                a.set_title(r'\textbf{{{}}}'.format(data_name), fontsize=10, weight='bold', fontdict=dict(weight='bold'), wrap=True)

  #              ax[0].set_yticklabels([])
                ax[1].set_yticklabels([])
                ax[2].set_yticklabels([])
                ax[3].set_yticklabels([])
                ax[4].set_yticklabels([])

  
            counter = counter + 1
            handles, labels = ax[0].get_legend_handles_labels() 
        
#        plt.title(r'\textbf{{{}}}'.format(model_name_map[model_names[i].upper()]), fontsize=8)
        #f.text(0.4, 0.00, 'False Positive Rate', ha='center', fontsize=7.5)
        f.legend(handles, labels, bbox_to_anchor=(0.69, 0.00), ncol=2, fontsize=9)
 #       f.ylabel('True Positive Rate',fontsize=7.5)            
            
        f.tight_layout()
        f.savefig(f"/MULTIX/DATA/HOME/covid-19-benchmarking/results/seg_{model_names[1]}_roc_new.pdf",dpi=300, bbox_inches='tight')   
       
for i in ['fusenet','mag_sd','res_attn','coronet','coronet_tfl','covidnet','ssl_am','ecovnet','xvitcos','capsnet']:
    model_prs(i)

#for i in ['coronet_tfl_seg','xvitcos_seg','ssl_am_seg']:
 #   model_rocs(i)
#model_names = [['xvitcos','xvitcos_seg'],['ssl_am','ssl_am_seg'],['coronet_tfl','coronet_tfl_seg']]
#for m in model_names:
 #   print(m)
  #  data_rocs(model_names=m)
