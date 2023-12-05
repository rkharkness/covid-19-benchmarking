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

def ci(y):
    # mean  +/- t*s/sqrt n
    return 1.96 * np.std(y) / np.sqrt(len(y)) #    return stats.t.interval(alpha=0.95, df = len(data)-1, loc= np.mean(data), scale=stats.sem(data))


#plt.setp(l.get_title(), multialignment='center')

ltht_pneum = {'No Chest X-ray Comorbidity': 929, 'Chest X-ray Comorbidity': 711, 'Asian': 267, 'Black': 147, 'Multiple': 28, 'Other': 42, 'Unknown': 170, 'White': 2312, 'Current': 298, 'Former': 615, 'Never': 1370, ' Age Group: 0-24': 176, 'Age Group: 100-125': 49, 'Age Group: 25-49': 386, 'Age Group: 50-74': 1373, 'Age Group: 75-99': 1964, 'F': 1691, 'M': 2256}
ltht_no_pneum = {'No Chest X-ray Comorbidity': 683, 'Chest X-ray Comorbidity': 477, 'Asian': 141, 'Black': 110, 'Multiple': 21, 'Other': 29, 'Unknown': 105, 'White': 959, 'Current': 42, 'Former': 143, 'Never': 551, ' Age Group: 0-24': 77, 'Age Group: 25-49': 116, 'Age Group: 50-74': 633, 'Age Group: 75-99': 615, 'F': 532, 'M': 919}
ltht = {'No Chest X-ray Comorbidity': 1412, 'Chest X-ray Comorbidity': 1214, 'Asian': 726, 'Black': 302, 'Multiple': 72, 'Other': 160, 'Unknown': 464, 'White': 6737, 'Current': 958, 'Former': 1548, 'Never': 3742, ' Age Group: 0-24': 1076, 'Age Group: 100-125': 109, 'Age Group: 25-49': 1497, 'Age Group: 50-74': 3764, 'Age Group: 75-99': 4758, 'F': 5108, 'M': 6095}

popsize_dict = ltht_pneum
#popsize_dict = {'Female':1246, 'Male':1658, 'Age Group: 0-24':1076,'Age Group: 100+': 109, \
#'Age Group: 50-74':3764, 'Age Group: 25-49':1497, 'Age Group: 75-99':4758, 'Current':958, 'Former':1548, 'Never':3742, \
#'Asian':726, 'Black':302, 'Multiple':72, 'Other':160, 'White':6737, 'Unknown':0, }


def barplot(data, subpop, popsize_dict, fig=None, ax=None, iter=0, group=0):
    data = data[subpop]
    print(data['roc_auc'], 'k')
    mean_auc = np.mean(data['roc_auc'])
    std_auc = np.std(data['roc_auc'])
#    plt.gca().set_prop_cycle(None)
#    x = np.arange(5)[iter]  # the label locations
    width = 0.55  # the width of the bars
    multiplier = group

    x = iter
    n = popsize_dict[subpop]
    offset = width
    rects = ax.barh(x * width+group, mean_auc, width, xerr=std_auc, capsize=2)
    format_map = {"All":"All","F":"Female", "M":"Male","Chest X-ray Comorbidity":"Yes", "No Chest X-ray Comorbidity":"No","Age Group: 100-125":"100+ years"," Age Group: 0-24": "0-24 years", "Age Group: 50-74":"50-74 years", 'Age Group: 75-99':"75-99 years", 'Age Group: 25-49':'25-49 years'}
    if subpop in format_map.keys():
        subpop = format_map[subpop]
    ax.bar_label(rects, labels=[r'\textbf{{{}}}: n = {}'.format(subpop,n)], fontsize=6, padding=2)
    return fig, ax



def read_results(data, subpop, popsize_dict, plot=False, fig=None, ax=None, iter=1):
#  if subpop != 'Unknown':
    data = data[subpop]
    print("\n", subpop)    
   # gender_map = {'F':'Female','M':'Male'} 
   # subpop = gender_map[subpop]
    mean_accuracy = np.mean(data['acc'])
    std_accuracy = np.std(data['acc'])
    ci_accuracy = ci(data['acc'])

    mean_f1 = np.mean(data['f1'])
    std_f1 = np.std(data['f1'])
    ci_f1 = ci(data['f1'])

    mean_auc = np.mean(data['roc_auc'])
    std_auc = np.std(data['roc_auc'])
    ci_roc_auc = ci(data['roc_auc'])

    mean_precision = np.mean(data['precision'])
    std_precision = np.std(data['precision'])
    ci_precision = ci(data['precision'])

    mean_recall = np.mean(data['recall'])
    std_recall = np.std(data['recall'])
    ci_recall = ci(data['recall'])

    mean_pr_auc = np.mean(data['prec_auc'])
    std_pr_auc = np.std(data['prec_auc'])
    ci_pr_auc = ci(data['prec_auc'])

    mean_ppv = np.mean(data['ppv'])
    std_ppv = np.std(data['ppv'])
    ci_ppv = ci(data['ppv'])

    print('\n accuracy \n')
    print('mean', mean_accuracy)
    print('std', std_accuracy)

    print('\n f1 \n')
    print('mean', mean_f1)
    print('std', std_f1)

    print('\n roc auc \n')
    print('mean', mean_auc)
    print('std', std_auc)

    print('\n precision \n')
    print('mean', mean_precision)
    print('std', std_precision)
    
    print('\n recall \n')
    print('mean', mean_recall)
    print('std', std_recall)

    print('\n pr auc \n')
    print('mean', mean_pr_auc)
    print('std', std_pr_auc)

    print('\n ppv \n')
    print('mean', mean_ppv)
    print('std', std_ppv) 

      
    if plot == True:

           # if iter % 2 != 0:
           #     marker = '^'
          #  else:
            markers = [None, None, None, None, None, None, None, None, None, None] #[None,'s','d','^','p','<','*','P','>','H']
        #if subpop !='All':
            ax[0].scatter(iter, mean_accuracy, marker=markers[iter-1])
            ax[0].errorbar(iter,mean_accuracy,yerr=ci_accuracy)
            ax[0].set_title('Accuracy', fontsize=7.5)
            ax[0].set_xticks([])
            ax[0].set_ylabel('Score', fontsize=7.5)
            ax[0].set_ylim(0, 1.05)


            ax[0].set_xlim(left=-0.5, right=iter+0.8)
            ax[1].scatter(iter, mean_precision, marker=markers[iter-1])
            ax[1].errorbar(iter,mean_precision,yerr=ci_precision)
            ax[1].set_title('Precision', fontsize=7.5)
            ax[1].set_xticks([])
            ax[1].set_ylim(0, 1.05)

            ax[1].set_xlim(left=-0.5, right=iter+0.8)
            #ax[1].set_yticks([])
#            for spine in ax[1].spines:
 #               ax[1].spines[spine].set_visible(False)
            ax[2].scatter(iter, mean_recall, marker=markers[iter-1])
            ax[2].errorbar(iter,mean_recall,yerr=ci_recall)
            ax[2].set_xlim(left=-0.5, right=iter+0.8)
            ax[2].set_title('Recall', fontsize=7.5)
            ax[2].set_xticks([])
            ax[2].set_ylim(0, 1.05)


            ax[3].scatter(iter, mean_ppv, marker=markers[iter-1])
            ax[3].errorbar(iter,mean_ppv,yerr=ci_ppv)
            ax[3].set_xlim(left=-0.5, right=iter+0.8)
            ax[3].set_title('PPV', fontsize=7.5)
            ax[3].set_xticks([])
            ax[3].set_ylim(0, 1.05)

            ax[4].scatter(iter, mean_auc, marker=markers[iter-1])
            ax[4].errorbar(iter,mean_auc,yerr=ci_roc_auc)
            ax[4].set_xlim(left=-0.5, right=iter+0.8)
            ax[4].set_title('AUC', fontsize=7.5)
            ax[4].set_xticks([])
            ax[4].set_ylim(0, 1.05)

            fig1, ax1 = plt.subplots()

            #ax[2].set_yticks([])
  #          for spine in ax[2].spines:
  #              ax[2].spines[spine].set_visible(False)
    return fig, ax

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training Script') 
    parser.add_argument('--results_csv', default=None, type=str, help='Path to data file')
    parser.add_argument('--subpop', default=False, type=bool, help='One of: Ethnicity, Comorbidity, Smoker, Gender, Age')
    parser.add_argument('--comparison', default=False, type=bool)
    parser.add_argument('--test', default='ltht', type=str, help='One of: ltht, covid_qu_ex, nccid_test, ltht_no_pneumonia')
    parser.add_argument('--seg', type=bool)
    parser.add_argument('--barplot', type=bool)
    args = parser.parse_args()

##    if os.isfile(args.results_csv) == False:
  #      subpop_files = glob.glob(args.results_csv)
   #     print(subpop_files)
    #if 'Age' in args.results_csv:
    #sorted_age_dict = sorted(age_dict.keys(), key=lambda x: int(x[12:13]))

    #fig, ax = plt.subplots(1, 3, sharey=True)
    #with open(args.results_csv) as f:     
     #   results_data = json.load(f)
    #if args.subpop == False:
    #    fig, ax = read_results(results_data, 'All',plot=True,fig=fig,ax=ax)
    test_map = {'nccid_test':'NCCID TEST', 'ltht':'LTHT', 'ltht_no_pneumonia': 'LTHT (NP)', 'ltht_pneumonia':'LTHT (P)', 'covidgr':'COVIDGR'}
    if args.comparison == True:
        if args.test == 'ltht':
            results_paths = glob.glob("/MULTIX/DATA/HOME/covid-19-benchmarking/results/*/ltht/*All.json")
        if args.test == 'covid_qu_ex':
            results_paths = glob.glob("/MULTIX/DATA/HOME/covid-19-benchmarking/results/*/covid_qu_ex/*All.json")
        if args.test == 'ltht_no_pneumonia':
            results_paths = glob.glob("/MULTIX/DATA/HOME/covid-19-benchmarking/results/*/ltht_no_pneumonia/*All.json")
        if args.test == 'nccid_test':
            results_paths = glob.glob("/MULTIX/DATA/HOME/covid-19-benchmarking/results/*/nccid_test/*All.json")
        if args.test == 'ltht_pneumonia':
            results_paths = glob.glob("/MULTIX/DATA/HOME/covid-19-benchmarking/results/*/ltht_pneumonia/*All.json")
        if args.test == 'covidgr':
            results_paths = glob.glob("/MULTIX/DATA/HOME/covid-19-benchmarking/results/*/covidgr/*All.json")

        if args.seg == True:
            seg_models = ['coronet_tfl','coronet_tfl_seg','ssl_am','ssl_am_seg', 'xvitcos', 'xvitcos_seg']
            results_paths = [i for i in results_paths if i.split('/')[6] in seg_models]
        else:
            not_models = ['coronet_tfl_seg','ssl_am_seg','xvitcos_seg']
            results_paths = [i for i in results_paths if i.split('/')[6] not in not_models]

        fig, ax = plt.subplots(1, 5, sharey=True, figsize=(6,3))
        for idx, i in enumerate(results_paths):
            with open(i) as f:
                i = json.load(f)
            fig, ax = read_results(i, 'All', popsize_dict=popsize_dict, plot=True, fig=fig, ax=ax, iter=idx)
        
        axbox = ax[2].get_position()
        model_name = [r.split('/')[6] for r in results_paths]
        model_name = [i.upper() for i in model_name]
        model_name_map = {'CORONET_TFL_SEG':'XCEPTION NET (ROI)', 'SSL_AM_SEG':'SSL-AM (ROI)','XVITCOS_SEG':'XVITCOS (ROI)','CAPSNET':'CAPSNET','FUSENET':'FUSENET','MAG_SD':'MAG-SD', 'CORONET':'CORONET','ECOVNET':'ECOVNET','XVITCOS':'XVITCOS','CORONET_TFL':'XCEPTION NET','RES_ATTN':'RES. ATTN.', 'COVIDNET':'COVIDNET', 'SSL_AM':'SSL-AM'}
        model_name = [model_name_map[k] for k in model_name]
        fig.legend(model_name, bbox_to_anchor=[axbox.x0*axbox.width+0.45, axbox.y0-3.], bbox_transform=fig.transFigure, loc='lower center', ncol=3, title=r'\textbf{Models}', prop={'size':7})
        title = test_map[args.test]
        fig.suptitle(r'\textbf{{{}}}'.format(title))
        fig.subplots_adjust(top=0.80)
        fig.savefig(f'/MULTIX/DATA/HOME/covid-19-benchmarking/results/{args.test}_model_comp_point_plot.pdf',bbox_inches="tight")
    else:    
        #popsize_dict = {'F':1246, 'M':1658,' Age Group: 0-24':1076,'Age Group: 100-125': 109, \
        #'Age Group: 50-74':3764, 'Age Group: 25-49':1497, 'Age Group: 75-99':4758, 'Current':958, 'Former':1548, 'Never':3742, \
        #'Asian':726, 'Black':302, 'Multiple':72, 'Other':160, 'White':6737, 'No Chest X-ray Comorbidity': 1412, 'Chest X-ray Comorbidity':1214}
        gender_map = {'F':'Female','M':'Male'}
        fig, ax = plt.subplots(1, 5, sharey=True)
#        fig1, ax1 = plt.subplots()
       # with open(args.results_csv) as f:     
#        #    results_data = json.load(f)
 #       if args.subpop == False:
 #           fig, ax = read_results(results_data,'All',None, plot=True,fig=fig,ax=ax)
        if args.barplot == True:
            results_list = glob.glob("/MULTIX/DATA/HOME/covid-19-benchmarking/results/coronet_tfl/ltht_pneumonia/*.json")
            results_list = [r for r in results_list if 'ppv' not in r]      
            results_list = [r for r in results_list if 'All' not in r]      
            results_list = [r for r in results_list if 'Comorbidity' not in r]      

            fig1, ax1 = plt.subplots(figsize=(6,4)) 
            iter = 0
            #if args.test == 'ltht': group_names = ["Age Group","Ethnicity", "Sex", "Smoker"]
            group_names = ["Age Group","Ethnicity", "Sex", "Smoker"]

            for n, res in enumerate(results_list):
                plt.gca().set_prop_cycle(None)
                plt.rcParams['axes.prop_cycle'] = cycler('color', plt.get_cmap('tab20').colors)
                print(res)
                with open(res) as f:
                    results_data = json.load(f)
                    if 'Age' in res:
                        print('soorting')
                        sorted_results_data = sorted(results_data.keys(), key=lambda x: int(x.split('-')[-1]))
                        print(sorted_results_data)
                        new_results_data = {}
                        for i in sorted_results_data:
                           new_results_data[i] = results_data[i]
                        results_data = new_results_data
                        print(results_data.keys())
                    if 'Unknown' in results_data: del results_data['Unknown']
             #       if args.test == 'ltht':
                    if 'Age Group: 100-125' in results_data: del results_data['Age Group: 100-125']
                    if 'No Chest X-ray Comorbidity' in results_data: del results_data['No Chest X-ray Comorbidity']
                    if 'Chest X-ray Comorbidity' in results_data: del results_data['Chest X-ray Comorbidity']
                    if 'Black' in results_data: del results_data['Black']
                    if 'Other' in results_data: del results_data['Other']
               # fig1, ax1 = barplot(results_data, i, popsize_dict,fig=fig1, ax=ax1, iter=idx)
                #iter = 0
#                fig1, ax1 = plt.subplots()
                y_ticks = [0.8,3.7,6.1,8.5]
     #           else: y_ticks = [1.0, 4, 7., 9.9, 12.25]
                ax1.set_yticks(y_ticks, group_names, rotation=45)   
                ax1.set_xlim(0.5, 1.0)
                ax1.set_xlabel('AUC')
                for idx, i in enumerate(results_data.keys()):
                     print(i, 'dic')
                     fig1, ax1 = barplot(results_data, i, popsize_dict,fig=fig1, ax=ax1, iter=iter, group=n)
                     iter +=1

 #                    break
#                break
            fig1.savefig('/MULTIX/DATA/HOME/covid-19-benchmarking/results/coronet_tfl/ltht_pneumonia/comp_subpop_bar.pdf')

        if args.subpop==True:
            #results_list = glob.glob("/MULTIX/DATA/HOME/covid-19-benchmarking/results/coronet_tfl/ltht/*.json")
            #for res in results_list:
            if 'Age' in args.results_csv:
                sorted_results_data = sorted(results_data.keys(), key=lambda x: int(x.split('-')[-1]))
                print(sorted_results_data)
                new_results_data = {}
                for i in sorted_results_data:
                    new_results_data[i] = results_data[i]
                results_data = new_results_data
                print(results_data.keys())
            if 'Unknown' in results_data: del results_data['Unknown']
  
            for idx, i in enumerate(results_data.keys()):
                fig, ax = read_results(results_data, i, popsize_dict, plot=True, fig=fig, ax=ax, iter=idx)            
 #           fig1.savefig('/MULTIX/DATA/HOME/covid-19-benchmarking/results/coronet_tfl/ltht/subpop_bar.pdf')

#                fig1, ax1 = barplot(results_data, i, popsize_dict,fig=fig1, ax=ax1, iter=idx)


            axbox = ax[2].get_position()
            if 'F' in results_data.keys():
                keys = [gender_map[k] for k in results_data.keys()]
                keys = [i + f' (n={popsize_dict[i]})' for i in keys]
            else:
#                keys = results_data.keys()
                if 'Unknown' in results_data: del results_data['Unknown']
                keys = results_data.keys()
               # keys = keys.drop('Unknown')
                keys = ['Age Group: 100+' if i == 'Age Group: 100-125' else i for i in keys]
                keys = ['Age Group: 0-24' if i == ' Age Group: 0-24' else i for i in keys]

                for i in keys:
                    print(i) 
                keys = [i + f' (n={popsize_dict[i]})' for i in keys]
            fig.legend(keys, bbox_to_anchor=[axbox.x0*axbox.width+0.45, axbox.y0-0.2], bbox_transform=fig.transFigure, loc='lower center', ncol=len(results_data.keys())//2+2, title=r'\textbf{Subgroups}', prop={'size': 6})
 #           fig.savefig('/MULTIX/DATA/HOME/covid-19-benchmarking/results/coronet_tfl/ltht_no_pneumonia/point_plot.pdf',bbox_inches="tight")
 #           fig1.savefig('/MULTIX/DATA/HOME/covid-19-benchmarking/results/coronet_tfl/ltht/subpop_bar.pdf')
