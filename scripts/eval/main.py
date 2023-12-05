from classification_metrics import ModelPerformanceEval, k_fold_eval
from create_subpops import create_subpopulations

import sys
sys.path.append('../')

from scripts.dataloaders import make_generators

from scripts.models.res_attn import AttentionResNetModified
from scripts.models.ecovnet import ECovNet
from scripts.models.coronet_tfl import CoroNet_Tfl

from scripts.models.coronet_tfl_seg import CoroNet_Tfl_Seg
from scripts.models.ssl_am_seg import SSL_AM_Seg
from scripts.models.ssl_am2 import SSL_AM as SSL_AM

from scripts.models.covidnet import CovidNet
from scripts.models.mag_sd import MAG_SD, config
from scripts.models.fusenet import FuseNet
from scripts.models.capsnet import CovidCaps

from scripts.models.xvitcos import xVitCOS
from scripts.models.xvitcos_seg import xVitCOS_Seg

from scripts.models.coronet import CoroNet

# from tensorflow.keras.models import Model
import tensorflow as tf

from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import pandas as pd
import argparse
import glob
# from tqdm import tqdm
import json
import ast
from scipy import stats
from create_subpops import create_subpopulations

plt.style.use(['science','nature'])

def ci(data):
    # mean  +/- t*s/sqrt n
    return stats.t.interval(alpha=0.95, df = len(data)-1, loc= np.mean(data), scale=stats.sem(data))

def data_to_subpop_dict(df, test, subpop):
    if subpop:
        data_dict = create_subpopulations(df, test)
    else:
        data_dict = {'All': df}
        data_dict = {'All': data_dict}
    
    return data_dict

def datawise_performance(model, data_name, mean_x, y, aucs, fig, ax, test, name, idx):
        mean_y = np.mean(y, axis=0)
        mean_auc = auc(mean_x, mean_y)
        std_auc = np.std(aucs)
        if name == 'roc':
            ax[0].plot(mean_x, mean_y, label=data_name, lw=1, alpha=.8)   #'%s: \n AUC = %0.2f $\pm$ %0.2f' % (data_name, mean_auc, std_auc),
            ax[0].set_title(model, rotation='vertical', x=-0.1,y=0.5) #,fontsize=8)
            ax[0].set_xlabel('False Positive Rate') #,fontsize=7)
            ax[0].set_ylabel('True Positive Rate') #,fontsize=7)
            ax[0].plot([0, 1], [0, 1], linestyle='--', lw=0.75, color='r', label=None, alpha=.6)

            if data_name == 'nccid_test':
              ax[1].plot(mean_x, mean_y, label=data_name,lw=1, alpha=.8)
              ax[2].plot(mean_x, mean_y, label=data_name,lw=1, alpha=.8)        
              ax[3].plot(mean_x, mean_y, label=data_name,lw=1, alpha=.8)
              ax[4].plot(mean_x, mean_y, label=data_name,lw=1, alpha=.8)
            else:
              ax[idx].plot(mean_x, mean_y, label=data_name,lw=1, alpha=.8)
              ax[idx].set_title(data_name)

        else:
            ax.plot(mean_x, mean_y,
                    label='%s: \n AP = %0.2f $\pm$ %0.2f' % (data_name, mean_auc, std_auc),
                    lw=1, alpha=.8)

        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])

        std_tpr = np.std(y, axis=0)

        conf_int = ci(y)
        tprs_upper = np.minimum(mean_y + conf_int[1], 1)
        tprs_lower = np.maximum(mean_y - conf_int[0], 0)
        ax.fill_between(mean_x, tprs_lower, tprs_upper, label=None, alpha=.2)

        if name == 'roc':
            ax.set_title(model,fontsize=14)
            ax.set_xlabel('False Positive Rate',fontsize=12)
            ax.set_ylabel('True Positive Rate',fontsize=12)
        else:
            ax.set_title(f"Cross-Validation Precision-Recall Curve of {model['model_name'].upper()}",fontsize=14)
            ax.set_xlabel('Precision',fontsize=12)
            ax.set_ylabel('Recall',fontsize=12)

        return fig, ax

def comparison_fn(model_name, dataset, name='roc',pneumonia_vs_rest=False,model_wise=False):

    if model_wise == True:
        test = dataset
        comp_list = ['coronet_tfl', 'fusenet', 'res_attn', 'ecovnet']
        data = [f"/MULTIX/DATA/HOME/covid-19-benchmarking/results/{i}/{i}_performance_{dataset}_df.csv" for i in comp_list]

    else:
        if pneumonia_vs_rest == True:
            results1 = glob.glob('/MULTIX/DATA/HOME/covid-19-benchmarking/results/*/ltht/*All.json')
            results2 = glob.glob('/MULTIX/DATA/HOME/covid-19-benchmarking/results/*/ltht_pneumonia/*All.json')
            results3 = glob.glob('/MULTIX/DATA/HOME/covid-19-benchmarking/results/*/ltht_no_pneumonia/*All.json')
            results = results1 + results2 + results3
        else:
            results = glob.glob('/MULTIX/DATA/HOME/covid-19-benchmarking/results/*/*/*All.json')
            results = [i for i in results if i.split('/')[7] not in ['ltht_pneumonia','nccid_val','covid_qu_ex','ltht_no_pneumonia']]

        test = 'All'

        print(results)
        models = [i.split('/')[6] for i in results]
        models = sorted(list(set(models)))
        data = [i.split('/')[7] for i in results]
        data = list(set(data))
        dataset_name = {'nccid_val':'NCCID Val','nccid_test':'NCCID Test','ltht_pneumonia':'LTHT (Pneumonia)', 'ltht':'LTHT', 'ltht_no_pneumonia':'LTHT (No Pneumonia)', 'covid_qu_ex':'COVID-QU-Ex'}
        model_name_map = {'FUSENET':'FUSENET','MAG_SD':'MAG-SD', 'CORONET':'CORONET','ECOVNET':'ECOVNET','XVITCOS':'XVITCOS','CORONET_TFL':'CORONET (TFL)','RES_ATTN':'RES. ATTN.', 'COVIDNET':'COVIDNET', 'SSL_AM':'SSL-AM'}

    with plt.style.context(['science', 'nature']):
        print(len(data)-1)
        print(len(models))
        fig, axs = plt.subplots(len(models), len(data)-1, figsize=(5,8)) #, sharex=True, sharey=True, constrained_layout=True)
        for m, ax in zip(models, axs.ravel()):

            counter = 0
            for d in data:
                counter += 1
                with open(f'/MULTIX/DATA/HOME/covid-19-benchmarking/results/{m}/{d}/{m}_performance_{d}_All.json') as f:
                    r = json.load(f)

                r = r['All']
                if name=='roc':
                    mean_x = np.linspace(0, 1, 100)
                    x = r['fpr']
                    aucs = r['roc_auc']
                    y = r['interp_tpr']
                else:
                    mean_x = np.linspace(1, 0, 100)
                    y = r['interp_rec']
                    aucs = r['prec_auc']

                fig, ax = datawise_performance(model_name_map[m.upper()], dataset_name[d], mean_x, y, aucs, fig, ax, test, name, counter)

        handles, labels = ax.get_legend_handles_labels()

        by_label = dict(zip(labels, handles))
        by_label = dict(sorted(by_label.items(), reverse=True))
        fig.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.5, -0.5), borderaxespad=0.6, loc='lower center', ncol=5, title=r'\textbf{Data}', prop={'size':6})

        plt.tight_layout()
        if pneumonia_vs_rest == True:
            fig.savefig(f"/MULTIX/DATA/HOME/covid-19-benchmarking/results/data_pneum_wise_roc.pdf",dpi=300, bbox_inches='tight')
        else:
            fig.savefig(f"/MULTIX/DATA/HOME/covid-19-benchmarking/results/data_wise_roc.pdf",dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training Script')
    parser.add_argument('--data_csv', default='/MULTIX/DATA/HOME/covid-19-benchmarking/data/nccid_preprocessed14_21.csv', type=str, help='Path to data file')
    parser.add_argument('--weights', default=None)
    parser.add_argument('--test', default=None, type=str, help='Choose on of: [ltht, nccid_test, nccid_val, nccid_leeds, chexpert, custom]')
    parser.add_argument('--subpop_analysis', default=False, type=bool, help='Perform subpop analysis or not, populations dependent on selected test dataset')
    parser.add_argument('--seg', default=False, type=bool)
    parser.add_argument('--model_name', type=str, default=None)
    args = parser.parse_args()

    assert args.weights != None, "Weights not provided"
    assert args.test != None, "Test dataset not provided"

    assert args.model_name in ['coronet', 
                               'coronet_tfl', 
                               'coronet_tfl_seg'
                               'covidnet', 
                               'covidcaps', 
                               'fusenet', 
                               'mag_sd', 
                               'ssl_am', 
                               'ssl_am_seg',
                               'xvitcos', 
                               'xvitcos_seg', 
                               'ecovnet', 
                               'res_attn'], "Model name not found"

    if args.seg: assert 'seg' in args.model_name, "Segmentation data used with non-segmentation model"
    if 'seg' in args.model_name: assert args.seg == True, "Segmentation model used with non-segmentation data"

    tf.config.run_functions_eagerly(True)

    df = pd.read_csv(args.data_csv)

    model_dict = {'coronet':CoroNet(supervised=True, pretrained=True),
                'coronet_tfl':CoroNet_Tfl(dropout_act=False),
                'cornet_tfl_seg':CoroNet_Tfl_Seg(dropout_act=False),
                'covidnet':CovidNet(pretrained=True),
                'capsnet':CovidCaps(supervised=True, pretrained=True),
                'fusenet':FuseNet(dropout_act=False),
                'mag_sd':MAG_SD(config=config),
                'ssl_am':SSL_AM(supervised=True, pretrained=True),
                'ssl_am_seg':SSL_AM_Seg(supervised=True, pretrained=True),
                'xvitcos':xVitCOS(pretrained=True),
                'xvitcos_seg':xVitCOS_Seg(pretrained=args.pretrained_weights),
                'ecovnet':ECovNet(dropout_act=False),
                'res_attn':AttentionResNetModified(dropout_act=False),
                }

    assert args.test in ['pneumonia_comparison', 
                         'data_wise_comparison',
                         'covidgr', 
                         'ltht', 
                         'ltht_no_pneumonia', 
                         'ltht_pneumonia', 
                         'nccid_test'], "Test dataset not found"
    
    ltht = False
    if 'ltht' in args.test:
        ltht = True
    else:
        assert args.subpop_analysis == False, "Subpop analysis not supported for non-LTHT datasets"

    if ltht:
        if args.seg == True:
            paths = df['paths'].values
            new_paths = [i.split('/')[4:] for i in paths]
            join_new_path = ['_'.join(i) for i in new_paths]
            df['cxr_path'] = ['/MULTIX/DATA/INPUT/ltht_dcm_seg/' + i for i in join_new_path]
        else:
            df['cxr_path'] = df['paths']

        df['xray_status'] = df['FinalPCR']
        df['kfold_1'] = 'test'

        data_dict = data_to_subpop_dict(df, args.test, args.subpop_analysis)

    if args.test == 'nccid_test':
        if args.seg == True:
            paths = df['cxr_path'].values
            new_paths = [i.split('/')[-1] for i in paths]
            join_paths = ['_'.join(i) for i in new_paths]
            final_paths = ['/MULTIX/DATA/HOME/covid-19-benchmarking/data/nccid_dcm_seg/' + i  for i in new_paths]
            df['cxr_path'] = final_paths
            df['xray_status'] = df['label']
            df_o = pd.read_csv('/MULTIX/DATA/HOME/covid-19-benchmarking/data/nccid_preprocessed14_21.csv')
            df = pd.merge(df_o, df, left_on='cxr_path', right_on = 'original_path')
            df['cxr_path'] = df['cxr_path_y']

        df = df[df['xray_status']!=np.nan]
        df = df.dropna(subset=['xray_status'])
        df = df[df['kfold_1'] == 'test']

        data_dict = data_to_subpop_dict(df, args.test, args.subpop_analysis)

    elif args.test == 'ltht_no_pneumonia':
        df = df[df['finding']!=np.nan]
        df = df[df['finding']!='pneumonia']
        df = df.dropna(subset=['finding'])

        df = df[df['finding']!=np.nan]
        df = df[df['finding']!='pneumonia']
        df = df.dropna(subset=['finding'])

        mapping = {'covid-19':1, 'negative':0}

        df['xray_status'] = df['finding'].map(mapping)
        df['cxr_path'] = df['paths']
        df['kfold_1'] = 'test'

        data_dict = data_to_subpop_dict(df, args.test, args.subpop_analysis)

    elif args.test == 'ltht_pneumonia':
        df = df[df['finding']!=np.nan]
        mapping = {'covid-19':1, 'pneumonia':1, 'negative':0}
        df = df.dropna(subset=['finding'])
        df['xray_status'] = df['finding'].map(mapping)
        df['kfold_1'] = 'test'
        
        data_dict = data_to_subpop_dict(df, args.test, args.subpop_analysis)

    elif args.test == 'covidgr':
       if args.seg == True:
           df_s = pd.read_csv('/MULTIX/DATA/HOME/covid-19-benchmarking/data/covidgr_data_severity.csv')
           df = pd.merge(df_s, df, left_on='paths', right_on='original_path')
       else:
           df['cxr_path'] = df['paths']

       data_dict = data_to_subpop_dict(df, args.test, args.subpop_analysis)

    model_class = model_dict[args.model_name]
    model = model_class.build_model()

    if args.test == 'data_wise_comparison':
        comparison_fn(model['model_name'], df)

    elif args.test == 'pneumonia_comparison':
        comparison_fn(model['model_name'], df, pneumonia_vs_rest=True)

    if args.test != 'data_wise_comparison':

        for subpop in data_dict.keys():
            pop = {k:v for k, v in data_dict[subpop].items()}    
            k_fold_eval(model_class.build_model(), args.weights, ltht, df, args.test, subpop, pop)
