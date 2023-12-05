import sys 
sys.path.append('../') 
import keras 
from scripts.dataloaders import make_generators 
from scripts.models.fusenet import FuseNet 
from scripts.models.capsnet import CovidCaps 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score 
from tensorflow.keras.utils import to_categorical 
import torch 
from tqdm import tqdm 
import pandas as pd 
from scipy import interp 
import itertools 
import tensorflow as tf 
import ast 
import json 
from keras import backend as K


class ModelPerformanceEval():
    def __init__(self, model, weights, df, k, test, ltht, subpop_name, subpop=None):

        # model = CovidCaps(pretrained=True)        
        # self.model = model.build_model((480,480,3),16)
        self.model = model
#        self.reinit_model()
        self.network = self.load_weights(self.model, k, weights)
        self.network = self.model['model']
        if self.model['model_type']=='pytorch':
          if self.model['model_name']!='mag_sd':
            self.network.cuda()

        self.k = k
        self.ltht = ltht
        self.test = test
        self.subpop_name = subpop_name

        # with open(threshold_data) as f:
        #             thresholds = json.load(f)
        #             thresholds = thresholds['All']
        self.best_threshold = 0.5
        print(f'loading threshold: {self.best_threshold}')
        self.subpop = subpop
        
        self.data = df


    def reinit_model(self):
     session = K.get_session()
     for layer in self.model['model'].layers: 
         for v in layer.__dict__:
             v_arg = getattr(layer,v)
             if hasattr(v_arg,'initializer'):
                 initializer_method = getattr(v_arg, 'initializer')
                 initializer_method.run(session=session)
                 print('reinitializing layer {}.{}'.format(layer.name, v))

    def load_weights(self, model, k, weights):
        print(f"loading weights: {weights} for {model['model_name']}...")
        
        def convert_path(weights, k, suffix):
            weights_path = f"{weights}_{k}{suffix}"
            return weights_path

        if model['model_type'] == 'keras':
            suffix = '.h5'
            weights = convert_path(weights, k, suffix)
            print(f"loading weights: {weights} for {model['model_name']}...")
            model['model'].built = True
            
            model['model'].load_weights(weights)
            print('loaded weights successfully')
            model['model'].built = True
            model['model'].trainable = False
        
        elif model['model_type'] == 'pytorch':
            if model['model_name'] == 'mag_sd':
                model['model'].resume_model_from_path(weights, k)
                model['model'].set_eval()
            else:
                suffix = '.pth'
                weights = convert_path(weights, k, suffix)
                print(f"loading weights: {weights} for {model['model_name']}...")
                model['model'].load_state_dict(torch.load(weights,map_location=torch.device('cpu')))
                model['model'].eval()
        
        return model

    def generate_predictions(self, dataloader, get_classes):
        gt_list = []
        pred_list = []
        path_list = []
        for data in tqdm(dataloader):
            if self.model['model_type']=='pytorch':
                img, label, path = data

                #print(label)
                img = img.cuda()
                #path = None
                if self.model['model_name'] == 'mag_sd':
                    pred, _, _ = self.network.encoder(img)
                else:
                    pred = self.network(img)
                
                label = label.numpy()
                pred = pred.detach().cpu().numpy()

            else:
                img, label, path = data
                pred = self.network(img)

                if self.model['model_name'] == 'capsnet':
                    label = to_categorical(label, 2)

            pred_list.append(pred)
            gt_list.append(label)
            path_list.append(path)


        gt_list  = np.array([item for sublist in gt_list for item in sublist])
        pred_list = np.array([item for sublist in pred_list for item in sublist])
        path_list = np.array([item for sublist in path_list for item in sublist])

        if get_classes == True:
        
          if self.model['model_name'] == 'mag_sd' or self.model['model_name'] == 'capsnet':
            fpr, tpr, thresholds, roc_auc = self.create_roc_curve(gt_list, pred_list, plot=False)
            output_list = np.array([np.argmax(i) for i in pred_list]).flatten()
            if self.best_threshold == None:              
              best_threshold = self.calculate_best_threshold(fpr, tpr, thresholds)
            else:
              best_threshold = self.best_threshold
          else:
            if self.best_threshold == None:
              fpr, tpr, thresholds, roc_auc = self.create_roc_curve(gt_list, pred_list, plot=False)
              best_threshold = self.calculate_best_threshold(fpr, tpr, thresholds)
              output_list = np.array([np.int(i>best_threshold) for i in pred_list])
            else:
              output_list = np.array([np.int(i>self.best_threshold) for i in pred_list])
              best_threshold = self.best_threshold
        else:
            output_list = []
        return gt_list, pred_list, output_list, best_threshold, path_list


    def plot_misclassified_attributes(self, missclassification_df):
        if self.test == 'nccid_test':
            cols_of_interest = ['view']
            xlab_map = {'view':'Projection','temperature_on_admission': 'Temperature','respiratory_rate_on_admission':'Respiratory Rate','days_around':'Days around diagnostic window (+/-)','lymphocyte_count_on_admission':'Lymphocyte Count','pao2':'PaO2', 'wcc_on_admission': 'White Cell Count','cxr_severity': 'CXR Severity','pmh_lung_disease':'Pre-existing Lung Disease'}
  
        elif self.test == 'covidgr':
            cols_of_interest = ['Severity']

        elif self.test == 'ltht':
            cols_of_interest = ['view']
            xlab_map = {'view': 'Projection'}

        for col in cols_of_interest:
            missclassification_df = missclassification_df[missclassification_df['kfold']==1]

            if self.test == 'covidgr':
                 missclassification_df = missclassification_df[missclassification_df['xray_status']==1.0]  
                 missclassification_df[col] = missclassification_df[col].astype(str)

            if self.test == 'ltht':
                def find_view(a):
                    if 'PA' in a.upper():
                        return 'PA'
                    elif 'AP' in a.upper():
                        return 'AP'
                    else:
                        return None

                if col == 'days_around':
                    missclassification_df = missclassification_df[missclassification_df['days_around'] < 31]
                    fps = missclassification_df[missclassification_df['FP']==1]
                    fns = missclassification_df[missclassification_df['FN']==1]

                if col == 'view':
                    missclassification_df['view'] = missclassification_df['cxr_path'].apply(find_view)
                    missclassification_df = missclassification_df[missclassification_df['view']!=None]  

                    fps = missclassification_df[missclassification_df['FP']==1]
                    fns = missclassification_df[missclassification_df['FN']==1]
                    
                    fps_values = fps[col].value_counts().to_dict()
                    view_totals = {'AP':7458,'PA':3733}

                    fps_perc = {}
                    for k,v in view_totals.items():
                        fps_perc[k] = np.round(100*fps_values[k]/view_totals[k], 1)
                    
                    x = np.arange(len(fps_perc.keys()))  # the label locations
                    width = 0.35  # the width of the bars
                    with plt.style.context(['science', 'nature']):
                        bar1 = fps_perc.values()
                        fig, ax = plt.subplots()

                        rects1 = ax.bar(x, bar1, width)
                        ax.set_ylabel('Frequency of False Positives (\%)')
                        ax.set_ylim(0,110)
                        ax.set_xlabel(xlab_map[col])
                        ax.set_xticks(x, fps_perc.keys())
                        ax.bar_label(rects1, padding=3)
                        fig.tight_layout()

                        plt.savefig(f"/MULTIX/DATA/HOME/covid-19-benchmarking/results/{self.model['model_name']}/{self.model['model_name']}_misclass_{col}_{self.test}_{self.k}_fp.png",dpi=300) 
    
            elif self.test == 'covidgr':
                    fns_values = fns['Severity'].value_counts().to_dict()
                    fps = fps.astype(str)
                    freq = {'NORMAL-PCR+':76,'MILD':100,'MODERATE':171,'SEVERE':79} #76 images of Normal-PCR+ severity, 100 mild (mild), 171 moderate (moderate) and 79 severe (serious).
                    fns_perc = {}
                    for k,v in freq.items():
                        fns_perc[k] = np.round(100*fns_values[k]/freq[k], 1)
                    
                    x = np.arange(len(fns_values.keys()))  # the label locations
                    width = 0.35 
                    with plt.style.context(['science', 'nature']):
                        bar1 = fns_perc.values()

                        fig, ax = plt.subplots()
                        rects1 = ax.bar(x, bar1, width)

                        ax.set_ylabel('Frequency of False Negatives (\%)')
                        ax.set_ylim(0, 110)
                        ax.set_xlabel('CXR Severity')

                        ax.set_xticks(x, fns_perc.keys())
                        ax.bar_label(rects1, padding=3)

                        fig.tight_layout()
                        plt.savefig(f"/MULTIX/DATA/HOME/covid-19-benchmarking/results/{self.model['model_name']}/{self.model['model_name']}_misclass_{col}_{self.test}_{self.k}_fp.png",dpi=300) 


    def misclassification_analysis(self, gt_list, pred_list, path):
        misclassifications = {'Path':[],'FN':[], 'FP':[], 'TP':[], 'TN':[], 'kfold': self.k}
        gt_list = np.array(gt_list).flatten()
        pred_list = np.array(pred_list).flatten()

        #path = [x for xs in path for x in xs]


        for i in range(len(pred_list)): 
            misclassifications['Path'].append(path[i])
            if pred_list[i]==1 and gt_list[i]!=pred_list[i]:
                misclassifications['FP'].append(1)
                misclassifications['FN'].append(0)
                misclassifications['TN'].append(0)
                misclassifications['TP'].append(0)

            if pred_list[i]==0 and gt_list[i]!=pred_list[i]:
                misclassifications['FN'].append(1)
                misclassifications['FP'].append(0)
                misclassifications['TN'].append(0)
                misclassifications['TP'].append(0)
            
            if pred_list[i]==1 and gt_list[i]==pred_list[i]:
                misclassifications['FN'].append(0)
                misclassifications['FP'].append(0)
                misclassifications['TN'].append(0)
                misclassifications['TP'].append(1)

            if pred_list[i]==0 and gt_list[i]==pred_list[i]:
                misclassifications['FN'].append(0)
                misclassifications['FP'].append(0)
                misclassifications['TN'].append(1)
                misclassifications['TP'].append(0)

        misclassifications_df = pd.DataFrame.from_dict(misclassifications)
        misclassifications_df = pd.merge(self.data, misclassifications_df, left_on='cxr_path', right_on='Path')
        
        misclassifications_df.to_csv(f"/MULTIX/DATA/HOME/covid-19-benchmarking/results/{self.model['model_name']}/{self.test}/{self.subpop_name}/{self.model['model_name']}_performance_{self.test}_misclassification_df.csv")
        return misclassifications_df

    def categorical_roc_calc(self, gt_list, pred_list):
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        thresholds = dict()

        for i in range(2):
            fpr[i], tpr[i], thresholds[i] = roc_curve(gt_list[:, i], pred_list[:, i], drop_intermediate=False)
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr["micro"], tpr["micro"], thresholds['micro'] = roc_curve(gt_list.flatten(), pred_list.flatten())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        return fpr[1], tpr[1], thresholds[1], roc_auc[1]
        
    def categorical_prec_recall_calc(self, gt_list, pred_list):
        # Compute ROC curve and ROC area for each class
        prec = dict()
        rec = dict()
        pr_auc = dict()
        thresholds = dict()
        for i in range(2):
            prec[i], rec[i], thresholds[i] = precision_recall_curve(gt_list[:, i], pred_list[:, i])
            pr_auc[i] = average_precision_score(gt_list[:, i], pred_list[:, i])

        return prec[1], rec[1], thresholds[1], pr_auc[1]    

    def create_roc_curve(self, gt_list, pred_list, plot=False):
        if self.model['model_name']=='mag_sd' or self.model['model_name']=='capsnet':
            fpr, tpr, thresholds, roc_auc = self.categorical_roc_calc(gt_list, pred_list)
        else:
            fpr, tpr, thresholds = roc_curve(gt_list, pred_list, pos_label=1, drop_intermediate=False)
            roc_auc = auc(fpr, tpr)

        if plot == True:
            with plt.style.context(['science', 'nature']):
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, lw=1, alpha=0.7,
                            label='ROC fold %d (AUC = %0.2f)' % (self.k, roc_auc))
                lw=1
                ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title(f"ROC of {self.model['model_name'].upper()}")
                ax.legend(loc="lower right")
                fig.savefig(f"/MULTIX/DATA/HOME/covid-19-benchmarking/results/{self.test}/{self.subpop_name}/{self.model['model_name']}/{self.model['model_name']}_roc_{self.test}_{self.k}_{self.subpop}.png",dpi=300)

        return fpr.tolist(), tpr.tolist(), thresholds.tolist(), roc_auc

    def create_precision_recall_curve(self, gt_list, pred_list, plot=True):
        if self.model['model_name']=='mag_sd' or self.model['model_name']=='capsnet':
            precision, recall, thresholds, pr_auc = self.categorical_prec_recall_calc(gt_list, pred_list)
        else:
            precision, recall, thresholds = precision_recall_curve(gt_list, pred_list)
            pr_auc = average_precision_score(gt_list, pred_list)

        precision[0] = 0

        if plot == True:
            with plt.style.context(['science', 'nature']):
                fig, ax = plt.subplots()
                ax.plot(precision, recall, lw=1, alpha=0.7,
                            label='Precision-Recall \n fold %d (AP = %0.2f)' % (self.k, pr_auc))
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('Precision')
                ax.set_ylabel('Recall')
                ax.set_title(f"Precision-Recall Curve of {self.model['model_name'].upper()}")
                ax.legend(loc="lower left")
                if self.subpop == None:
                    fig.savefig(f"/MULTIX/DATA/HOME/covid-19-benchmarking/results/{self.model['model_name']}/{self.test}/{self.subpop_name}/{self.model['model_name']}_precision_recall_{self.test}_{self.k}.png",dpi=300)
                else:
                    fig.savefig(f"/MULTIX/DATA/HOME/covid-19-benchmarking/results/{self.model['model_name']}/{self.test}/{self.subpop_name}/{self.model['model_name']}_precision_recall_{self.test}_{self.k}_{self.subpop}.png",dpi=300)

        return precision.tolist(), recall.tolist(), thresholds.tolist(), pr_auc
    
    def calculate_best_threshold(self, fpr, tpr, thresholds) -> float:
        assert self.best_threshold == None
        J = np.array(tpr) - np.array(fpr)
        ix = np.argmax(J)
        best_threshold = thresholds[ix]     

        return best_threshold


    def generate_cm(self, gt_list, output_list, plot=True):
        cm = confusion_matrix(gt_list, output_list,labels = [True, False])
        print(cm)
        print(self.model['model_name'])
        print(self.best_threshold)
        
        if plot == True:
            self.plot_confusion_matrix(cm, title=f"{self.model['model_name'].upper()}: CONFUSION MATRIX (k{self.k})")
            


        tn, fp, fn, tp = cm.flatten()

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)
        acc = (tp + tn) / (tp + fp + tn + fn)

        if plot == True:
            self.plot_confusion_matrix(cm, title=f"{self.model['model_name'].upper()}: CONFUSION MATRIX (k{self.k})")


        return precision, recall, f1, acc


    def plot_confusion_matrix(self, cm,
                            target_names=["NEGATIVE","POSITIVE"],
                            title='Residual Attn: Confusion matrix (k1)',
                            cmap=None,
                            normalize=False):
        """
        given a sklearn confusion matrix (cm), make a nice plot

        Arguments
        ---------
        cm:           confusion matrix from sklearn.metrics.confusion_matrix

        target_names: given classification classes such as [0, 1, 2]
                    the class names, for example: ['high', 'medium', 'low']

        title:        the text to display at the top of the matrix

        cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                    see http://matplotlib.org/examples/color/colormaps_reference.html
                    plt.get_cmap('jet') or plt.cm.Blues

        normalize:    If False, plot the raw numbers
                    If True, plot the proportions

        Usage
        -----
        plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                                # sklearn.metrics.confusion_matrix
                            normalize    = True,                # show proportions
                            target_names = y_labels_vals,       # list of names of the classes
                            title        = best_estimator_name) # title of graph

        Citiation
        ---------
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

        """

        if cmap is None:
            cmap = plt.get_cmap('Greens')

        with plt.style.context(['science', 'nature']):
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(cm, interpolation='nearest', cmap=cmap)
            ax.set_title(title, fontsize=14)
           # ax.colorbar()

            if target_names is not None:
                tick_marks = np.arange(len(target_names))
                ax.set_xticks(tick_marks, target_names, fontsize=14, rotation=45)
                ax.set_yticks(tick_marks, target_names,fontsize=14,rotation=45)

            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            thresh = cm.max() / 1.5 if normalize else cm.max() / 2
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                if normalize:
                    ax.text(j, i, "{:0.4f}".format(cm[i, j]),
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black")
                else:
                    ax.text(j, i, "{:,}".format(cm[i, j]),fontsize=16,
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black")

            ax.set_ylabel('True label',fontsize=14)
            ax.set_xlabel('Predicted label',fontsize=14)

            fig.savefig(f"/MULTIX/DATA/HOME/covid-19-benchmarking/results/{self.model['model_name']}/{self.test}/{self.subpop_name}/{self.model['model_name']}_cm_{self.test}_{self.k}_{self.subpop}.png")


    def __call__(self, dataloader, plot_curves) -> dict:
        ## k-wise model evaluations
        results_dict = {}
        prediction_data = {}
        gt_list, pred_list, output_list, best_threshold, path_list = self.generate_predictions(dataloader, True)
#        path_list = [x.decode() for x in path_list]
        print(path_list)
        if self.model['model_name'] == 'mag_sd' or  self.model['model_name']=='capsnet':
            gt_label = np.array([np.argmax(i) for i in gt_list]).flatten()
            #misclass_df = self.misclassification_analysis(gt_list, output_list,path_list, self.k)
#            self.plot_misclassified_attributes(misclass_df)
            precision, recall, f1, acc = self.generate_cm(gt_label, output_list, plot=plot_curves)
        else:
          #  misclass_df = self.misclassification_analysis(gt_list, output_list,path_list)
          #  self.plot_misclassified_attributes(misclass_df)
            precision, recall, f1, acc = self.generate_cm(gt_list, output_list, plot=True)
        
        fpr, tpr, roc_thresholds, roc_auc = self.create_roc_curve(gt_list, pred_list, plot=plot_curves)
        prec, rec, prec_thresholds, prec_auc = self.create_precision_recall_curve(gt_list, pred_list, plot=plot_curves)


        #        | precision | recall | ... | 
        # -------|---------------------------
        # k1     |

        results_dict['precision'] = precision
        results_dict['recall']= recall
        results_dict['f1'] = f1
        results_dict['acc'] = acc
        results_dict['tpr'] = tpr
        results_dict['fpr'] = fpr
        results_dict['roc_thresholds'] = roc_thresholds
        results_dict['roc_auc'] = roc_auc
        results_dict['best_threshold'] = best_threshold

        results_dict['precision_curve'] = prec
        results_dict['recall_curve'] = rec
        results_dict['prec_thresholds'] = prec_thresholds
        results_dict['prec_auc'] = prec_auc

        prediction_data['gt'] = gt_list.tolist()
        prediction_data['pred'] = output_list.tolist()
        prediction_data['pred_prob'] = pred_list.tolist()
        prediction_data['id'] = path_list.tolist()

        if self.subpop == None:
          return results_dict, prediction_data
        else:
          return results_dict, prediction_data
        

def plot_k_fold_performance(model, mean_x, y, aucs, test, name, subpop_name, subpop, fig, ax, n, last):
        # model = CovidCaps(pretrained=True)
        # model = model.build_model((480,480,3),16)
#        model = model.build_model()
        mean_tpr = np.mean(y, axis=0)
        mean_auc = auc(mean_x, mean_tpr)
        std_auc = np.std(aucs)
        
        if name == 'roc':
            label_metric = 'AUC'
        else:
            label_metric = 'AP'
        
        if subpop == 'All':
                ax.plot(mean_x, mean_tpr, 
                            label='%s = %0.2f $\pm$ %0.2f (n = %d)' % (label_metric, mean_auc, std_auc, n),
                            lw=1, alpha=.8)
        elif subpop != '100-124':
                ax.plot(mean_x, mean_tpr, 
                            label='%s: %s = %0.2f $\pm$ %0.2f (n = %d)' % (subpop, label_metric, mean_auc, std_auc, n),
                            lw=1, alpha=.8)
        
        if name == 'roc':
            mean_tpr[-1] = 1.0
            if last == True:
                ax.plot([0, 1], [0, 1], linestyle='--', lw=0.75, color='r',
                label='Chance', alpha=.6)
                
        std_tpr = np.std(y, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

        if subpop != '100-124':
          ax.fill_between(mean_x, tprs_lower, tprs_upper,
                 alpha=.1)
 
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])

        if name == 'roc':
  #          ax.set_title(f"Cross-Validation ROC of {model['model_name'].upper()}",fontsize=8)
            ax.legend(loc='lower right', fontsize=4.5)
            ax.set_xlabel('False Positive Rate',fontsize=7)
            ax.set_ylabel('True Positive Rate',fontsize=7)

        else:
   #         ax.set_title(f"Cross-Validation Precision-Recall Curve of {model['model_name'].upper()}",fontsize=8)
            ax.legend(loc='lower left', fontsize=4.5)
            ax.set_xlabel('Precision',fontsize=7)
            ax.set_ylabel('Recall',fontsize=7)
        
            
        fig.savefig(f"/MULTIX/DATA/HOME/covid-19-benchmarking/results/{model['model_name']}/{test}/{subpop_name}/{model['model_name']}_{test}_{name}_{subpop_name}.pdf",dpi=300)
        

def k_fold_eval(model, weights, ltht, df, test, subpop_name, subpop=None):
    kfold_results_dict = {}
    all_misclass_df = {}
    kfold_prediction_dict = {}

    count_iter = 0
    for pop, df in subpop.items(): 
        count_iter = count_iter + 1

        subpop_misclass_df = []

        subpop_tprs = []
        subpop_roc_aucs = []
        subpop_roc_threshold = []
        subpop_mean_fpr = np.linspace(0, 1, 100)

        subpop_recall_list = []
        subpop_prec_aucs = []
        subpop_prec_thresholds = []
        subpop_mean_prec = np.linspace(1, 0, 100)

        subpop_kfold_results_dict = {'precision':[],'recall':[],'f1':[],'acc':[], 'tpr':[], 'fpr': [], 'roc_thresholds': [],
            'roc_auc':[],'precision_curve': [],'recall_curve':[],'prec_thresholds':[],'prec_auc':[] , 'best_threshold':[], 'interp_tpr':[], 'interp_rec':[]}

        subpop_kfold_prediction_dict = {'gt':[],'pred':[],'pred_prob':[],'id':[]}

        for k in range(1,6):
            model_performance_eval = ModelPerformanceEval(model, weights, df, k, test, ltht, subpop_name, pop)
            params = {'batchsize':24, "num_workers":4, "k":1}
            
            if test == 'nccid_val':
                df_val = df[df[f'kfold_{k}'] == 'val']

   #             model = CovidCaps(pretrained=True)
                _, _, test_loader = make_generators(model, df_val, df_val, df_val, params)
            else:
#                model = CovidCaps(pretrained=True)
                _, _, test_loader = make_generators(model, df, df, df, params)            
           
     #       if subpop != None:
            results_dict, prediction_data = model_performance_eval(test_loader, False)  
 #           else:
  #              results_dict, misclass_df, prediction_data = model_performance_eval(test_loader, True)
   #             misclass_df['kfold'] = k
    #            subpop_misclass_df.append(misclass_df)
                
         #   else:
          #      results_dict, prediction_data = model_performance_eval(test_loader, False)         

            fpr = results_dict['fpr']
            tpr = results_dict['tpr']
            roc_auc = results_dict['roc_auc']
            thresholds = results_dict['roc_thresholds']

            subpop_tprs.append(interp(np.array(subpop_mean_fpr).flatten(), np.array(fpr).flatten(), np.array(tpr).flatten()).tolist())
            subpop_tprs[-1][0] = 0.0
            subpop_roc_aucs.append(roc_auc)
            subpop_roc_threshold.append(thresholds)

            prec = results_dict['precision_curve']
            rec = results_dict['recall_curve']
            prec_auc = results_dict['prec_auc']
            thresholds = results_dict['prec_thresholds']

            subpop_prec_aucs.append(prec_auc)
            subpop_recall_list.append(interp(np.array(subpop_mean_prec).flatten(), np.array(prec).flatten(), np.array(rec).flatten()).tolist())
            subpop_recall_list[-1][0] = 0.0
            subpop_prec_thresholds.append(thresholds)
            
            subpop_kfold_results_dict['interp_tpr'].append(interp(np.array(subpop_mean_fpr).flatten(), np.array(fpr).flatten(), np.array(tpr).flatten()).tolist())
            subpop_kfold_results_dict['interp_rec'].append(interp(np.array(subpop_mean_prec).flatten(), np.array(prec).flatten(), np.array(rec).flatten()).tolist())
            
            for key, val in results_dict.items():
                subpop_kfold_results_dict[key].append(val)

            for key, val in prediction_data.items():
                subpop_kfold_prediction_dict[key].append(val)

        if subpop == None:
            all_misclass_df[pop] = pd.concat(subpop_misclass_df)

        best_k = np.argmax(np.array(subpop_kfold_results_dict['f1'])) 
            
        with plt.style.context(['science', 'nature']):
            if count_iter == 1:
                fig_roc, ax_roc = plt.subplots()
                
            if count_iter == len(subpop):
                last = True
            else:
                last = False

#                model = CovidCaps(pretrained=True)                  
            plot_k_fold_performance(model, subpop_mean_fpr, subpop_tprs, subpop_roc_aucs, test, 'roc', subpop_name, pop, fig_roc, ax_roc, len(df), last)
            
            if count_iter == 1:
                fig_pr, ax_pr = plt.subplots()
#                model = CovidCaps(pretrained=True)   
 
            plot_k_fold_performance(model, subpop_mean_prec, subpop_recall_list, subpop_prec_aucs, test, 'precision_recall', subpop_name, pop, fig_pr, ax_pr, len(df), last)

        # subpop_kfold_results_df = pd.DataFrame.from_dict(subpop_kfold_results_dict)
        subpop_kfold_results_dict['kfold'] =['kfold1', 'kfold2', 'kfold3', 'kfold4', 'kfold5', f'kfold{best_k+1}']
        subpop_kfold_prediction_dict['kfold'] =['kfold1', 'kfold2', 'kfold3', 'kfold4', 'kfold5', f'kfold{best_k+1}']
        
        kfold_results_dict[pop] = subpop_kfold_results_dict
        kfold_prediction_dict[pop] = subpop_kfold_prediction_dict
    
    kfold_results_df = kfold_results_dict
    kfold_prediction_df = kfold_prediction_dict
    
    # model = CovidCaps(pretrained=True)
    # model = model.build_model((480,480,3),16)
#    model = model.build_model()

    results_path = f"/MULTIX/DATA/HOME/covid-19-benchmarking/results/{model['model_name']}/{test}/{model['model_name']}_performance_{test}_{subpop_name}.json"
    predictions_path = f"/MULTIX/DATA/HOME/covid-19-benchmarking/results/{model['model_name']}/{test}/{model['model_name']}_predictions_{test}_{subpop_name}.json"

    with open(results_path, 'w') as fp:
        json.dump(kfold_results_df, fp)
    
    with open(predictions_path, 'w') as fp:
       json.dump(kfold_prediction_df, fp)
