import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_weights
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import torch
from tqdm import tqdm

def load_weights(model, weights):
    if model['model_type'] == 'keras':
        model['model'].load_weights(weights)
        model['model'].built = True
        model['model'].trainable = False
    
    elif model['model_type'] == 'pytorch':
        model['model'] = model['model'].load_state_dict(torch.load(weights))
        model['model'].eval()
    
    return model

def roc_cm_metrics(model, dataloader, k, precision=False):
        
        gt_list = []
        pred_list = []
        for img, label in tqdm(dataloader):
            pred = model['model'](img)
            pred_list.append(pred)
            gt_list.append(label)

            if precision == False:
                fpr, tpr, thresholds = roc_curve(np.array(gt_list).flatten(), np.array(pred_list).flatten(), drop_intermediate=False)
                roc_auc = auc(fpr, tpr)

                plt.plot(fpr, tpr, lw=1, alpha=0.3,
                        label='ROC fold %d (AUC = %0.2f)' % (k, roc_auc))

                lw=2
                plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f"{model['model_name']}")
                plt.legend(loc="lower right")
                plt.savefig(f"/MULTIX/DATA/nccid/{model['model_name']}_roc_{k}.png")

                J = tpr - fpr
                ix = np.argmax(J)
                best_thresh = thresholds[ix]

                gt_list = np.array(gt_list).flatten()
                pred_list = np.array(pred_list).flatten()
                output_list = np.array([np.int(i>best_thresh) for i in pred_list])

                print(classification_report(gt_list, output_list, target_names=['Negative','Positive'])) # performance metrics
                print(confusion_matrix(gt_list, output_list))
                plt.savefig(f"/MULTIX/DATA/nccid/{model['model_name']}_cm_{k}.png")
                
                J = tpr - fpr
                ix = np.argmax(J)
                best_thresh = thresholds[ix]

                gt_list = np.array(gt_list).flatten()
                pred_list = np.array(pred_list).flatten()
                output_list = np.array([np.int(i>best_thresh) for i in pred_list])

                print(classification_report(gt_list, output_list, target_names=['Negative','Positive'])) # performance metrics
                print(confusion_matrix(gt_list, output_list))
                plt.savefig(f"/MULTIX/DATA/nccid/{model['model_name']}_cm_{k}.png")
                return fpr, tpr, thresholds, best_thresh, roc_auc
            
        elif precision == True:
            pass
