from sklearn.model_selection import KFold
from feature_visualisation import TorchVisuals, KerasExtras, KerasGradCAM, KerasGradient
from classification_metrics import load_weights, roc_cm_metrics
import sys
sys.path.append('../')

from scripts.models.res_attn import AttentionResNetModified
from scripts.models.ecovnet import ECovNet
from scripts.models.coronet_tfl import CoroNet
from scripts.models.fusenet import FuseNet
from scripts.models.coronanet import CoronaNet

from tensorflow.keras.models import Model

from scripts.dataloaders import make_generators

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from scipy import interp
import pandas as pd
import argparse
from tqdm import tqdm

def test_iter(model, test_loader, k, class_metrics=True, feature_vis=True):

    if feature_vis==True:
        data = next(iter(test_loader))
        if model['model_type']=='pytorch':
            print(data.shape)
            torch_visuals = TorchVisuals(data, model)
        
        elif model['model_type']=='keras':
            keras_gc = KerasGradCAM(data, model)
            keras_grad = KerasGradient(data, model)
            keras_extras = KerasExtras(data, model)

            keras_gc.guided_gradcam_plus_fn()
            keras_grad.gradient_visuals()
            keras_extras.all()

    if class_metrics==True:
        fprs, tprs, thresholds, best_threshold = roc_cm_metrics(model, test_loader, k)
    
    return fprs, tprs, thresholds, best_threshold
            
def k_fold_eval(model, df, weights, class_metrics=True, feature_vis=True):

    if model['model_type'] == 'pytorch':
        suffix = '.pth'
    else:
        suffix = '.h5'
    
    params = {'batchsize':4, "num_workers":1, "k":1}
    train_df = df[df['kfold_1'] == "train"]
    val_df = df[df['kfold_1'] == "val"]
    test_df = df[df['kfold_1'] == 'test']
    _, _, test_loader = make_generators(model, train_df, val_df, test_df, params)


    tprs = []
    aucs = []
    threshold_l = []
    mean_fpr = np.linspace(0, 1, 100)
    for k in range(1,6):
        model = load_weights(model, f"{weights}_{k}{suffix}")

        if k > 1:
            feature_vis = False
    
        fpr, tpr, thresholds, best_threshold, roc_auc = test_iter(model, test_loader, k, feature_vis=feature_vis)

        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(roc_auc)
        threshold_l.append(thresholds)

    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate',fontsize=10)
    plt.ylabel('True Positive Rate',fontsize=10)
    plt.title(f"Cross-Validation ROC of {model['model_name']}",fontsize=14)
    plt.legend(loc="lower right", prop={'size': 8})
    plt.savefig(f"/MULTIX/DATA/nccid/{model['model_name']}_roc.png")
    plt.show()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training Script')
    parser.add_argument('--data_csv', default='/MULTIX/DATA/nccid_preprocessed.csv', type=str, help='Path to data file')
    parser.add_argument('--weights')
    args = parser.parse_args()
    
    df = pd.read_csv(args.data_csv)
    mapping = {'negative':0, 'positive':1}
    
    df = df[df['xray_status']!=np.nan]
    df = df.dropna(subset=['xray_status'])
    
    df['xray_status'] = df['xray_status'].map(mapping)

    model = AttentionResNetModified(dropout_act=False).build_model()
    k_fold_eval(model, df, args.weights)
