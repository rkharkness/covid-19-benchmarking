from xplique.attributions import Saliency, GradientInput, IntegratedGradients, SmoothGrad, VarGrad, SquareGrad, GradCAM, Occlusion, Rise, GuidedBackprop, GradCAMPP, Lime, KernelShap 
import numpy as np

import pandas as pd 
from matplotlib import pyplot as plt 
import tensorflow as tf 
from math import ceil 

import cv2 
import argparse
#from keras.utils import to_categorical
#%matplotlib inline
#%config InlineBackend.figure_format='retina'
#import torch
import xplique
from xplique.plots import plot_attributions
from tqdm import tqdm

import sys 
sys.path.append('../')


from models.ssl_am_seg import SSL_AM as SSL_AM_Seg
from models.ssl_am2 import SSL_AM

from models.coronet_tfl import CoroNet
from models.coronet_tfl_seg import CoroNet as CoroNet_Seg

from models.xvitcos import xVitCOS as xVitCOS
from models.xvitcos_seg import xVitCOS as xVitCOS_Seg
from dataloaders import make_generators

     
def explain_fn(model, seg_model, test_loader, seg_test_loader):
  batch_size = 1
  explainers = [
           GuidedBackprop(model['model'], batch_size=batch_size),
           GradCAM(model['model'])]
  seg_explainers = [
           GuidedBackprop(seg_model['model'], batch_size=batch_size),
           GradCAM(seg_model['model'])]
  idx = 0
  for data, seg_data in tqdm(zip(test_loader,seg_test_loader)):
      x, y = data[0], data[1]
      seg_x, seg_y = seg_data[0],seg_data[1]
      assert y == seg_y
      pred_proba = model['model'](x)
      pred_proba_seg = seg_model['model'](seg_x)

      pred = np.round(pred_proba) # thresh = 0.5  
      pred_seg = np.round(pred_proba_seg) # thresh = 0.5

      for pr, pr_s, y_i in zip(pred, pred_seg, y):
        if pr == 1.0 and pr_s == 0.0:       
            print('True')
            for seg_explainer, explainer in zip(seg_explainers,explainers):
                y_cat = tf.keras.utils.to_categorical(y, num_classes=2)
                explanations = explainer.explain(x, y_cat)
                seg_explanations = seg_explainer.explain(seg_x, y_cat)

                plt.figure(figsize=(10, 10))
                plot_attributions(explanations, x, img_size=2., cmap='jet', alpha=0.6,
                        cols=1, absolute_value=True, clip_percentile=0.5)
                plt.savefig(f"/MULTIX/DATA/HOME/covid-19-benchmarking/results/{model['model_name']}/ltht/feat_vis/full_{model['model_name']}_fulltp_segfn_{explainer.__class__.__name__}_{idx}.pdf", dpi=300)
                
                plt.figure(figsize=(10, 10))
                plot_attributions(seg_explanations, seg_x, img_size=2., cmap='jet', alpha=0.6,
                        cols=1, absolute_value=True, clip_percentile=0.5)
                plt.savefig(f"/MULTIX/DATA/HOME/covid-19-benchmarking/results/{model['model_name']}/ltht/feat_vis/seg_{model['model_name']}_fulltp_segfn_{explainer.__class__.__name__}_{idx}.pdf", dpi=300)

                idx += 1

if __name__ == "__main__":
   ## args parse for data, threshold etc.
    parser = argparse.ArgumentParser(description='Model Training Script')
    parser.add_argument('--data_csv', default='/MULTIX/DATA/INPUT/ltht_dcm_binary_data14_21.csv', type=str, help='Path to data file')
    args = parser.parse_args()
    
    tf.config.run_functions_eagerly(True)

    df = pd.read_csv(args.data_csv)

    seg_df = df.copy()
    paths = seg_df['paths'].values #0,/MULTIX/DATA/INPUT/ltht_dcm_seg/LTH20029_CovidX_CovidX_DeIdentified_20200916_01D945F8A05C4371E3DC229503D729F024AB>
# ltht - MULTIX/DATA/INPUT/LTH20029_CovidX/CovidX_DeIdentified_20200916/01D945F8A0
    new_paths = [i.split('/')[4:] for i in paths]
    join_new_path = ['_'.join(i) for i in new_paths]
    final_new_path = ['/MULTIX/DATA/INPUT/ltht_dcm_seg/' + i for i in join_new_path]
    seg_df['cxr_path'] = final_new_path 
    seg_df['xray_status'] = seg_df['FinalPCR']  
  
    df['xray_status'] = df['FinalPCR']
    df['cxr_path'] = df['path']
    df['kfold_1'] = 'test'

    ltht = True
    
    params = {'batchsize':1, "num_workers":1, "k":1}
    model = CoroNet(dropout_act=True)
    model = model.build_model()
    model['model'].load_weights('/MULTIX/DATA/HOME/covid-19-benchmarking/weights/coronet_tfl/coronet_tfl_l_supervised_1.h5')

    seg_model = CoroNet_Seg(dropout_act=True)
    seg_model = seg_model.build_model()
    seg_model['model'].load_weights('/MULTIX/DATA/HOME/covid-19-benchmarking/weights/coronet_tfl_seg/coronet_tfl_seg_supervised_1.h5')

    seg_df = seg_df[seg_df['xray_status']==1.0] 
    seg_df = seg_df[:1500]  

    df = df[df['xray_status']==1.0] 
    df = df[:1500]  
  
    _, _, test_loader = make_generators(model, df, df, df, params)
    print(seg_df['cxr_path'].values)
    _, _, seg_test_loader = make_generators(model, seg_df, seg_df, seg_df, params)
    model['model'].layers[-1].activation = tf.keras.activations.linear
    seg_model['model'].layers[-1].activation = tf.keras.activations.linear    
  
    explain_fn(model, seg_model, test_loader, seg_test_loader)
