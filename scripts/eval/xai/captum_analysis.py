import numpy as np 

import skimage 
import pandas as pd

from matplotlib import pyplot as plt 
import tensorflow as tf

from math import ceil 

import cv2 
import argparse 

import torch

#from keras.utils import to_categorical
#%matplotlib inline
#%config InlineBackend.figure_format='retina'
#import torch
#import xplique
#from xplique.plots import plot_attributions
from captum.attr import visualization as viz
import torch.nn as nn
#from captum.attr.visualization import visualize_image_attr

import sys 
sys.path.append('../')

import torch.nn.functional as F
from captum.attr import GuidedBackprop, IntegratedGradients, LayerGradCam, Saliency
from models.ssl_am2 import SSL_AM
from models.mag_sd import MAG_SD, config
from models.coronet import CoroNet
from models.xvitcos import xVitCOS
from dataloaders import make_generators
import torchvision
from torchvision.utils import make_grid

#plt.style.use("dark_background")

def plot_attributions(attr, original_image, fig, ax):
    print(attr.shape)
    print(original_image.shape)
    # visualize_image_attr_multiple attr = np.transpose(attr.squeeze().cpu().detach().numpy(), (1,2,0))
    #original_image = np.transpose(origina_image.squeeze().cpu().detach().numpy(), (1,2,0))
    results = []
    for i in range(8):
        ax[i].imshow(np.transpose(original_image.cpu().detach().numpy()[i], (1,2,0)))
        ax[i].imshow(skimage.transform.resize(np.transpose(attr.cpu().detach().numpy()[i], (1,2,0)), (480,480)), alpha=0.4, cmap='jet')
        ax[i].axis("off")
        #viz.visualize_image_attr(np.transpose(attr.cpu().detach().numpy()[i], (1,2,0)), np.transpose(original_image.cpu().detach().numpy()[i], (1,2,0)),plt_fig_axis=(fig,ax[i]), method="blended_heat_map", alpha_overlay=0.4, cmap='jet')
#    print(results[0].shape)

def xplique():
    for explainer in explainers:
      explanations = explainer.attribute(X_preprocessed, Y)
      print(f"Method: {explainer.__class__.__name__}")
      plot_attributions(explanations, X_preprocessed)
      plt.savefig(path)
      print("\n")


if __name__ == "__main__":
   ## args parse for data, threshold etc.
    parser = argparse.ArgumentParser(description='Model Training Script')
    parser.add_argument('--data_csv', default='/MULTIX/DATA/HOME/covid-19-benchmarking/data/nccid_preprocessed14_21.csv', type=str)
    parser.add_argument('--weights', default=None)
    parser.add_argument('--test', default=None, type=str, help='Choose on of: [ltht, nccid_test, nccid_val, nccid_leeds, chexpert, custom]')
    parser.add_argument('--threshold_data', type=str)
    parser.add_argument('--subpop_analysis', default=False, type=bool)
    parser.add_argument('--prevalence_analysis', default=False, type=bool, help='Vary prevalence of COVID in test population')

    args = parser.parse_args()
    
    tf.config.run_functions_eagerly(True)

    df = pd.read_csv(args.data_csv)
    ltht = False

    params = {'batchsize':8, "num_workers":1, "k":1}

    if args.test == 'ltht':
        df['xray_status'] = df['FinalPCR']
        df['cxr_path'] = df['path']
        df['kfold_1'] = 'test'
        ltht = True

        if args.subpop_analysis == True:
            data_dict = create_subpopulations(df, args.test)
        else:
            data_dict = {'All': df}
            data_dict = {'All': data_dict}


    model = CoroNet(pretrained=True, supervised=True).build_model()
    model = model['model']
    model.load_state_dict(torch.load('/MULTIX/DATA/HOME/covid-19-benchmarking/weights/coronet/coronet_supervised_1.pth'))

    model.eval()
    batch_size = 8
#    print(model.layers)
    explainers = [
             #Saliency(model, target=1),
#             GradientInput(model),
             GuidedBackprop(model)]
#             IntegratedGradients(model),
             #LayerGradCam(model, layer=list(model.modules())[-3])]
            # SmoothGrad(model, nb_samples=80, batch_size=batch_size),
            # SquareGrad(model, nb_samples=80, batch_size=batch_size),
            # VarGrad(model, nb_samples=80, batch_size=batch_size),
            # GradCAM(model),
            # GradCAMPP(model),
            # Occlusion(model, patch_size=10, patch_stride=5, batch_size=batch_size),
            # Rise(model, nb_samples=4000, batch_size=batch_size)]
    df = data_dict['All']['All']
    df = df[df['xray_status']==1.0] 
    _, _, test_loader = make_generators(model.build_model(), df, df, df, params)       
    
    data_batch = next(iter(test_loader))
    print(data_batch[1].shape)

    for i, (name, layer) in enumerate(model.named_modules()):
      if isinstance(layer, nn.ReLU):
          layer = nn.ReLU(inplace=False)
#  cmap='jet', alpha=0.4, cols=8, absolute_value=True, clip_percentile=0.5)   
    for explainer in explainers:
      #y = tf.keras.utils.to_categorical(data_batch[1], 2)
      x = data_batch[0]
      print(x.shape)
      for y in data_batch[1]:
          print(y)
      y = torch.tensor(data_batch[1], dtype=torch.int64)
#      y = torch.tensor(np.array([[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]]))
      #y = torch.tensor([torch.nn.functional.one_hot(y,1) for y in batchy]) #torch.ones(8,1)]) #for y in data_batch[1]])
#      y = torch.eye(2) 
#      y = y[data_batch[1]] 
  #    y = torch.tensor([i.unsqueeze(0) for i in batchy])      #y = torch.tensor([[0,1]*8], dtype=torch.float, requires_grad=True) #F.one_hot(data_batch[1], 2)
      #print(y)
      #explanations = explainer.attribute(X_preprocessed, Y)
      print(f"Method: {explainer.__class__.__name__}")
      #plot_attributions(explanations, X_preprocessed)
      pred = model(x)
#      pred = torch.tensor([torch.tensor([i, 1-i]) for i in pred])
      explanations = explainer.attribute(x, None) #, absolute_value=False) #, return_convergence_delta=False, internal_batch_size=1) 
      explanations =  explanations*100     
      print(explanations)
      #explanations = np.clip(explanations, 0.9)
      print('plot')
      fig, ax  = plt.subplots(1,8, figsize=(4 * 8, 4), tight_layout=True)
    # get width and height of our images
      l_width, l_height = 480, 480 #explanations.shape[1:]
      img_size = 2.
    # define the figure margin, width, height in inch
      margin = 0.3
      spacing = 0.3
      cols = 8
      rows =1
      figwidth = cols * img_size + (cols-1) * spacing + 2 * margin
      figheight = rows * img_size * l_height/l_width + (rows-1) * spacing + 2 * margin

      left = margin/figwidth
      bottom = margin/figheight

     # fig = plt.figure()
 #     fig.set_size_inches(figwidth, figheight)

#      fig.subplots_adjust(
#3       left = left,
#        bottom = bottom,
#        right = 1.-left,
#        top = 1.-bottom,
#        wspace = spacing/img_size,
#        hspace= spacing/img_size * l_width/l_height
 #     )
      plot_attributions(explanations, x, fig, ax)
      #print([np.max(np.array(e).flatten()) for e in explanations])
      print(explanations.shape)
      
      print(f"Method: {explainer.__class__.__name__}")
      
      #plt.imshow(grid)
      #plot_attributions(explanations, x, img_size=2., cmap='jet', alpha=0.4,
       #             cols=8, absolute_value=True, clip_percentile=0.5)
      plt.savefig(f'/MULTIX/DATA/HOME/covid-19-benchmarking/results/coronet/ltht/covid_pos_{explainer.__class__.__name__}.pdf', dpi=300)
      print("\n")
