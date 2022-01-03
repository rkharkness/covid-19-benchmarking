import albumentations as A

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import argparse
import cv2
import random

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Multiply
from tensorflow.keras.layers import Lambda

# from keras import optimizers

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from tensorflow.keras.regularizers import l2
from tensorflow.python.keras import backend as K

import tensorflow.keras as keras

from tensorflow.keras.models import load_model

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from sklearn.utils import class_weight
import os

from keras_tuner import HyperModel
import keras_tuner as kt

from keras.models import load_model

from residual_attn_train import residual_block, attention_block, AttentionResNet56, model_builder
from utils.tf_callbacks import recall_m, precision_m, f1_m, PlotLosses

from skimage.segmentation import mark_boundaries

from deep_viz_keras import utils

from tensorflow.python.keras.backend import eager_learning_phase_scope
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc


try:
    import lime
except:
    sys.path.append(os.path.join('..', '..')) # add the current directory
    import lime
from lime import lime_image



class KerasDropoutPrediction(object):
  """Sourced from: https://stackoverflow.com/questions/43529931/how-to-calculate-prediction-uncertainty-using-keras.
  Puts dropout layer in training mode."""
  def __init__(self,model):
    self.model = model
    self.f = K.function([self.model.layers[0].input],      
                              [self.model.output])
  def predict(self,x, n_iter=10):
    with eager_learning_phase_scope(value=1): # 0=test, 1=train
      Yt_hat = np.array([self.f((x))[0] for _ in range(n_iter)])
    
    # result = []
    # for _ in range(n_iter):
    #   result.append(self.f([x , 1]))

    # result = np.array(result).reshape(n_iter,len(x)).T
    return Yt_hat

def lime(img, model):
    explanation = explainer.explain_instance(images[0].astype('float64'), model.predict, top_labels=5, hide_color=0, num_samples=1000)

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
    plt.imshow(mark_boundaries(temp, mask))
    plt.savefig("/MULTIX/DATA/HOME/lime_pos_5.png")

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
    plt.imshow(mark_boundaries(temp, mask))
    plt.savefig("/MULTIX/DATA/HOME/lime_neg_10.png")

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=1000, hide_rest=False, min_weight=0.1)
    plt.imshow(mark_boundaries(temp, mask))
    plt.savefig("/MULTIX/DATA/HOME/lime_neg_1000_min_weight.png")

    temp, mask = explanation.get_image_and_mask(106, positive_only=True, num_features=5, hide_rest=True)
    plt.imshow(mark_boundaries(temp, mask))
    plt.savefig("/MULTIX/DATA/HOME/lime_pos_5.png")


def guided_backprop(img, model):
    guided_bprop = utils.GuidedBackprop(model)
    mask = guided_bprop.get_mask(img) # compute the gradients
    show_image(mask)
    plt.savefig("/MULTIX/INPUT/DATA/guided_bprop.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Residual Attention Net Testing Script')

    parser.add_argument('--bs', default=16, type=int, help='Batch size')
    parser.add_argument('--img_size', default=480, type=int, help='Image size')
    parser.add_argument('--img_channel', default=1, type=int, help='Image channel')
    parser.add_argument('--data_csv', default='/MULTIX/DATA/INPUT/binary_data.csv', type=str, help='Path to data csv file')
    parser.add_argument('--save_dir', default='/home/ubuntu/', type=str, help='Name of folder to store training checkpoints')
    parser.add_argument('--data_dir', default='/MULTIX/DATA/INPUT/binary_data/', type=str, help='Path to data folder')
    parser.add_argument('--weights_path', default='residual_attn', help='Core filename for saved weights (excl. _fold and .h5')
    parser.add_argument('--savefile', default='residual_attn', help='Filename for saved weights')


    args = parser.parse_args()

    def get_model_name(k):
      return args.savefile + str(k)+'.h5'

    total_data = pd.read_csv(args.data_csv, dtype=str)
    print(total_data['finding'].value_counts())

    test_df = total_data[total_data['split']=='test']
    test_df = test_df.reset_index(drop=True)

    explainer = lime_image.LimeImageExplainer()

    tuner = kt.Hyperband(hypermodel=model_builder, objective = kt.Objective("val_f1_m", direction="max"),
      max_epochs = 10, project_name=f'res_attn_tuner')

    def transform(image):
      image = image.astype(np.float64)
      return image

    test_datagen=ImageDataGenerator(
                preprocessing_function=transform,
                rescale=1/255.0)

    test_generator=test_datagen.flow_from_dataframe(dataframe=test_df, directory=args.data_dir,
                                                        x_col="structured_path", y_col='finding', class_mode="categorical", target_size=(args.img_size,args.img_size), color_mode='grayscale',
                                                        batch_size=args.bs, shuffle=False)

    # dataframe = val_df, directory=directory,x_col="filename", y_col=target, class_mode="categorical", target_size=target_size, color_mode='grayscale',
                                                        # batch_size=batch_size
    STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0] #get hps
    model = tuner.hypermodel.build(best_hps) # build model with best hps

    model_dropout = KerasDropoutPrediction(model) # put dropout layer in training phase

    cv_accuracy = []
    cv_loss = []

    fold_no = [1,2,3,4,5]
    for i in fold_no:
        filepath = args.save_dir+get_model_name(i)
        print(f"Loading weights from {filepath}")
        model.load_weights(filepath)
        with tf.device('/device:GPU:0'):
            scores = model.evaluate(test_generator, steps=STEP_SIZE_TEST)
            y_pred = model.predict(test_generator)
            yhat = y_pred[:, 1]
            
            y_pred = np.argmax(y_pred, axis=1)
            y_test = test_generator.classes
            matrix = confusion_matrix(y_test, y_pred)
            print(matrix)

            fpr, tpr, thresholds = roc_curve(y_test, yhat, drop_intermediate=False)
            plt.figure()
            plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
            plt.plot(fpr, tpr, marker='.', label='Res Attn')
            
            # axis labels
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()
            plt.savefig(f'/MULTIX/DATA/HOME/res_attn_roc_curve_{i}')

            # prediction under dropout
            y_pred_do = model_dropout.predict(test_generator,n_iter=100) 
            y_pred_do_mean = y_pred_do.mean(axis=1) # mean of d.o predictions 

            plt.figure(figsize=(5,5))
            plt.scatter(y_hat_do_mean , y_hat, alpha=0.1)
            plt.xlabel("The average of dropout predictions")
            plt.ylabel("The prediction without dropout from Keras")
            plt.savefig(f'/MULTIX/DATA/HOME/dropout_predictions_{i}.png')
            plt.show()

            # get model visualisations - bs =  1 for this 
            lime(test_generator[0], model)
            guided_bprop(test_generator[0], model)

            print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')

            cv_accuracy.append(scores[1]*100)
            cv_loss.append(scores[0])



    # == Provide average scores ==
    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(cv_accuracy)):
      print('------------------------------------------------------------------------')
      print(f'> Fold {i+1} - Loss: {cv_loss[i]} - Accuracy: {cv_accuracy[i]}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(cv_accuracy)} (+- {np.std(cv_accuracy)})')
    print(f'> Loss: {np.mean(cv_loss)}')
    print('------------------------------------------------------------------------')



