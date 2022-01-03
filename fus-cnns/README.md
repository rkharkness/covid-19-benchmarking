## Fus-CNNs_COVID-19 (Pytorch)

# Introduction:

This is the code to repoduce the study of [Chest X-ray image phase features for improved diagnosis of COVID-19 using convolutional neural network](https://link.springer.com/article/10.1007/s11548-020-02305-w#citeas) from IJCAR 2021. Please cite our study if you are using this dataset and referring to our method.

The highest mean accuracy on sub-dataset one of COVID-Ti is 95.57% (+/- 0.3)

# Usage:

- Dataset:

Dataset: [Kaggle](https://www.kaggle.com/endiqq/largest-covid19-dataset). (Please use same fold name CXR_ijcar_mix and Enh_ijcar_mix)

5 fold validation was used to evaludate our method and mean accuracy was reported. The ratio of train:val:test is 60%:20%:20%. The texts files in k5_train_val_test fold are train, validation, and test files are each fold. There are five columns in each text file. First column is subject name, second is root for saving dataset (Please change coresspondingly), third is image name, forth is class (0: normal; 1: pneumonia; 2:COVID-19), and fifth is source dataset. Please refer to [Kaggle](https://www.kaggle.com/endiqq/largest-covid19-dataset) for detailed informaction of source datasets.

- Train

  - Train a mono model:

    python --main.py --action=train or python --main.py --action=train --dataset=Enh_ijcar_mix

    (The default dataset is CXR_ijcar_mix. If you would like to use enhanced images, please add --dataset=Enh_ijcar_mix
    The default network is Res50. You can change to any other investigated networks in our study by adding --network=name of network, such as --network=Alexnet)

  - Train a middle-fusion model:

    python --main.py --action=middlefusion 

  - Train a latefusion model:

    python --main.py --action=latefusion
    
- Test

  - Test a mono model:

    python --main.py --action=test or python --main.py --action=test --dataset=Enh_ijcar_mix
  
  - Test middle fusion model or late fusion model:

    python --main.py --action=test_middlefusion or python --main.py --action=test_latefusion



