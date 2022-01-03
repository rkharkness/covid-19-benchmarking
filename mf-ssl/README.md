# Multi-Feature-Semi-Supervised-Learning_COVID-19 (Pytorch)


## Introduction ##

This is the code to repoduce the study of [Multi-Feature Semi-Supervised Learning for COVID-19 Diagnosis from Chest X-ray Images](https://arxiv.org/abs/2104.01617). Please cite our study if you are using this dataset and referring to our method.

## Network Structure ##

![image](https://user-images.githubusercontent.com/31194584/114582499-a2582580-9c4e-11eb-88df-ae9dbc6f3fed.png)

## Result ##

- Test-1

Method | Labeled Sample (%) | Precision  | Recall | F1-Scores | Top-1(%)
------ | ------------------ | ---------- | ------ | --------- |-------- 
MF-TS  | 30 | 0.94  | 0.94 | 0.94 | 93.61

- Test-2

Method | Labeled Sample (%) | Precision  | Recall | F1-Scores | Top-1(%)
------ | ------------------ | ---------- | ------ | --------- |-------- 
MF-TS  | 30 | 0.93  | 0.94 | 0.93 | 92.47

## Usage ##

- Dataset and Trained model weights:
  - Download them from [Kaggle](https://www.kaggle.com/endiqq/largest-covid19-dataset?select=covid_metadata.csv). CXR folder are all origianl CXR images and Enh folder are all corresponding enhanced images. All weights are in the weight folder

- Preparation:
  - Create a folder to save all downloaded files from this repo and files from Kaggle in one folder. Please modify the coloumn of both test_ds.txt and additional_test_ds.txt to the directory where you create the folder

- Test:
  - CXR-TS: python Test.py --action=retest --dataset=test_ds/additional_test_ds --per_teacher=0.1/0.2/0.3 (test_ds=Test-1; additiona_test_ds=Test-2; 0.1=10% labeled samples etc.)
  - Enh-TS: python Test.py --action=retest --dataset=test_ds/additional_test_ds --type=Enh --per_teacher=0.1/0.2/0.3 (type = image type; default = CXR)
  - MF-T: python Test.py --action=retest_based_both --dataset=test_ds/additional_test_ds --per_teacher=0.1/0.2/0.3
  - MF-TS: python Test.py --action=latefusion_retest_2models --dataset=test_ds/additional_test_ds --per_teacher=0.1/0.2/0.3




