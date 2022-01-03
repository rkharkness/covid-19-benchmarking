%% Clear environment variables
clc;clear all;close all;

%% Sample generation
training_labels=csvread('training_labels.csv');%1: COVID-19, 0: Normal
training_features=csvread('training_features.csv');
training_set=[training_labels training_features];%Training set

testing_labels=csvread('verlabels.csv');
testing_features=csvread('verfeatures.csv');
testing_set=[testing_labels testing_features];%Testing set

%Checking
[row_training,column_training]=size(training_set);
[row_testing,column_testing]=size(testing_set);

%Calculate the numbers of COVID-19 and normal separately and make label
COVID_trainnum=0;normal_trainnum=0;
for i=1:row_training
    if training_set(i,1)==1
        COVID_trainnum=COVID_trainnum+1;
        Ytrain_pre(i,:)=[1 -1];
        Ltrain_pre(i,:)=[1 0];
    else
        normal_trainnum=normal_trainnum+1;
        Ytrain_pre(i,:)=[-1 1];
        Ltrain_pre(i,:)=[0 1];
    end
end

COVID_testnum=0;normal_testnum=0;
for i=1:row_testing
    if testing_set(i,1)==1
        COVID_testnum=COVID_testnum+1;
        Ytest(i,:)=[1 -1];
        Ltest(i,:)=[1 0];
    else
        normal_testnum=normal_testnum+1;
        Ytest(i,:)=[-1 1];
        Ltest(i,:)=[0 1];
    end
end

%10-fold Validation (Randomly sample for bagging)
COVID_training=training_set(1:COVID_trainnum,:);
normal_training_pre=training_set(COVID_trainnum+1:row_training,:);
%Randomly select 1000 from the total 1368
normal_training=normal_training_pre(randperm(normal_trainnum, 1000),:); 
Ytrain=Ytrain_pre(1:1736,:);
Ltrain=Ltrain_pre(1:1736,:);

fold1=[COVID_training(1:74,:);normal_training(1:100,:)];
fold2=[COVID_training(75:148,:);normal_training(101:200,:)];
fold3=[COVID_training(149:222,:);normal_training(201:300,:)];
fold4=[COVID_training(223:296,:);normal_training(301:400,:)];
fold5=[COVID_training(297:370,:);normal_training(401:500,:)];
fold6=[COVID_training(371:444,:);normal_training(501:600,:)];
fold7=[COVID_training(445:518,:);normal_training(601:700,:)];
fold8=[COVID_training(519:592,:);normal_training(701:800,:)];
fold9=[COVID_training(593:666,:);normal_training(801:900,:)];
fold10=[COVID_training(667:736,:);normal_training(901:1000,:)];

save ('training_set','training_set');
save ('testing_set','testing_set');
save ('Ytrain_pre','Ytrain_pre');
save ('Ltrain_pre','Ltrain_pre');
save ('Ytrain','Ytrain');
save ('Ltrain','Ltrain');
save ('Ytest','Ytest');
save ('Ltest','Ltest');

save ('fold1','fold1');
save ('fold2','fold2');
save ('fold3','fold3');
save ('fold4','fold4');
save ('fold5','fold5');
save ('fold6','fold6');
save ('fold7','fold7');
save ('fold8','fold8');
save ('fold9','fold9');
save ('fold10','fold10');
