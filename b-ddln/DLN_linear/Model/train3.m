%% Load dataset
clc;clear all;close all;
load('Ytrain.mat');
load('Ltrain.mat');
load('fold1.mat');load('fold2.mat');load('fold3.mat');load('fold4.mat');load('fold5.mat');
load('fold6.mat');load('fold7.mat');load('fold8.mat');load('fold9.mat');load('fold10.mat');

%% The preparation of training
inputnum=512;%Number of neurons in the input layer
hidenum=50;%Number of neurons in the hidden layer
outnum=2;%Number of neurons in the output layer

alpha=0.01;%NDA const
max_epoch=1000;%The maxinum of training rounds
w1=random('Uniform',-1,1,inputnum,hidenum);
w1_init=w1;
w2=random('Uniform',-1,1,hidenum,outnum);
w2_init=w2;

validdata=fold3;
traindata=[fold1;fold2;fold4;fold5;fold6;fold7;fold8;fold9;fold10];
Yvalid=[Ytrain(149:222,:);Ytrain(937:1036,:)];
Ytrain=[Ytrain(1:74,:);Ytrain(737:836,:);
    Ytrain(75:148,:);Ytrain(837:936,:);
    Ytrain(223:296,:);Ytrain(1037:1136,:);
    Ytrain(297:370,:);Ytrain(1137:1236,:);
    Ytrain(371:444,:);Ytrain(1237:1336,:);
    Ytrain(445:518,:);Ytrain(1337:1436,:);
    Ytrain(519:592,:);Ytrain(1437:1536,:);
    Ytrain(593:666,:);Ytrain(1537:1636,:);
    Ytrain(667:736,:);Ytrain(1637:1736,:)];

Lvalid=[Ltrain(149:222,:);Ltrain(937:1036,:)];
Ltrain=[Ltrain(1:74,:);Ltrain(737:836,:);
    Ltrain(75:148,:);Ltrain(837:936,:);
    Ltrain(223:296,:);Ltrain(1037:1136,:);
    Ltrain(297:370,:);Ltrain(1137:1236,:);
    Ltrain(371:444,:);Ltrain(1237:1336,:);
    Ltrain(445:518,:);Ltrain(1337:1436,:);
    Ltrain(519:592,:);Ltrain(1437:1536,:);
    Ltrain(593:666,:);Ltrain(1537:1636,:);
    Ltrain(667:736,:);Ltrain(1637:1736,:)];

max_epoch=1000;

Qtrain=zeros(size(traindata,1),hidenum);
Qvalid=zeros(size(validdata,1),hidenum);

%% The process of training
trainstart=cputime;
for ii=1:max_epoch
    %Forward
    hi_train=soft(traindata(:,2:513)*w1);
    ytrain=soft(hi_train*w2);
    errortrain=ytrain-Ytrain;
    
    %Classification
    P1train=softmaxs(ytrain(:,1),ytrain(:,1),ytrain(:,2));
    P2train=softmaxs(ytrain(:,2),ytrain(:,1),ytrain(:,2));
    Etrain(ii)=-sum(Ltrain(:,1).*log(P1train)+Ltrain(:,2).*log(P2train));%Cross Entropy
    for k=1:size(traindata,1)
        if P1train(k)>P2train(k)
            label_train(k,1)=1;
        else
            label_train(k,1)=0;
        end
    end
    gaptrain=label_train-traindata(:,1);
    train_acc(ii,1)=length(find(gaptrain==0))./size(traindata,1)*100;
    
    %Iteration of w2
    w2=pinv(hi_train)*softinv(ytrain-alpha.*(errortrain));
    
    %Validation
    hi_valid=soft(validdata(:,2:513)*w1);
    yvalid=soft(hi_valid*w2);
    
    %Classification
    P1valid=softmaxs(yvalid(:,1),yvalid(:,1),yvalid(:,2));
    P2valid=softmaxs(yvalid(:,2),yvalid(:,1),yvalid(:,2));
    Evalid(ii)=-sum(Lvalid(:,1).*log(P1valid)+Lvalid(:,2).*log(P2valid));%Cross Entropy
    for k=1:size(validdata,1)
        if P1valid(k)>P2valid(k)
            label_valid(k,1)=1;
        else
            label_valid(k,1)=0;
        end
    end
    gapvalid=label_valid-validdata(:,1);
    valid_acc(ii,1)=length(find(gapvalid==0))./size(validdata,1)*100;
  
end
traintime=cputime-trainstart;
[validbest,position]=max(valid_acc);
