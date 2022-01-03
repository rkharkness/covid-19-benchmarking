%% Load dataset
clc;clear all;close all;
load('Result6_1.mat');%Load hyper-parameters
load('testing_set.mat');
load('Ytest.mat');
load('Ltest.mat');
load('fold1.mat');load('fold2.mat');load('fold3.mat');load('fold4.mat');load('fold5.mat');
load('fold6.mat');load('fold7.mat');load('fold8.mat');load('fold9.mat');load('fold10.mat');
training_set=[fold1;fold2;fold3;fold4;fold5;fold6;fold7;fold8;fold9;fold10];

for i=1:size(training_set,1)
    if training_set(i,1)==1
        Ytrain(i,:)=[1 -1];
        Ltrain(i,:)=[1 0];
    else
        Ytrain(i,:)=[-1 1];
        Ltrain(i,:)=[0 1];
    end
end

%% The preparation of training
max_epoch=1000;%The maxinum of training rounds
w1=w1_init;
w2=w2_init;

Qtrain=zeros(size(training_set,1),hidenum);
Qtest=zeros(size(testing_set,1),hidenum);

%% The process of training
trainstart=cputime;
for ii=1:max_epoch
    %Forward
    hi_train=soft(training_set(:,2:513)*w1);
    for i=1:hidenum
        Qtrain(:,i)=hi_train(:,i).^(i-1);
    end
    ytrain=soft(Qtrain*w2);
    errortrain=ytrain-Ytrain;
    
    %Classification
    P1train=softmaxs(ytrain(:,1),ytrain(:,1),ytrain(:,2));
    P2train=softmaxs(ytrain(:,2),ytrain(:,1),ytrain(:,2));
    Etrain_linear(ii)=-sum(Ltrain(:,1).*log(P1train)+Ltrain(:,2).*log(P2train));%Cross Entropy
    for k=1:size(training_set,1)
        if P1train(k)>P2train(k)
            label_train(k,1)=1;
        else
            label_train(k,1)=0;
        end
    end
    gaptrain=label_train-training_set(:,1);
    train_acc(ii,1)=length(find(gaptrain==0))./size(training_set,1)*100;
    
    % Testing
    hi_test=soft(testing_set(:,2:513)*w1);
    for i=1:hidenum
        Qtest(:,i)=hi_test(:,i).^(i-1);
    end
    ytest=soft(Qtest*w2);

    %Classification
    P1test=softmaxs(ytest(:,1),ytest(:,1),ytest(:,2));
    P2test=softmaxs(ytest(:,2),ytest(:,1),ytest(:,2));
    Etest_linear(ii)=-sum(Ltest(:,1).*log(P1test)+Ltest(:,2).*log(P2test));%Cross Entropy
    for k=1:size(testing_set,1)
        if P1test(k)>P2test(k)
            label_test(k,1)=1;
        else
            label_test(k,1)=0;
        end
    end
    gaptest=label_test-testing_set(:,1);
    test_acc(ii,1)=length(find(gaptest==0))./size(testing_set,1)*100;
    
    %Quit training
    if Etrain_linear(ii)./size(training_set,1)<Etrain(position)./size(traindata,1)
        break;
    end
    
    %Iteration of w2
    w2=pinv(Qtrain)*softinv(ytrain-alpha.*tanh(errortrain));
    
end
traintime=cputime-trainstart;

test_acc=test_acc(ii);
error_index=find(gaptest~=0);% Search the index of wrong sample

figure(1);
plot(1:ii,Etrain_linear(1:ii)./size(training_set,1),'b','Linewidth',2);
hold on;
plot(1:ii,Etest_linear(1:ii)./size(testing_set,1),'r','Linewidth',2);
grid on;
xlabel('\it k');
ylabel('Error');
legend('Training','Testing');
