clc;clear all;close all;
timeA=[];timeB=[];
for i=1:10
    strA=strcat('result',num2str(i),'.mat');
    load(strA,'traintime');
    timeA(i)=traintime;  
    clear traintime;
    strB=strcat('result',num2str(i),'_1.mat');
    load(strB,'traintime');
    timeB(i)=traintime;
end
clear traintime;
load('resultAlinear','traintime');
timeA(i+1)=traintime;
clear traintime;
load('resultBlinear','traintime');
timeB(i+1)=traintime;
totalAlinear=sum(timeA);
totalBlinear=sum(timeB);
