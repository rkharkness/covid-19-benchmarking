timeA=[];timeB=[];
for i=1:10
    strA=strcat('Result',num2str(i),'.mat');
    load(strA,'traintime');
    timeA(i)=traintime;  
    clear traintime;
    strB=strcat('Result',num2str(i),'_1.mat');
    load(strB,'traintime');
    timeB(i)=traintime;
end
clear traintime;
load('resultAtanh','traintime');
timeA(i+1)=traintime;
clear traintime;
load('resultBtanh','traintime');
timeB(i+1)=traintime;
totalAtanh=sum(timeA);
totalBtanh=sum(timeB);
