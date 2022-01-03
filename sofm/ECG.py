#   Bayley King
#   SOFM MNIST Application 
#   Python 3.7.3
#   Feb 7 2020

####### Decerations #######
from network import Network 
import funcs
from paramsE import *
###########################

train,test = funcs.loadECG()

NN = Network(layers)

trainBool = NN.train(train,max_epochs,no,tau,tauN,sigmaP,trainBool) #Comment out line to run on saved weights
NN.test(test,MapPickle,trainBool,BMUPickle,Weights)
'''
print('saving metrics')
NN.saveMetrics(max_epochs,no,tau,tauN,sigmaP,layers[-1],Metrics,SimParm)
print('saving weights')
NN.saveWeights(Weights)

# Testing, kinda

weight = funcs.threshWeights(Weights,layers,0.4)
#weight = funcs.loadWeights(OutputW,layers) # uncomment to plot true weight map
funcs.plotMetrics(max_epochs,Metrics,OutputMet,tau,tauN,no,sigmaP)
funcs.graphHeatmap(MapPickle,OutputMap)
#funcs.weightPlot(weight,OutputW)
'''