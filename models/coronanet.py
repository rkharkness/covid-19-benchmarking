import torch
import torch.nn as nn
from torchvision import models


class WeightedBCE():
    def  __init__(self) -> None:
        super().__init__()
        self.weights = {'neg':1.32571275, 'pos':0.80276873}
    
    def forward(self, target, output):
        output = torch.clamp(output,min=1e-10,max=1-1e-10)  
        loss =  self.weights['pos'] * (target * torch.log(output)) + self.weights['neg'] * ((1 - target) * torch.log(1 - output))
        return torch.neg(torch.mean(loss))

class Interpolate(nn.Module):
    def __init__(self, size):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.dummy = 8

    def forward(self, x):
        x = self.interp(x, size=self.size, mode='nearest')
        return x

class Decoder224(nn.Module):
    def __init__(self):
        super(Decoder224, self).__init__()

        self.upsample1= nn.ConvTranspose2d(1, 6, kernel_size=(2, 2), stride=(1, 1), padding=0) #256,1,2,10
        self.upsample2=nn.ConvTranspose2d(6, 8, kernel_size=(2,2), stride=(2,2), padding=0)# (8,10,4,4)
        self.upsample3=nn.ConvTranspose2d(8, 10, kernel_size=(2, 2), stride=(2, 2), padding=0)# (8,10,8,8)
        self.upsample4=nn.ConvTranspose2d(10, 14, kernel_size=(3, 3), stride=(1, 1), padding=0)# (8,10,10,10)
        self.upsample5=nn.ConvTranspose2d(14, 16, kernel_size=(4, 4), stride=(3,3), padding=0) # (8,10,40,40)
        self.upsample6= nn.ConvTranspose2d(16, 18, kernel_size=(3, 3), stride=(2,2), padding=0)#(-,-, 120,120)
  
        self.upsample7=nn.ConvTranspose2d(18, 20, kernel_size=(3, 3), stride=(1, 1), padding=1) #(_,_, 122,122)
        self.upsample8=nn.ConvTranspose2d(20, 22, kernel_size=(3, 3), stride=(2, 2), padding=1)#(_,_, 254,245)
        self.upsample9=nn.ConvTranspose2d(22, 24, kernel_size=(3, 3), stride=(2, 2), padding=2) # (_,_, 239,239)
        self.upsample10=nn.ConvTranspose2d(24, 32, kernel_size=(3, 3), stride=(1, 1), padding=3)# (_,_, 233,233)
        self.upsample11=nn.ConvTranspose2d(32, 3, kernel_size=(3, 3), stride=(2,2), padding=2)  # (_,_, 227,227)
        self.upsample12=nn.ConvTranspose2d(3, 3, kernel_size=(4, 4), stride=(1, 1), padding=3) # (_,_, 224,224)

        self.interp1 = Interpolate(size=(16, 16))
        self.conv1 = nn.Conv2d( in_channels=32,out_channels=10, kernel_size=3)
        self.batchnorm1 = nn.BatchNorm2d(6, 1e-3)
        self.batchnorm2 = nn.BatchNorm2d(8, 1e-3)
        self.batchnorm3 = nn.BatchNorm2d(10, 1e-3)
        self.batchnorm4 = nn.BatchNorm2d(14, 1e-3)
        self.batchnorm5 = nn.BatchNorm2d(16, 1e-3)
        self.batchnorm6 = nn.BatchNorm2d(18, 1e-3)
        self.batchnorm7 = nn.BatchNorm2d(20, 1e-3)
        self.batchnorm8 = nn.BatchNorm2d(22, 1e-3)
        self.batchnorm9 = nn.BatchNorm2d(24, 1e-3)
        self.batchnorm10 = nn.BatchNorm2d(32, 1e-3)
        self.batchnorm11 = nn.BatchNorm2d(3, 1e-3)
        self.batchnorm12 = nn.BatchNorm2d(3, 1e-3)

        self.interp2 = Interpolate(size=(32, 32)) 
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=15, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=15, out_channels=3, kernel_size=3, padding=1)
        self.fc = torch.nn.Linear(32 * 32 * 1, out_features=1000, bias=False)
        self.fc1 = torch.nn.Linear(1000, out_features=255 * 5, bias=False)

    def forward(self, x):
        x = x.view(-1, 1, 1, 1)

        x = torch.tanh(self.batchnorm1(self.upsample1(x)))
        x = torch.tanh(self.batchnorm2(self.upsample2(x)))
        x = torch.tanh(self.batchnorm3(self.upsample3(x)))
        x = torch.tanh(self.batchnorm4(self.upsample4(x)))
        x = torch.tanh(self.batchnorm5(self.upsample5(x)))
        x = torch.tanh(self.batchnorm6(self.upsample6(x)))
        x = torch.tanh(self.batchnorm7(self.upsample7(x)))
        x = torch.tanh(self.batchnorm8(self.upsample8(x)))
        x = torch.tanh(self.batchnorm9(self.upsample9(x)))
        x = torch.tanh(self.batchnorm10(self.upsample10(x)))
        x = torch.tanh(self.batchnorm11(self.upsample11(x)))
        x = torch.tanh(self.batchnorm12(self.upsample12(x)))
        return x 


class CoronaNet(nn.Module):
    
    def __init__(self, supervised):
            super(CoronaNet, self).__init__()

            self.supervised = supervised
            self.optimizer = 'adam'
            self.model_name = 'coronanet'
            self.model_type = 'pytorch'
            self.lr = 1e-4
            
            if self.supervised:
                self.loss_fn = WeightedBCE()
            else:
                self.loss_fn = nn.MSELoss()

            self.dec = Decoder224()
            net = models.vgg16_bn(pretrained=True)

            for param in net.parameters():
                param.requires_grad=True # allow tuning

            self.encoder = net.features[3:-14]

            self.pool = nn.MaxPool2d((2,2))

            self.pre_encoder = nn.Sequential(
                nn.Conv2d(in_channels=3,
                          out_channels=32,
                          kernel_size=3,
                          stride=(2,2),
                          padding=1),

                nn.Conv2d(in_channels=32,
                          out_channels=64,
                          kernel_size=3,
                          stride=(2,2),
                          padding=1),                          
            )

            self.nscnn=nn.Sequential(
                 nn.Conv2d(in_channels=512,
                                   out_channels=128,
                                   kernel_size=1,
                                   padding=0),
                  nn.Conv2d(in_channels=128,
                                   out_channels=64, 
                                   kernel_size=1,
                                   padding=0),
                  nn.Conv2d(in_channels=64,
                                   out_channels=24,
                                   kernel_size=1,
                                   stride=(2,2),
                                   padding=0),   
                  nn.Conv2d(in_channels=24,
                                   out_channels=4,
                                   kernel_size=1,
                                   padding=0)
                )

            for param in self.nscnn.parameters():
                param.requires_grad=True   #False
# 4900
            self.classifier = nn.Sequential( 
                nn.Linear(in_features=64, out_features=32, bias=True),
                nn.ReLU(inplace=False),
                nn.Dropout(0.5),
                nn.Linear(in_features=32, out_features=1, bias=True)
                )

    def build_model(self):
        model = CoronaNet()
        model = {'model':model, 'optimizer':self.optimizer, 'loss_fn':self.loss_fn, 'lr':self.lr,
        'model_name':self.model_name, 'model_type':self.model_type, 'supervised':self.supervised}
        return model

    def forward(self,x):
              x = self.pre_encoder(x)
              x_feat = self.encoder(x) # truncateed vgg19
              x_feat = self.pool(x_feat)
              conv = self.nscnn(x_feat)
              flatt = conv.reshape(-1,conv.size(1)*conv.size(2)*conv.size(3))
              # flatt = conv.view(conv.size(0), -1)

              # bs, 1, 1, 1 (decode from pred?)
              c = torch.sigmoid(self.classifier(flatt))
              dec = self.dec(c)


              return c, dec # class_pred, decoded image