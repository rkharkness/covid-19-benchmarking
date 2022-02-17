"""code adapted from: https://github.com/chirag126/CoroNet"""

# import required libraries
import torch
import torch.nn as nn
import torch.utils.data
from torchvision import models
from torchsummary import summary

class WeightedBCE():
    def  __init__(self) -> None:
        super().__init__()
        self.weights = {'neg':1.32571275, 'pos':0.80276873}
    
    def forward(self, target, output):
        output = torch.clamp(output,min=1e-10,max=1-1e-10)  
        loss =  self.weights['pos'] * (target * torch.log(output)) + self.weights['neg'] * ((1 - target) * torch.log(1 - output))
        return torch.neg(torch.mean(loss))

class FPAE(nn.Module):
    def __init__(self):
        super(FPAE, self).__init__()
        self.bn0 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=24, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=24)
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=16, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(num_features=16)
        self.conv6 = nn.Conv2d(in_channels=16, out_channels=12, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(num_features=12)
        self.conv7 = nn.Conv2d(in_channels=12, out_channels=8, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(num_features=8)
        self.down = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=2)
        self.conv_smooth1 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.conv_smooth2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1)
        self.conv_smooth3 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.convtrans1 = nn.ConvTranspose2d(in_channels=8, out_channels=12, kernel_size=3, padding=1)
        self.convtrans2 = nn.ConvTranspose2d(in_channels=12, out_channels=16, kernel_size=3, padding=1)
        self.convtrans3 = nn.ConvTranspose2d(in_channels=16, out_channels=24, kernel_size=3, padding=1)
        self.convtrans4 = nn.ConvTranspose2d(in_channels=24, out_channels=32, kernel_size=3, padding=1)
        self.convtrans5 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.convtrans6 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, padding=1)
        self.convtrans7 = nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=3, padding=1)

    def forward(self, x):
        x_small = x.clone()
        x_clone = x_small.clone()

        # ============ Encoder ===========
        # ====== Bottom Up Layers =====
        x = self.bn0(x_small)
        res1_x = self.conv1(x)
        x = self.relu(res1_x)
        x = self.bn1(x)
        res2_x = self.conv2(x)
        x = self.relu(res2_x)
        x = self.bn2(x)
        res3_x = self.conv3(x)
        x = self.relu(res3_x)
        x = self.bn3(x)
        _, _, H1, W1 = x.size()

        ### ======= Branch network ======
        x_d1 = self.down(x)  # 128x128
        _, _, H2, W2 = x_d1.size()
        x_d2 = self.down(x_d1)  # 64x64
        _, _, H3, W3 = x_d2.size()
        x_d3 = self.down(x_d2)  # 32x32

        ### ======= First Branch =======
        res4_x = self.conv4(x)
        x = self.relu(res4_x)
        x = self.bn4(x)
        res5_x = self.conv5(x)
        x = self.relu(res5_x)
        x = self.bn5(x)
        res6_x = self.conv6(x)
        x = self.relu(res6_x)
        x = self.bn6(x)
        res7_x = self.conv7(x)
        x = self.relu(res7_x)
        x = self.bn7(x)

        ### ======= Second Branch ========
        x_d1 = self.conv4(x_d1)
        x_d1 = self.relu(x_d1)
        x_d1 = self.bn4(x_d1)
        x_d1 = self.conv5(x_d1)
        x_d1 = self.relu(x_d1)
        x_d1 = self.bn5(x_d1)
        x_d1 = self.conv6(x_d1)
        x_d1 = self.relu(x_d1)
        x_d1 = self.bn6(x_d1)
        x_d1 = self.conv7(x_d1)
        x_d1 = self.relu(x_d1)
        z1 = self.bn7(x_d1)
        x_d1 = self.upsample(z1, size=(H1, W1))

        ### ======= Third Branch ========
        x_d2 = self.conv4(x_d2)
        x_d2 = self.relu(x_d2)
        x_d2 = self.bn4(x_d2)
        x_d2 = self.conv5(x_d2)
        x_d2 = self.relu(x_d2)
        x_d2 = self.bn5(x_d2)
        x_d2 = self.conv6(x_d2)
        x_d2 = self.relu(x_d2)
        x_d2 = self.bn6(x_d2)
        x_d2 = self.conv7(x_d2)
        x_d2 = self.relu(x_d2)
        z2 = self.bn7(x_d2)
        x_d2 = self.upsample(z2, size=(H2, W2))
        x_d2 = self.upsample(x_d2, size=(H1, W1))

        ### ======= Fourth Branch ========
        x_d3 = self.conv4(x_d3)
        x_d3 = self.relu(x_d3)
        x_d3 = self.bn4(x_d3)
        x_d3 = self.conv5(x_d3)
        x_d3 = self.relu(x_d3)
        x_d3 = self.bn5(x_d3)
        x_d3 = self.conv6(x_d3)
        x_d3 = self.relu(x_d3)
        x_d3 = self.bn6(x_d3)
        x_d3 = self.conv7(x_d3)
        x_d3 = self.relu(x_d3)
        z3 = self.bn7(x_d3)
        x_d3 = self.upsample(z3, size=(H3, W3))
        x_d3 = self.upsample(x_d3, size=(H2, W2))
        x_d3 = self.upsample(x_d3, size=(H1, W1))

        ### ======= Concat maps ==========
        x = torch.cat((x, x_d1, x_d2, x_d3), 1)

        x = self.conv_smooth1(x)
        x = self.conv_smooth2(x)
        x = self.conv_smooth3(x)
       
        ### ============ Decoder ==========
        x = self.convtrans1(x)
        x = self.relu(x+res6_x)
        x = self.convtrans2(x)
        x = self.relu(x+res5_x)
        x = self.convtrans3(x)
        x = self.relu(x+res4_x)
        x = self.convtrans4(x)
        x = self.relu(x+res3_x)
        x = self.convtrans5(x)
        x = self.relu(x+res2_x)
        x = self.convtrans6(x)
        x = self.relu(x+res1_x)
        x = self.convtrans7(x)
        x = x + x_clone
        x = self.sigmoid(x)

        return [x, x_small, z3] # recon img, orig, latent

    def upsample(self, x, size):
        up = nn.Upsample(size=size, mode="bilinear")
        return up(x)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.resnet = models.resnet18(pretrained=True)
        self.features_conv = nn.Sequential(*list(self.resnet.children())[:-2])

        self.avgpool = self.resnet.avgpool

        num_ftrs = self.resnet.fc.in_features      
        self.resnet.fc = nn.Linear(num_ftrs, 1)
        self.classifier = self.resnet.fc

    def forward(self, x):
        x = self.features_conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        x = torch.sigmoid(x)
        return x

class CoroNet(nn.Module):
    supervised = False
    def __init__(self, supervised=supervised):
        super(CoroNet, self).__init__()

        self.supervised = supervised

        if self.supervised:
            self.loss_fn = WeightedBCE()
            self.lr = 1e-4
        else:
            self.loss_fn = nn.MSELoss()
            self.lr = 1e-5

        self.fpae = FPAE()
        self.classifier = Classifier()

        self.model_name = 'coronet'
        self.model_type = 'pytorch'
        self.optimizer = 'adam'

    def forward(self, x):
        if self.supervised:
            output = self.classifier(x)
        else:
            output = self.fpae(x) # return list of outputs

        return output 
        
    def build_model(self):
        model = CoroNet()
        model = {'model':model, 'optimizer':self.optimizer, 'loss_fn':self.loss_fn, 'lr':self.lr,
        'model_name':self.model_name, 'model_type':self.model_type, 'supervised':self.supervised}
        return model


if __name__ == "__main__":
    coronet = CoroNet.build_model()
    print(summary(coronet['model']))

    





