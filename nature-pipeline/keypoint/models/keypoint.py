import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation import fcn_resnet50


class BackboneAdapter(nn.Module):

    def __init__(self, backbone, classifier, num_classes):
        super(BackboneAdapter, self).__init__()
        
        last_layer_old = classifier[-1]
        out_layer = nn.Conv2d(last_layer_old.in_channels, num_classes, 1)
        nn.init.kaiming_normal_(out_layer.weight)
        classifier[-1] = out_layer
        
        self.num_classes = num_classes
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = features["out"]
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x


models = {
    'fcn': fcn_resnet50,
    'deeplabv3': deeplabv3_resnet50,
}


class CoordLocationNetwork(nn.Module):
    def __init__(self, n_locations, input_size, backbone):
        super(CoordLocationNetwork, self).__init__()
        self.input_size = input_size
        self.heatmap_size = input_size//4

        model_func = models[backbone]
        try:
            model_pre = model_func(pretrained=True)
        except Exception as e:
            model_pre = model_func(pretrained=False)
            print(e)
        
        model = BackboneAdapter(model_pre.backbone,model_pre.classifier,n_locations)
        self.encode_decode = model
        self.out_channels = n_locations
        self.heatmap_size = input_size

        self.head = nn.Conv2d(self.out_channels, n_locations, kernel_size=1, bias=False)
        size = self.heatmap_size
        first = -(size - 1) / size
        last = -first
        mat_x = torch.linspace(first, last, size, dtype=torch.float)
        mat_x = mat_x.reshape(1,1,1,-1)
        self.mat_x = nn.Parameter(mat_x, requires_grad=False)
        mat_y = torch.linspace(first, last, size, dtype=torch.float)
        mat_y = mat_y.reshape(1,1,-1,1)
        self.mat_y = nn.Parameter(mat_y, requires_grad=False)
        self.softmax = nn.Softmax(dim=2)


    def forward(self, images):
        featmap = self.encode_decode(images)
        unnormalized_heatmaps = self.head(featmap)
        heatmaps = self.softmax(unnormalized_heatmaps.view(*unnormalized_heatmaps.size()[:2], -1)).view_as(unnormalized_heatmaps)
        coords_x = (heatmaps*self.mat_x).sum(dim=2).sum(dim=2,keepdim=True)
        coords_y = (heatmaps*self.mat_y).sum(dim=2).sum(dim=2,keepdim=True)
        coords = torch.cat([coords_x,coords_y],dim=2)
        return coords, heatmaps
