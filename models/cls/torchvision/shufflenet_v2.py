import torch
import torch.nn as nn
import torchvision

from ..register import register_model

def shufflenet_v2_base(model, num_classes):
    model.feat_dim = model.fc.in_features
    model.downsample_layers = [
        ['conv1/0', 'Conv2d'],
        ['maxpool', 'MaxPool2d'],
        ['stage2/0', 'InvertedResidual'],
        ['stage3/0', 'InvertedResidual'],
        ['stage4/0', 'InvertedResidual'],
    ]
    model.num_classes = num_classes
    if model.num_classes != 1000:
        model.fc = nn.Linear(model.feat_dim, model.num_classes)

    member_fns = {}

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        return x
    member_fns[forward_features.__name__] = forward_features

    def forward_head(self, x):
        x = x.mean([2, 3])  # globalpool
        x = self.fc(x)
        return x
    member_fns[forward_head.__name__] = forward_head

    return model, member_fns

@register_model('torchvision')
def shufflenet_v2_x0_5(num_classes):
    model = torchvision.models.shufflenet_v2_x0_5(
        weights=torchvision.models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1)
    return shufflenet_v2_base(model, num_classes)

@register_model('torchvision')
def shufflenet_v2_x1_0(num_classes):
    model = torchvision.models.shufflenet_v2_x1_0(
        weights=torchvision.models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
    return shufflenet_v2_base(model, num_classes)
    
if __name__ == '__main__':
    model = torchvision.models.shufflenet_v2_x0_5()
    print(model)  
