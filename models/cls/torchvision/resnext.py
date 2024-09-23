import torch
import torch.nn as nn
import torchvision

from ..register import register_model
from .resnet import resnet50_base

# def resnext50_base(model, num_classes):
#     model.feat_dim = model.fc.in_features
#     model.downsample_layers = [
#         ['conv1', 'Conv2d'],
#         ['maxpool', 'MaxPool2d'],
#         ['layer2/0', 'BottleNeck'],
#         ['layer3/0', 'BottleNeck'],
#         ['layer4/0', 'BottleNeck'],
#     ]
#     model.num_classes = num_classes
#     if model.num_classes != 1000:
#         model.fc = nn.Linear(model.feat_dim, model.num_classes)

#     member_fns = {}

#     def forward_features(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         return x
#     member_fns[forward_features.__name__] = forward_features

#     def forward_head(self, x):
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         return x
#     member_fns[forward_head.__name__] = forward_head

#     return model, member_fns

@register_model('torchvision')
def resnext50_32x4d(num_classes):
    model = torchvision.models.resnext50_32x4d(
        weights=torchvision.models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
    return resnet50_base(model, num_classes)
    
if __name__ == '__main__':
    model = torchvision.models.resnext50_32x4d()
    print(model)  
