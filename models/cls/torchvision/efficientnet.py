import torch
import torch.nn as nn
import torchvision

from ..register import register_model

def efficientnet_base(model, num_classes):
    model.feat_dim = model.classifier[1].in_features
    model.downsample_layers = [
        ['features/0/0', 'Conv2d'],
        ['features/2/0/block/1/0', 'Conv2d'],
        ['features/3/0/block/1/0', 'Conv2d'],
        ['features/4/0/block/1/0', 'Conv2d'],
        ['features/6/0/block/1/0', 'Conv2d'],
    ]
    model.num_classes = num_classes
    if model.num_classes != 1000:
        model.fc = nn.Linear(model.feat_dim, model.num_classes)

    member_fns = {}

    def forward_features(self, x):
        x = self.features(x)
        return x
    member_fns[forward_features.__name__] = forward_features

    def forward_head(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    member_fns[forward_head.__name__] = forward_head

    return model, member_fns

@register_model('torchvision')
def efficientnet_b0(num_classes):
    model = torchvision.models.efficientnet_b0(
        weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    return efficientnet_base(model, num_classes)
    
if __name__ == '__main__':
    model = torchvision.models.efficientnet_b0()
    print(model)  
