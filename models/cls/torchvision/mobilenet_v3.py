import torch
import torch.nn as nn
import torchvision

from ..register import register_model

def mobilenet_v3_base(model, num_classes):
    model.feat_dim = model.classifier[0].in_features
    model.downsample_layers = [
        ['features/0/0', 'Conv2d'],
        ['features/1/block/0/0', 'Conv2d'],
        ['features/2/block/1/0', 'Conv2d'],
        ['features/4/block/1/0', 'Conv2d'],
        ['features/9/block/1/0', 'Conv2d'],
    ]
    model.num_classes = num_classes
    if model.num_classes != 1000:
        in_dim = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_dim, model.num_classes)

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
def mobilenet_v3_small(num_classes):
    model = torchvision.models.mobilenet_v3_small(
        weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    return mobilenet_v3_base(model, num_classes)

if __name__ == '__main__':
    model = torchvision.models.mobilenet_v3_small()
    print(model)  
