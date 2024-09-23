import torch
import torch.nn as nn
import torchvision

from ..register import register_model

@register_model('torchvision')
def mobilenet_v2(num_classes):

    model = torchvision.models.mobilenet_v2(
        weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.feat_dim = 1280
    model.downsample_layers = [
        ['features/0/0', 'Conv2d'],
        ['features/2/conv/1/0', 'Conv2d'],
        ['features/4/conv/1/0', 'Conv2d'],
        ['features/7/conv/1/0', 'Conv2d'],
        ['features/14/conv/1/0', 'Conv2d'],
    ]
    model.num_classes = num_classes
    if model.num_classes != 1000:
        model.classifier[1] = nn.Linear(model.feat_dim, model.num_classes)

    member_fns = {}

    def forward_features(self, x):
        x = self.features(x)
        return x
    member_fns[forward_features.__name__] = forward_features

    def forward_head(self, x):
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    member_fns[forward_head.__name__] = forward_head

    return model, member_fns
