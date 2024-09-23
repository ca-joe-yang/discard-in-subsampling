import torch
import torch.nn as nn
import torchvision

from ..register import register_model

def swin_base(model, num_classes):
    model.feat_dim = model.head.in_features
    model.downsample_layers = [
        ['features/0/0', 'Conv2d'],
        ['features/2', 'PatchMerging'],
        ['features/4', 'PatchMerging'],
        ['features/6', 'PatchMerging'],
    ]
    model.num_classes = num_classes
    if model.num_classes != 1000:
        model.fc = nn.Linear(model.feat_dim, model.num_classes)

    member_fns = {}

    def forward_features(self, x):
        x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)
        return x
    member_fns[forward_features.__name__] = forward_features

    def forward_head(self, x):
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.head(x)
        return x
    member_fns[forward_head.__name__] = forward_head

    return model, member_fns

@register_model('torchvision')
def swin_t(num_classes):
    model = torchvision.models.swin_t(
        weights=torchvision.models.Swin_T_Weights.IMAGENET1K_V1)
    return swin_base(model, num_classes)
    
if __name__ == '__main__':
    model = torchvision.models.swin_t()
    print(model)  
