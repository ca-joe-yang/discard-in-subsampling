from .base import get_timm_pretrained
from ..register import register_model

@register_model('timm')
def resnet18(num_classes):
    model, member_fns = get_timm_pretrained('resnet18', num_classes)
    model.feat_dim = model.fc.in_features
    model.downsample_layers = [
        ['conv1', 'Conv2d'],
        ['maxpool', 'MaxPool2d'],
        ['layer2/0', 'BasicBlock'],
        ['layer3/0', 'BasicBlock'],
        ['layer4/0', 'BasicBlock'],
    ]
    return model, member_fns

@register_model('timm')
def resnet50(num_classes):
    model, member_fns = get_timm_pretrained('resnet50', num_classes)
    model.feat_dim = model.fc.in_features
    model.downsample_layers = [
        ['conv1', 'Conv2d'],
        ['maxpool', 'MaxPool2d'],
        ['layer2/0', 'BottleNeck'],
        ['layer3/0', 'BottleNeck'],
        ['layer4/0', 'BottleNeck'],
    ]
    return model, member_fns
