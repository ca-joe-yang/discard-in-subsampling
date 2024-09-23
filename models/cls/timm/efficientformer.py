from .base import get_timm_pretrained
from ..register import register_model

@register_model('timm')
def efficientformer(num_classes):
    model, member_fns = get_timm_pretrained('efficientformer_l1', num_classes)
    model.feat_dim = model.head.in_features
    model.downsample_layers = [
        ['stem/conv1', 'Conv2d'],
        ['stem/conv2', 'Conv2d'],
        ['stages/1/downsample/conv', 'Conv2d'],
        ['stages/2/downsample/conv', 'Conv2d'],
        ['stages/3/downsample/conv', 'Conv2d'],
    ]
    return model, member_fns