from .base import get_timm_pretrained
from ..register import register_model

@register_model('timm')
def efficientnet(num_classes):
    model, member_fns = get_timm_pretrained('efficientnet_b0', num_classes)
    model.feat_dim = model.classifier.in_features
    model.downsample_layers = [
        ['conv_stem', 'Conv2d'],
        ['blocks/1/0/conv_dw', 'Conv2d'],
        ['blocks/2/0/conv_dw', 'Conv2d'],
        ['blocks/3/0/conv_dw', 'Conv2d'],
        ['blocks/5/0/conv_dw', 'Conv2d'],
    ]
    return model, member_fns
