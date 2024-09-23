from .base import get_timm_pretrained
from ..register import register_model

@register_model('timm')
def mobilenet_v3(num_classes):
    model, member_fns = get_timm_pretrained('mobilenetv3_small_100', num_classes)
    model.feat_dim = model.classifier.in_features
    model.downsample_layers = [
        ['conv_stem', 'Conv2d'],
        ['blocks/0/0/conv_dw', 'Conv2d'],
        ['blocks/1/0/conv_dw', 'Conv2d'],
        ['blocks/2/0/conv_dw', 'Conv2d'],
        ['blocks/4/0/conv_dw', 'Conv2d'],
    ]
    return model, member_fns

