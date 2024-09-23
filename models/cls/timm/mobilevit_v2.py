from .base import get_timm_pretrained
from ..register import register_model

@register_model('timm')
def mobilevit_v2(num_classes):
    model, member_fns = get_timm_pretrained('mobilevitv2_050', num_classes)
    model.feat_dim = model.head.fc.in_features
    model.downsample_layers = [
        ['stem/conv', 'Conv2d'],
        ['stages/1/0/conv2_kxk/conv', 'Conv2d'],
        ['stages/2/0/conv2_kxk/conv', 'Conv2d'],
        ['stages/3/0/conv2_kxk/conv', 'Conv2d'],
        ['stages/4/0/conv2_kxk/conv', 'Conv2d'],
    ]
    return model, member_fns
