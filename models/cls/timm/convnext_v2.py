from .base import get_timm_pretrained
from ..register import register_model

@register_model('timm')
def convnext_v2(num_classes):
    model, member_fns = get_timm_pretrained('convnextv2_tiny', num_classes)
    model.feat_dim = 768
    model.downsample_layers = [
        ['stem/0', 'Conv2d'], #4
        ['stages/1/downsample/1', 'Conv2d'], #2
        ['stages/2/downsample/1', 'Conv2d'], #2
        ['stages/3/downsample/1', 'Conv2d'], #2
    ]
    return model, member_fns
