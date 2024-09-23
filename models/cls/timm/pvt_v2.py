from .base import get_timm_pretrained
from ..register import register_model

@register_model('timm')
def pvt_v2(num_classes):
    model, member_fns = get_timm_pretrained('pvt_v2_b0', num_classes)
    model.feat_dim = model.head.in_features
    model.downsample_layers = [
        ['patch_embed/proj', 'Conv2d'],
        ['stages/1/downsample/proj', 'Conv2d'],
        ['stages/2/downsample/proj', 'Conv2d'],
        ['stages/3/downsample/proj', 'Conv2d'],
    ]
    return model, member_fns