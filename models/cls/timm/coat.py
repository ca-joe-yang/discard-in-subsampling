from .base import get_timm_pretrained
from ..register import register_model

def coat_base(model):
    model.feat_dim = model.head.in_features
    model.downsample_layers = [
        ['patch_embed1/proj', 'Conv2d'],
        ['patch_embed2/proj', 'Conv2d'],
        ['patch_embed3/proj', 'Conv2d'],
        ['patch_embed4/proj', 'Conv2d'],
    ]
    return model

@register_model('timm')
def coat_tiny(num_classes: int):
    model, member_fns = get_timm_pretrained('coat_tiny', num_classes)
    model = coat_base(model)
    return model, member_fns

if __name__ == '__main__':
    import timm
    model = timm.create_model('coat_tiny')
    print(model)  
