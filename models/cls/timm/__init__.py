from .coat import coat_tiny
from .resnet import resnet18, resnet50
from .convnext_v2 import convnext_v2
from .efficientformer import efficientformer
from .efficientnet import efficientnet
from .mobilenet_v2 import mobilenet_v2
from .mobilenet_v3 import mobilenet_v3
from .mobilevit_v2 import mobilevit_v2
from .pvt_v2 import pvt_v2

from ..register import get_model_fn

def get_timm_backbone(
    model_name: str, 
    num_classes: int
):
    ModelModule = get_model_fn('timm', model_name)
    model, member_fns = ModelModule(num_classes)
    return model, member_fns