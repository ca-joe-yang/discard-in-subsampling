from .resnet import *
from .resnext import *
from .mobilenet_v2 import *
from .mobilenet_v3 import *
from .inception_v3 import *
from .swin import *
from .swin_v2 import *
from .efficientnet import *
from .shufflenet_v2 import *

from ..register import *

def get_model(backbone, num_classes):
    model_fn = get_model_fn('torchvision', backbone)
    model, member_fns = model_fn(num_classes)
    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
    member_fns[forward.__name__] = forward
    return model, member_fns