from .base import *

from ..register import *

def get_model(backbone, dataset):
    model_fn = get_model_fn('clip', backbone)
    model, member_fns = model_fn(dataset)
    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
    member_fns[forward.__name__] = forward
    return model, member_fns