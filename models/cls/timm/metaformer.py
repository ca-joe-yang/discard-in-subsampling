import types
from timm.models.metaformer import MetaFormer

def aggregate_metaformer(model, variant=None):
    if isinstance(model, MetaFormer):
        def build_downsample(self, cfg=None):
            print(self)
            self.downsample_layers = [
                {'name': 'conv_stem', 'type': 'conv', 'kwargs': {}},
                {'name': 'blocks/1/0/conv_dw', 'type': 'conv', 'kwargs': {}},
                {'name': 'blocks/2/0/conv_dw', 'type': 'conv', 'kwargs': {}},
                {'name': 'blocks/3/0/conv_dw', 'type': 'conv', 'kwargs': {}},
                {'name': 'blocks/5/0/conv_dw', 'type': 'conv', 'kwargs': {}},
            ]
        model.build_downsample = types.MethodType(build_downsample, model)
        return True, model
    return False, model

