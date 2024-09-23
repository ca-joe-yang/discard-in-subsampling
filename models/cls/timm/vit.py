import types
from timm.models.vision_transformer import VisionTransformer

def aggregate_vit(model, variant=None):
    if isinstance(model, VisionTransformer):
        def build_downsample(self, cfg=None):
            self.downsample_layers = [
                {'name': 'patch_embed/proj', 'type': 'conv', 'kwargs': {}},
            ]
        model.build_downsample = types.MethodType(build_downsample, model)
        return True, model
    return False, model
