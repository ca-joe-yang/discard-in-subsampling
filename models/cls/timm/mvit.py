import types
from timm.models import MultiScaleVit

def aggregate_mvit(model, variant=None):
    if isinstance(model, MultiScaleVit):
        def build_downsample(self, cfg=None):
            self.downsample_layers = [
                {'name': 'patch_embed/proj', 'type': 'conv', 'kwargs': {}}, # 4
                {'name': 'stages/1/blocks/0', 'type': 'multiscaleblock', 'kwargs': {}}, #2
                {'name': 'stages/2/blocks/0', 'type': 'multiscaleblock', 'kwargs': {}}, #2
                {'name': 'stages/3/blocks/0', 'type': 'multiscaleblock', 'kwargs': {}}, #2
            ]
        model.build_downsample = types.MethodType(build_downsample, model)
        return True, model
    return False, model
