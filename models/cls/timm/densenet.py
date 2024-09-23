import types
from timm.models.densenet import DenseNet

def aggregate_densenet(model, variant=None):
    if isinstance(model, DenseNet):
        def forward_head(self, x):
            x = self.global_pool(x)
            try:
                x = self.head_drop(x)
            except:
                pass
            x = self.classifier(x)
            return x

        def build_downsample(self, cfg=None):
            self.downsample_layers = [
                {'name': 'features/conv0', 'type': 'conv', 'kwargs': {}},
                {'name': 'features/pool0', 'type': 'maxpool', 'kwargs': {}},
            ]
        model.forward_head = types.MethodType(forward_head, model)
        model.build_downsample = types.MethodType(build_downsample, model)
        return True, model
    return False, model
