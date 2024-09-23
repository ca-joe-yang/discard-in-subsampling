import types
from timm.models.vgg import VGG
import torch.nn as nn

def aggregate_vgg(model, variant=None):
    if isinstance(model, VGG):
        def build_downsample(self, cfg=None):
            self.downsample_layers = []
            for i, layer in enumerate(self.features):
                if isinstance(layer, nn.MaxPool2d):
                    self.downsample_layers.append({
                        'name': f'features/{i}', 
                        'type': 'maxpool', 
                        'kwargs': {}
                    })
        model.build_downsample = types.MethodType(build_downsample, model)
        return True, model
    return False, model
