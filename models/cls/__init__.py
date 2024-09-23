import types
from .factory import *

from .base import PoolSearchModelCls

# ATTRIBUTES = [
#     'state2dxy', 'get_layer', 'replace_downsample_module',
#     'get_neighboring_states', 'aggregate_logits', 'aggregate_feats',
#     'build', 'forward_state', 'forward', 
#     'build_cls', 'check_cfg', 'set_state', 'search_criterion'
# ]
def convert(model, cfg, num_classes: int):
    # model.forward_original = model.forward
    # setattr(model, 'forward_original', 
    #     types.MethodType(getattr(model, 'forward'), model))
    # for attr in ATTRIBUTES:
    #     setattr(model, attr, 
    #         types.MethodType(getattr(PoolSearchModelCls, attr), model))

    # for attr in ['forward_features', 'forward_head']:
    #     if hasattr(module, attr):
    #         if hasattr(model, attr):
    #             print(f'[!] Replacing {cfg.MODEL.BACKBONE}\'s {attr} with customized one.')
    #         setattr(model, attr, 
    #             types.MethodType(getattr(module, attr), model))

    model = PoolSearchModelCls(model)
    model.num_classes = num_classes
    # model.downsample_modules = cfg.MODEL.DOWNSAMPLE_LAYERS
    model.cfg_model = cfg.MODEL
    model.build(cfg)
    return model
