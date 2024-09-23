from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.BACKBONE = 'deit'
_C.MODEL.DOWNSAMPLE_LAYERS = [
    ['patch_embed/proj', 'conv'],
]

def get_cfg():
    return _C.clone()
