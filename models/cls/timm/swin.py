from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.BACKBONE = 'swin'
_C.MODEL.DOWNSAMPLE_LAYERS = [
    ['patch_embed/proj', 'conv'],
    ['layers/0/downsample', 'patch_merge'],
    ['layers/1/downsample', 'patch_merge'],
    ['layers/2/downsample', 'patch_merge'],
]

def get_cfg():
    return _C.clone()
