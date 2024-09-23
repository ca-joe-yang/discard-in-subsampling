# TODO: CrossViT has multiscale F_theta
# timm.CrossViT
from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.BACKBONE = 'crossvit'
_C.MODEL.DOWNSAMPLE_LAYERS = [
    ['patch_embed/0/proj', 'conv'], #12
    ['patch_embed/1/proj', 'conv'], #16
]

def get_cfg():
    return _C.clone()