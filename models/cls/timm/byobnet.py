from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.BACKBONE = 'repvgg'
_C.MODEL.DOWNSAMPLE_LAYERS = [
    ['stem', 'repvgg_block'],
    ['stages/0/0', 'repvgg_block'],
    ['stages/1/0', 'repvgg_block'],
    ['stages/2/0', 'repvgg_block'],
    ['stages/3/0', 'repvgg_block'],
]

def get_cfg():
    return _C.clone()
