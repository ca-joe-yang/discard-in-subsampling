import os
from yacs.config import CfgNode as CN

_C = CN()

_C.DATA = CN()
_C.DATA.ROOT = 'datasets'
_C.DATA.NAME = 'imagenet'
_C.DATA.INPUT = CN()
_C.DATA.INPUT.PIXEL_MEAN = [0.5, 0.5, 0.5]
_C.DATA.INPUT.PIXEL_STD = [0.5, 0.5, 0.5]
_C.DATA.INPUT.ORIG_SIZE = 256
_C.DATA.INPUT.CROP_SIZE = 224
_C.DATA.INPUT.INTERPOLATE = 'BILINEAR'

_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 8
_C.DATALOADER.BATCH_SIZE = 256

_C.MODEL = CN()
_C.MODEL.BACKBONE = 'Undefined'
_C.MODEL.NAME = 'None'
_C.MODEL.VERSION = 'torchvision'
_C.MODEL.DOWNSAMPLE_LAYERS = []
_C.MODEL.FEAT_DIM = 512

_C.SEARCH = CN()

_C.SEARCH.BUDGET = 1
_C.SEARCH.CRITERION = 'min_delta'
_C.SEARCH.SEARCH_SPACE = [1, 2, 3]
_C.SEARCH.AGGREGATE = CN()
_C.SEARCH.AGGREGATE.MODE = 'features'
_C.SEARCH.AGGREGATE.ALIGN = False
_C.SEARCH.AGGREGATE.OP = 'avg'
_C.SEARCH.ONE_BY_ONE = False

_C.SEARCH.OPTIM = CN()
_C.SEARCH.OPTIM.MAX_EPOCH = 10
_C.SEARCH.OPTIM.LR = 1e-7

_C.SEARCH.ATTENTION = CN()
_C.SEARCH.ATTENTION.HIDDEN_DIM_SCALE = 1.0
_C.SEARCH.ATTENTION.NUM_HEAD = 1

def get_cfg(filename: str, opts: list) -> CN:
    cfg = _C.clone()
    cfg.merge_from_file(filename)
    cfg.merge_from_list(opts)
    match cfg.MODEL.VERSION:
        case 'timm' | 'torchvision':
            cfg.DATA.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
            cfg.DATA.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
        case _:
            raise ValueError(cfg.MODEL.VERSION)
    cfg.freeze()
    return cfg