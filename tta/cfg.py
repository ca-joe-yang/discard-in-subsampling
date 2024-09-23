from yacs.config import CfgNode as CN

_C = CN()

_C.DATA = CN()
_C.DATA.ROOT = 'datasets'
_C.DATA.NAME = 'cifar100'
_C.DATA.INPUT = CN()
_C.DATA.INPUT.CROP_SIZE = 224
_C.DATA.INPUT.ORIG_SIZE = 256
_C.DATA.INPUT.INTERPOLATE = 'BILINEAR'

_C.MODEL = CN()
_C.MODEL.BACKBONE = 'cnn'
_C.MODEL.VERSION = 'torchvision'
_C.MODEL.FEAT_DIM = 512

_C.OPTIM = CN()
_C.OPTIM.MAX_EPOCH = 30
_C.OPTIM.LR = 0.01
_C.OPTIM.MOMENTUM = 0.9

_C.DATALOADER = CN()
_C.DATALOADER.BATCH_SIZE_TRAIN = 1000
_C.DATALOADER.BATCH_SIZE_EVAL = 128

_C.TTA = CN()
_C.TTA.POLICY = 'standard'
_C.TTA.BUDGET = 30

def get_cfg_tta(opts=[]):
    cfg = _C.clone()
    cfg.merge_from_list(opts)
    cfg.freeze()
    return cfg
