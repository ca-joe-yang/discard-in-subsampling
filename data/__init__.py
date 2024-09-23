from .base import BaseDataModule
from .imagenet import ImageNet
from .toy import Toy

def get_datamodule(cfg, tta: bool=False) -> BaseDataModule:
    dataset_name = cfg.DATA.NAME
    match dataset_name:
        case 'imagenet':
            return ImageNet(cfg, tta)
        case 'flowers102':
            return Flowers102(cfg, tta)
        case _:
            raise ValueError(dataset_name)

