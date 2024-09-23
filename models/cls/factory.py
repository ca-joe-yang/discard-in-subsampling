import torch
from .register import get_model_fn
from .timm import get_timm_backbone

import types

def get_pretrained_model(
    cfg, 
    num_classes: int | None = None, 
    ckpt_path: str | None = None
):
    match cfg.MODEL.VERSION:
        case 'torchvision':
            from .torchvision import get_model
            model, member_fns = get_model(cfg.MODEL.BACKBONE, num_classes)
        case 'timm':
            model, member_fns = get_timm_backbone(cfg.MODEL.BACKBONE, num_classes)
        case 'clip':
            from .clip import get_model
            model, member_fns = get_model(cfg.MODEL.BACKBONE, cfg.DATA.NAME)
        case 'custom':
            model_fn = get_model_fn('custom', cfg.MODEL.BACKBONE)
            model, member_fns = model_fn(num_classes)
        case _:
            raise ValueError(cfg.MODEL.VERSION)

    for attr in member_fns.keys():
        if hasattr(model, attr):
            print(f'[!] Replacing {cfg.MODEL.BACKBONE}\'s {attr} with customized one.')
        setattr(model, attr, 
            types.MethodType(member_fns[attr], model))

    if ckpt_path is not None:
        model.load_state_dict(torch.load(ckpt_path))
    return model
