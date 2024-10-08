import os
import argparse
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

import poolsearch
from poolsearch.utils.random_helper import set_random_seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch TIMM')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--config', default='configs/cls/timm/resnet18.py', type=str)
    parser.add_argument('--pretrained-path', type=str, default=None)
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--exp-name', type=str, default=None)
    parser.add_argument('--exp-session', type=str, default='results')
    parser.add_argument('--evaluate-split', type=str, default='test')
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    set_random_seed(args.seed)
    cfg = poolsearch.configs.timm.get_cfg(args.config, args.opts)

    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    
    
    datamodule = poolsearch.data.get_datamodule(cfg)
    
    model = poolsearch.models.get_pretrained_model(
        cfg, datamodule.num_classes, args.pretrained_path).to(device)
    model = poolsearch.models.cls.convert(model, cfg)
    model = model.to(device)
    
    trainer = poolsearch.TrainerCls(cfg, model, datamodule, device)
    trainer.train(args.evaluate_split)
