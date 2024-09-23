import os
import argparse

import torch

from trainers import TrainerCls
from utils.random_helper import set_random_seed
from configs.cls import get_cfg

def main(args) -> None:
    # os.makedirs(args.results_dir, exist_ok=True)
    set_random_seed(args.seed)
    cfg = get_cfg(args.config_file, args.opts)

    torch.backends.cudnn.benchmark = True

    trainer = TrainerCls(cfg)
    trainer.load()
    trainer.test(args.evaluate_split)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch TIMM')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--config-file', default='configs/cls/timm/resnet18.py', type=str)
    parser.add_argument('--exp-name', type=str, default=None)
    parser.add_argument('--evaluate-split', type=str, default='test')
    parser.add_argument(
        '--pretrained-path', type=str, default=None,
        help='Pretrained path')
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
