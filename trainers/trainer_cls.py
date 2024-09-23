import os
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.utils import accuracy, AverageMeter

from models import get_pretrained_model
from models.cls import convert
from data import get_datamodule
from evaluators import get_evaluator

class TrainerCls:

    def __init__(self, 
        cfg, 
        pretrained_path: str | None = None
    ):
        self.device = torch.device('cuda')
        datamodule = get_datamodule(cfg)
        model = get_pretrained_model(
            cfg, datamodule.num_classes, pretrained_path)
        self.model = convert(model, cfg, datamodule.num_classes)
        self.model.to(self.device)
        self.model_name = cfg.MODEL.NAME

        self.dm = datamodule

        ckpt_dir = f'checkpoints/poolsearch/{cfg.DATA.NAME}/{cfg.MODEL.VERSION}'
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)
        self.ckpt_fname = os.path.join(
            ckpt_dir, f'{cfg.MODEL.BACKBONE}-agg.pt')
        self.epoch_idx = -1

        self.evaluator = get_evaluator(
            'NoTTA',
            filename=os.path.join('results', f'{cfg.DATA.NAME}-{cfg.MODEL.VERSION}.tsv')
        )
        self.evaluator.load()
    
    def load(self, fname=None):
        if fname is None:
            fname = self.ckpt_fname
        if not os.path.exists(fname):
            print(f'[!] No checkpoint found at {fname}')
            return
        print(f'[*] Load from {fname}')
        self.model.agg_layer.load_state_dict(torch.load(fname))

    def save(self, fname=None):
        if fname is None:
            fname = self.ckpt_fname
        print(f'[*] Save to {fname}')
        torch.save(self.model.agg_layer.state_dict(), fname)

    def train(self):
        self.model.train()
        loader = self.dm.train_loader
        max_epoch = self.model.cfg.OPTIM.MAX_EPOCH
        params = [p for p in self.model.agg_layer.parameters()]
        self.optim = torch.optim.AdamW(
            params, lr=self.model.cfg.OPTIM.LR)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim, max_epoch * len(loader))
        loss_fn = nn.CrossEntropyLoss()
        best_acc = -np.inf

        for self.epoch_idx in range(max_epoch):
            self.model.criterion = 'random-max_learn'
            loss_meter = AverageMeter()
            acc_meter = AverageMeter()
            pbar = tqdm(loader)
            for images, labels in pbar:
                labels = labels.to(self.device)
                images = images.to(self.device)
                # compute output
                output = self.model(images)

                loss = loss_fn(output, labels)
                acc, = accuracy(output.detach(), labels, topk=(1,))
                
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                sched.step()

                loss_meter.update(loss.item(), len(images))
                acc_meter.update(acc.item(), len(images))
                pbar.set_description(f"Train [{self.epoch_idx+1}] [L: {loss_meter.avg:.3f}, A: {acc_meter.avg:.2f}]")
            
            if (self.epoch_idx + 1) % 1 == 0:
                results = self.test('val', criterion='max_learn')
                if results['top1'] > best_acc:
                    best_acc = results['top1']
                    self.save()

    @torch.no_grad()
    def test(self, 
        split='test', 
    ) -> None:
        match split:
            case 'val':
                loader = self.dm.val_loader
            case 'test':
                loader = self.dm.test_loader
            case _:
                raise ValueError(split)
        
        self.model.eval()
        # print(self.model.budget, self.model.criterion, self.model.aggregate_fn)

        latency = AverageMeter()
        acc_meter = AverageMeter()
        pbar = tqdm(loader)
        for (images, target) in pbar:
            end = time.time()
            target = target.to(self.device)
            images = images.to(self.device)
            batch_size = len(images)

            logits = self.model(images)

            # measure accuracy and record loss
            acc1 = accuracy(logits.detach(), target, topk=(1, ))[0]
            acc_meter.update(acc1.item(), batch_size)
            latency.update(1000*(time.time() - end) / batch_size, batch_size)
            pbar.set_description(
                f"Eval [{self.epoch_idx+1}] [A: {acc_meter.avg:.2f}]")

        self.evaluator.update({
            'model': self.model_name,
            'budget': self.model.budget,
            'criterion': self.model.criterion,
            'aggregate': self.model.aggregate_fn,
            'top1_acc': acc_meter.avg,
            'top1_err': 100 - acc_meter.avg
        })
        self.evaluator.log()
