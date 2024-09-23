import os
import time
import numpy as np
from tqdm import tqdm
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tta.policy import get_ttamodule
from tta.methods.aug_tta import AugTTA
from tta.methods.class_tta import ClassTTA
from tta.methods.agg_tta import AggTTA
from tta.methods.gps import GPS
from tta.methods.naive import MeanTTA, MaxTTA
from tta.wrapper import WrapperTTA
from timm.utils import accuracy, AverageMeter
from evaluators import get_evaluator
from data import get_datamodule
from models import get_pretrained_model
from models.cls import convert

from configs.cls import get_cfg

class TrainerTTA:

    def __init__(self, cfg, args):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dm = get_datamodule(cfg, tta=True)

        self.pretrained_model = get_pretrained_model(
            cfg, self.dm.num_classes, args.pretrained_path).to(self.device)
        self.pretrained_model.eval()
        print("[*] Pretrained model loaded!")

        self.tm = get_ttamodule(cfg, self.dm)
        self.num_classes = self.dm.num_classes

        self.tta_policy = cfg.TTA.POLICY
        self.model_version = cfg.MODEL.VERSION
        self.model_name = cfg.MODEL.BACKBONE
        self.lr = cfg.OPTIM.LR
        self.momentum = cfg.OPTIM.MOMENTUM
        self.num_epochs = cfg.OPTIM.MAX_EPOCH

        self.exp_dir = os.path.join(
            args.exp_root, cfg.DATA.NAME, cfg.MODEL.BACKBONE
        )

        self.batch_size_train = cfg.DATALOADER.BATCH_SIZE_TRAIN
        self.batch_size_eval = cfg.DATALOADER.BATCH_SIZE_EVAL
        
        # Directory
        self.agg_models_dir = os.path.join(
            self.exp_dir, self.tta_policy, 'agg_models')
        self.result_fname = os.path.join(
            self.exp_dir, self.tta_policy, f'results.csv')
        
        self.train_data_dir = os.path.join(
            self.exp_dir, self.tta_policy, 'train')
        self.train_labels_fname = os.path.join(
            self.train_data_dir, 'labels.pt')
        self.val_data_dir = os.path.join(
            self.exp_dir, self.tta_policy, 'val')
        self.val_labels_fname = os.path.join(
            self.val_data_dir, 'labels.pt')
        self.test_data_dir = os.path.join(
            self.exp_dir, self.tta_policy, 'test')
        self.test_labels_fname = os.path.join(
            self.test_data_dir, 'labels.pt')

        for d in [
            self.exp_dir,
            self.agg_models_dir,
            self.train_data_dir,
            self.val_data_dir,
            self.test_data_dir,
        ]:
            if not os.path.exists(d):
                os.makedirs(d, exist_ok=True)

        self.evaluator = get_evaluator('TTA', filename=self.result_fname)
        self.evaluator.load()
        
    def gen_aug_logits(self):
        self._gen_aug_logits('train')
        self._gen_aug_logits('val')
        self._gen_aug_logits('test')

    def check_aug_logits(self, data_dir, labels_fname):
        names = []
        for i, name in enumerate(self.tm.transforms_names):
            if not os.path.exists(
                os.path.join(data_dir, f'{name}.pt')):
                names.append(name)
        return names
        
    @torch.no_grad()
    def _gen_aug_logits(self, split):
        if split == 'train':
            loader = self.dm.train_loader
            data_dir = self.train_data_dir
            labels_fname = self.train_labels_fname
        elif split == 'val':
            loader = self.dm.val_loader
            data_dir = self.val_data_dir
            labels_fname = self.val_labels_fname
        elif split == 'test':
            loader = self.dm.test_loader
            data_dir = self.test_data_dir
            labels_fname = self.test_labels_fname
        else:
            raise ValueError(split)
        
        all_transform_names = self.check_aug_logits(
            data_dir, labels_fname)

        if len(all_transform_names) == 0:
            return

        budget_batch = 100
        for i in range(0, len(all_transform_names), budget_batch):
            start_idx = i
            end_idx = np.minimum(i + budget_batch, len(all_transform_names))
            batch = end_idx - start_idx
            transform_names = all_transform_names[start_idx:end_idx]

            aug_logits = []
            for _ in range(batch):
                aug_logits.append([])

            tta_transforms = self.tm.name2transtorms(transform_names)
            tta_wrapper = WrapperTTA(
                self.pretrained_model, 
                tta_transforms
            )
            tta_wrapper.eval()
            tta_wrapper.to(self.device)
            if i == 0:
                all_labels = []
            for images, labels in tqdm(
                loader,
                desc=f'Generate aug logits of {split} [{start_idx}:{end_idx}]'
            ):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = tta_wrapper(images)#.permute(1, 0, 2) # N A K -> A N K
                for a in range(batch):
                    aug_logits[a].append(outputs[:, a])
                if i == 0:
                    all_labels.append(labels)
        
            for a in tqdm(range(batch), desc='Save aug logits'):
                t_name = transform_names[a]
                logits = torch.cat(aug_logits[a], dim=0)
                torch.save(
                    logits, 
                    os.path.join(data_dir, f'{t_name}.pt'))
            if i == 0:
                torch.save(torch.cat(all_labels, 0), labels_fname)

    @torch.no_grad()
    def test(self, 
        budget: int = 1, 
        agg_name: str = 'Mean', 
        split: str = 'test', 
        agg_model=None, 
        best=False,
    ):
        self.build_loader(budget, split=split, best=best)
        print(f"AUG: {budget} AGG: {agg_name}")
        if agg_model is None:
            agg_model = self.get_agg_model(agg_name, budget)
        agg_model.eval()
        
        if split == 'test':
            pbar = tqdm(self.test_aug_loader)
        elif split == 'val':
            pbar = tqdm(self.val_aug_loader)
        else:
            raise ValueError(split)
        
        latency = AverageMeter()
        top1 = AverageMeter()
        for inputs, labels in pbar:
            end = time.time()
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = agg_model(inputs)

            scores = accuracy(outputs, labels, topk=(1,))
            top1.update(scores[0].item(), len(inputs))
            # top5.update(scores[1].item(), len(inputs))
            latency.update(1000*(time.time() - end) / len(inputs), len(inputs))
            
            pbar.set_description(f'Eval [A : {top1.avg:.3f}]')
        
        if split == 'test':
            self.evaluator.update({
                'model': self.model_name,
                'budget': budget,
                'agg': agg_name,
                'top1': top1.avg, 
                'policy': self.tta_policy
            })
            self.evaluator.log()

        return top1.avg

    @torch.no_grad()
    def test_model(self, model, budget=1, agg_name='Mean', tta_idxs=None):
        model.eval()
        print(f"AUG: {budget} \tAGG: {agg_name}")
        
        pbar = tqdm(self.dm.test_loader)

        latency = AverageMeter()
        top1 = AverageMeter()
        for inputs, labels in pbar:
            end = time.time()
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = model(inputs, tta_idxs)

            scores = accuracy(outputs, labels, topk=(1,))

            top1.update(scores[0].item(), len(inputs))
            latency.update(1000*(time.time() - end) / len(inputs), len(inputs))
            
            pbar.set_description(f'Eval [A : {top1.avg:.3f}]')
        
        result = {
            'top1': top1.avg, 
            'latency': latency.avg
        }
        return result

    def train(self, budget=2, best=False):
        self.build_loader(budget, split='train', best=best)

        self.model_plr = AugTTA(
            budget, 1.0)
        self.model_plr = self.train_agg_model(budget, mode='AugTTA', best=best)

        self.model_flr = ClassTTA(
            budget, self.num_classes, 1.0)
        self.model_flr = self.train_agg_model(budget, mode='ClassTTA', best=best)
        
        n_subpolicies = 3
        self.model_gps = GPS(
            np.minimum(budget, n_subpolicies))
        self.train_agg_model(budget, mode='GPS', best=best)
    
    def train_agg_model(self, budget, mode='AugTTA', best=True):
        if mode == 'ClassTTA':
            agg_model_path = os.path.join(
                self.agg_models_dir, f'{self.tm.name}-{budget:02d}-ClassTTA.pth')
            agg_model = self.model_flr
        elif mode == 'AugTTA':
            agg_model_path = os.path.join(
                self.agg_models_dir, f'{self.tm.name}-{budget:02d}-AugTTA.pth')
            agg_model = self.model_plr
        elif mode == 'GPS':
            agg_model_path = os.path.join(
                self.agg_models_dir, f'{self.tm.name}-{budget:02d}-GPS.pth')
            agg_model = self.model_gps
        
        if os.path.exists(agg_model_path):
            agg_model.load_state_dict(torch.load(agg_model_path))
            return agg_model

        if mode == 'GPS':
            agg_model.find_temperature(
                self.train_aug_logits, self.train_labels, self.device)
            agg_model.find_idxs(
                self.train_aug_logits, self.train_labels, self.device)
            torch.save(agg_model.state_dict(), agg_model_path)
        else:
            losses = AverageMeter()
            top1 = AverageMeter()

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(
                agg_model.parameters(), 
                lr=self.lr, momentum=self.momentum, weight_decay=1e-4)
            agg_model.to(self.device)
            criterion.to(self.device)
            agg_model.train()
            lambda1 = .01 
            params = torch.cat([x.view(-1) for x in agg_model.parameters()])

            # self.test(budget, agg_name='Mean')

            best_acc = 0
            for epoch in tqdm(
                range(self.num_epochs),
                desc=f'Training {mode}'
            ):
                for aug_logits, labels in self.train_aug_loader:
                    aug_logits, labels = aug_logits.to(self.device), labels.to(self.device)
                    outputs = agg_model(aug_logits)
                    nll_loss = criterion(outputs, labels)
                    l1_loss = lambda1 * torch.norm(params, 1)
                    loss = nll_loss 
                    acc1, = accuracy(outputs, labels, topk=(1,))

                    bs = len(aug_logits)
                    losses.update(loss.item(), bs)
                    top1.update(acc1.item(), bs)
                            
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()   
                    for p in agg_model.parameters():
                        p.data.clamp_(0)
                
                if (epoch == 0) or (epoch+1) % 5 == 0 or epoch == self.num_epochs - 1:
                    acc = self.test(
                        budget, agg_name=mode, split='val', agg_model=agg_model, best=best)
                    if acc > best_acc:
                        best_acc = acc
                        torch.save(agg_model.state_dict(), agg_model_path)
        return agg_model

    def get_agg_model(self, agg_name, budget=None):     
        match agg_name:
            case 'Mean':
                return MeanTTA()
            case 'Max':
                return MaxTTA()
            case 'ClassTTA':
                agg_model = ClassTTA(
                    budget, self.num_classes, 1.0)
                agg_model_path = os.path.join(
                    self.agg_models_dir, f'{self.tm.name}-{budget:02d}-ClassTTA.pth')
                agg_model.load_state_dict(torch.load(agg_model_path))
                return agg_model.to(self.device)
            case 'AugTTA':
                agg_model = AugTTA(
                    budget, 1.0)
                agg_model_path = os.path.join(
                    self.agg_models_dir, f'{self.tm.name}-{budget:02d}-AugTTA.pth')
                agg_model.load_state_dict(torch.load(agg_model_path))
                return agg_model.to(self.device)
            case 'GPS':
                agg_model = GPS(
                    np.minimum(budget, 3))
                agg_model_path = os.path.join(
                    self.agg_models_dir, f'{self.tm.name}-{budget:02d}-GPS.pth')
                agg_model.load_state_dict(torch.load(agg_model_path))
                return agg_model.to(self.device)
            case _:
                raise ValueError(agg_name)
            
    def build_loader(self, budget=1, split='train', best=True):
        if best:
            tta_transforms, tta_idxs = self.get_best_transforms(
                128, budget)
            aug_fnames = [t.name for t in tta_transforms]
        else:
            aug_fnames = self.tm.budget2augs(budget)
        num_workers = 8

        # Train
        if split == 'train':
            train_aug_logits = [ torch.load(os.path.join(
                self.train_data_dir, f"{fname}.pt"
            )).cpu() for fname in aug_fnames ]
            self.train_aug_logits = torch.stack(train_aug_logits, 1)
            self.train_labels = torch.load(self.train_labels_fname).cpu()

            train_aug_dataset = torch.utils.data.TensorDataset(
                self.train_aug_logits, self.train_labels
            )
            self.train_aug_loader = DataLoader(
                train_aug_dataset, batch_size=self.batch_size_train, shuffle=True,
                num_workers=num_workers)

        # Val
        elif split == 'val':
            val_aug_logits = [ torch.load(os.path.join(
                self.val_data_dir, f"{fname}.pt"
            )).cpu() for fname in aug_fnames ]
            self.val_aug_logits = torch.stack(val_aug_logits, 1)
            self.val_labels = torch.load(self.val_labels_fname).cpu()

            val_aug_dataset = torch.utils.data.TensorDataset(
                self.val_aug_logits, self.val_labels
            )
            self.val_aug_loader = DataLoader(
                val_aug_dataset, batch_size=self.batch_size_eval, shuffle=False,
                num_workers=num_workers)

        # Test
        elif split == 'test':
            test_aug_logits = [ torch.load(os.path.join(
                self.test_data_dir, f"{fname}.pt"
            )).cpu() for fname in aug_fnames ]
            self.test_aug_logits = torch.stack(test_aug_logits, 1)
            self.test_labels = torch.load(self.test_labels_fname).cpu()

            test_aug_dataset = torch.utils.data.TensorDataset(
                self.test_aug_logits, self.test_labels
            )
            self.test_aug_loader = DataLoader(
                test_aug_dataset, batch_size=self.batch_size_eval, shuffle=False,
                num_workers=num_workers)

    def get_best_transforms(self, train_budget, test_budget):
        agg_model = GPS(test_budget)
        agg_model_path = os.path.join(
            self.agg_models_dir, f'{self.tm.name}-{train_budget:02d}-GPS-{test_budget:02d}.pth')

        if os.path.exists(agg_model_path):
            agg_model.load_state_dict(torch.load(agg_model_path))
        else:
            self.build_loader(train_budget, 'train', False)
            agg_model.find_temperature(
                self.train_aug_logits, self.train_labels, self.device)
            agg_model.find_idxs(
                self.train_aug_logits, self.train_labels, self.device)
            torch.save(agg_model.state_dict(), agg_model_path)

        tta_transforms, tta_idxs = self.tm.get_best_transforms(agg_model, test_budget)
        print('best:', [t.name for t in tta_transforms])

        return tta_transforms, tta_idxs

    @torch.no_grad()
    def test_poolsearch(self, 
        agg_name, config_fname, exp_name, 
        search_budget, tta_budget, 
        learn=False, best=False
    ):
        if learn:
            opts = [
                'SEARCH.BUDGET', str(search_budget),
                'SEARCH.AGGREGATE.ALIGN', True,
                'SEARCH.AGGREGATE.OP', 'learn',
                'SEARCH.CRITERION', 'max_learn'
            ]
            best = True
        else:
            opts = [
                'SEARCH.BUDGET', str(search_budget),
                'SEARCH.AGGREGATE.ALIGN', False,
                'SEARCH.AGGREGATE.OP', 'entropy',
                'SEARCH.CRITERION', 'min_delta'
            ]
        ps_cfg = get_cfg(config_fname, opts)
        
        model = copy.deepcopy(self.pretrained_model)
        model = convert(model, ps_cfg, self.dm.num_classes)
        if learn:
            model.agg_layer.load_state_dict(torch.load(
                f'checkpoints/poolsearch/{self.dm.name}/{self.model_version}/{self.model_name}-agg.pt'
            ))

        if best:
            tta_transforms, _ = self.get_best_transforms(
                train_budget=128, test_budget=tta_budget)
        else:
            tta_transforms = self.tm.get_transforms(end_idx=tta_budget)

        tta_wrapper = WrapperTTA(model, tta_transforms)

        agg_f = self.get_agg_model(agg_name, tta_budget)
        agg_model = AggTTA(tta_wrapper, agg_f)
        agg_model = agg_model.to(self.device)
        
        result = self.test_model(
            agg_model, budget=tta_budget, 
            agg_name=agg_name + f'-{exp_name}({search_budget})')

        self.evaluator.update({
            'model': self.model_name,
            'budget': tta_budget * search_budget,
            'agg': f'Ours-{agg_name}',
            'top1': result['top1'], 
            'policy': self.tta_policy,
        })
        self.evaluator.log()
