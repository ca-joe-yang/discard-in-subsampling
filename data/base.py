import os
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

import torch
import torch.nn as nn

class BaseDataModule:

    def __init__(self, cfg, tta=False):
        try:
            self.batch_size_train = self.batch_size_eval = cfg.DATALOADER.BATCH_SIZE
        except:
            self.batch_size_train = cfg.DATALOADER.BATCH_SIZE_TRAIN
            self.batch_size_eval = cfg.DATALOADER.BATCH_SIZE_EVAL
        self.num_workers = 8
        self.tta = tta
        
        self.normalize = T.Normalize(
            mean=self.mean,
            std=self.std)

        # if cfg.MODEL.BACKBONE == 'inception_v3':
        #     self.orig_size = 342
        #     self.crop_size = 299

    def build_loader(self, weighted_sampler=False):
        if weighted_sampler:
            self.train_loader = DataLoader(self.train_dataset,
                batch_size=self.batch_size_train,
                num_workers=self.num_workers, pin_memory=False,
                sampler=self.weighted_random_sampler())
        else:
            self.train_loader = DataLoader(self.train_dataset,
                batch_size=self.batch_size_train, shuffle=not self.tta,
                num_workers=self.num_workers, pin_memory=False)
        self.val_loader = DataLoader(self.val_dataset,
            batch_size=self.batch_size_eval, shuffle=False,
            num_workers=self.num_workers, pin_memory=False)
        self.test_loader = DataLoader(self.test_dataset,
            batch_size=self.batch_size_eval, shuffle=False,
            num_workers=self.num_workers, pin_memory=False)

    def weighted_random_sampler(self):
        class_sample_counts = [0 for x in range(self.num_classes)]
        train_targets = []

        for x in self.train_dataset:
            train_targets.append(x[1])
            class_sample_counts[x[1]] += 1

        weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
        samples_weights = weights[train_targets]

        sampler = torch.utils.data.WeightedRandomSampler(
            weights=samples_weights,
            num_samples=len(samples_weights))
        return sampler