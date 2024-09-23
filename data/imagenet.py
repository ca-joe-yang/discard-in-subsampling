import os
import random
import numpy as np

import torch
import torchvision
import torchvision.transforms as T

from .base import BaseDataModule

class ImageNet(BaseDataModule):

    name = 'imagenet'
    num_classes = 1000

    # torchvision
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def __init__(self, cfg, tta=False):
        super().__init__(cfg, tta)

        self.crop_size = cfg.DATA.INPUT.CROP_SIZE
        self.orig_size = cfg.DATA.INPUT.ORIG_SIZE

        if self.tta:
            test_preprocess = T.Compose([
                T.Resize((self.orig_size, self.orig_size)),
                T.ToTensor(),
                self.normalize
            ])
            train_preprocess = test_preprocess

        else:
            train_preprocess = T.Compose([
                T.RandomResizedCrop(self.crop_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                self.normalize,
            ])

            test_preprocess = T.Compose([
                T.Resize(self.orig_size,
                    getattr(T.InterpolationMode, cfg.DATA.INPUT.INTERPOLATE)
                ),
                T.CenterCrop(self.crop_size),
                T.ToTensor(),
                self.normalize,
            ])

        self.image_dir = os.path.join(cfg.DATA.ROOT, 'imagenet')
        
        train_set = torchvision.datasets.ImageNet(
            root=self.image_dir, split='train',
            transform=train_preprocess)
        val_set = torchvision.datasets.ImageNet(
            root=self.image_dir, split='train',
            transform=test_preprocess)
        self.test_dataset = torchvision.datasets.ImageNet(
            root=self.image_dir, split='val',
            transform=test_preprocess)
        
        n_train = 20000
        n_val = 5000
        
        train_indices = np.random.choice(
            range(len(train_set)), size=n_val+n_train, replace=False)
        self.train_dataset = torch.utils.data.Subset(train_set, train_indices[:n_train])
        self.val_dataset = torch.utils.data.Subset(val_set, train_indices[n_train:])
        self.build_loader()
