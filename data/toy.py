from PIL import Image
import os
import glob
import numpy as np

import PIL
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split

import torch
import torch.nn as nn

class Toy:
    
    name = 'toy'
    num_classes = 4
    orig_size = 256
    crop_size = 224
    mean = [0.4137334859705493, 0.2875161046291306, 0.21944456761001857]
    std = [0.2981972911842248, 0.2206355698761603, 0.17820205228247024]

    def __init__(self):
        
        self.normalize = T.Normalize(
            mean=self.mean,
            std=self.std)

        self.transform = T.Compose([
            T.Resize((self.orig_size, self.orig_size)),
            T.ToTensor(),
            self.normalize
        ])

    def unnormalize(self, x):
        std = torch.FloatTensor(self.std).to(x.device).view(3, 1, 1)
        mean = torch.FloatTensor(self.mean).to(x.device).view(3, 1, 1)
        return x.mul_(std).add_(mean)
