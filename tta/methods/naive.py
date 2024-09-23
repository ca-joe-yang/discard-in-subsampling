import numpy as np

import torch.nn as nn
from torch import Tensor

class MeanTTA(nn.Module):

    def __init__(self):
        super().__init__()
        self.name = 'Mean'
    
    def forward(self, x: Tensor) -> Tensor:
        return x.mean(dim=1)

class MaxTTA(nn.Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        n_examples = x.shape[0]
        return x[np.arange(n_examples), x.max(axis=2).values.max(axis=1).indices, :]
