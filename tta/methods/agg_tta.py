import torch
import torch.nn as nn

class AggTTA(nn.Module):

    def __init__(self, tta_wrapper, agg_model):
        super().__init__()
        self.tta_wrapper = tta_wrapper
        self.agg_model = agg_model
        
    def forward(self, x, tta_idxs=None):
        x = self.tta_wrapper(x) # N, A, K
        x = self.agg_model(x, tta_idxs)  
        return x