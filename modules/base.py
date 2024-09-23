import numpy as np
import torch
import torch.nn as nn

class BaseModule(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.select_indices = None
        self.child_modules = []

    def set_select_indices(self, select_indices):
        self.select_indices = select_indices
        for module in self.child_modules:
            module.set_select_indices(select_indices)

    def _forward(self, x, *args, **kwargs):
        if np.all(np.array(self.select_indices) == self.select_indices[0]):
            return self.forward_single(x, self.select_indices[0], *args, **kwargs)        
        
        y = []
        NP, C1, H1, W1 = x.shape
        N = len(self.select_indices)
        if NP > N:
            x = x.view(N, -1, C1, H1, W1)#.permute(1, 0, 2, 3, 4)
        for b_x, select_idx in zip(x, self.select_indices):
            if len(b_x.shape) == 3:
                b_x = b_x.unsqueeze(0)
            b_y = self.forward_single(b_x, select_idx, *args, **kwargs) 
            y.append(b_y)
        y = torch.cat(y, 0) 

        return y
        

