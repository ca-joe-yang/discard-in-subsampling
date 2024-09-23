import torch
import torch.nn as nn

class AugTTA(nn.Module):

    name = 'AugTTA'
    
    def __init__(self, 
        n_augs: int, 
        temperature: float = 1.0
    ):
        super().__init__()
        self.coeffs = nn.Parameter(
            torch.randn((n_augs, 1), requires_grad=True, dtype=torch.float))
        self.coeffs.data.fill_(1.0 / n_augs) 
        self.temperature = temperature        
        
    def forward(self, x, idxs=None):
        '''
        x: N, A, C, H, W
        coeffs: A, 1
        '''
        x = x / self.temperature

        if idxs is not None:
            coeffs = self.coeffs[idxs]
        else:
            coeffs = self.coeffs

        if len(x.shape) == 3:
            coeffs = coeffs.unsqueeze(0)#.expand_as(x)
        elif len(x.shape) == 5:
            coeffs = coeffs.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    
        x = (coeffs * x).sum(1) / coeffs.sum(1)
        return x
