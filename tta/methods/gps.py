import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class GPS(nn.Module):
    def __init__(self, n_subpolicies: int):
        super().__init__()
        self.temperature = torch.nn.Parameter(torch.ones(1)[0])
        self.n_subpolicies = n_subpolicies
        idxs = torch.zeros(n_subpolicies).long()
        self.register_buffer('idxs', idxs)
        self.name = 'GPS'
    
    def forward(self, x: Tensor) -> Tensor:
        return torch.mean(x[:, self.idxs], dim=1)
    
    def find_temperature(self, logits, labels, device):
        logits = logits[:, 0]
        logits, labels = logits.to(device), labels.to(device)
        self.temperature.to(device)

        nll_criterion = nn.CrossEntropyLoss().to(device)
        
        optimizer = torch.optim.LBFGS(
            [self.temperature], lr=0.01, max_iter=50)
        def closure():
            optimizer.zero_grad()
            loss = nll_criterion(logits/self.temperature, labels)
            loss.backward()
            return loss
        optimizer.step(closure)
        # print(self.temperature)
         
    @torch.no_grad
    def find_idxs(self, logits, labels, device):         
        outputs = logits.permute(1, 0, 2)#.cpu().numpy()
        nll_criterion = nn.CrossEntropyLoss().to(device)

        outputs = outputs / self.temperature
        n_augs, n_examples, n_classes = outputs.shape
        remaining_idxs = list(np.arange(n_augs))
        outputs = F.softmax(outputs, dim=-1)
        # softmaxed = [F.softmax(aug_o, dim=1)[None, ...] for aug_o in outputs]
        # outputs = np.concatenate(softmaxed, axis=0)
        current_preds = torch.zeros((n_examples, n_classes))
        for i in range(self.n_subpolicies):
            print(i)
            aug_outputs = outputs[remaining_idxs]
            # should be of shape number of remaining augs, 
            old_weight = i / self.n_subpolicies
            new_weight = 1 - old_weight
            # calculate NLL for each possible output
            nll_vals = []
            for j in range(len(remaining_idxs)):
                possible_outputs = new_weight*aug_outputs[j] + old_weight* current_preds
                # possible_outputs /= possible_outputs.sum(1, keepdims=True)
                nll_vals.append(nll_criterion(possible_outputs, labels).item())
            next_idx = remaining_idxs[torch.argmin(torch.FloatTensor(nll_vals)).item()]
            remaining_idxs.remove(next_idx)
            self.idxs[i] = next_idx
            current_preds = new_weight * outputs[next_idx] + old_weight * current_preds
