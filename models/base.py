import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch import Tensor

from utils.math_helper import torch_normalized_entropy, torch_entropy, majority_vote
from utils.search_helper import s2dxy
from modules.register import get_pool_search_module

class PoolSearchModelBase(nn.Module):

    def build(self, cfg):
        self.cfg = cfg
        self.budget = cfg.BUDGET
        self.aggregate_fn = cfg.AGGREGATE.OP
        self.aggregate_mode = cfg.AGGREGATE.MODE
        self.aggregate_align = cfg.AGGREGATE.ALIGN
        self.criterion = cfg.CRITERION
        self.downsample_modules = []
        for i, module in enumerate(self.model.downsample_layers):
            if i in cfg.SEARCH_SPACE: # NOTE: must be consecutive
                self.downsample_modules.append(
                    self.replace_downsample_module(module))
        self.num_downsample = len(self.downsample_modules)
        
        self.max_shift = [1, 1]
        for module in self.downsample_modules:
            self.max_shift[0] *= module.R
            self.max_shift[1] *= module.R
        
    def get_layer(self, name, delimiter='/', return_pair=False):
        ret = self.model
        tokens = name.split(delimiter)
        if return_pair:
            child_name = tokens[-1]
            tokens = tokens[:-1]
        for token in tokens:
            ret = getattr(ret, token)
        if return_pair:
            return ret, child_name, getattr(ret, child_name)
        return ret
    
    def replace_downsample_module(self, old_module):
        module_name = old_module[0]
        module_type = old_module[1]
        module_args = old_module[2:]
        parent, child_name, child = self.get_layer(
            module_name, return_pair=True)
        new_module = get_pool_search_module('PoolSearch' + module_type)(child, *module_args)
        setattr(parent, child_name, new_module)
        return new_module

    def set_state(self, state) -> None:
        """
        :params imgs_state [N num_downsample]
        """
        for i, module in enumerate(self.downsample_modules):
            module.set_select_indices(state[:, i])
    
    def search_criterion(self, 
        states, 
        feats: Tensor, 
        logits: Tensor,
    ) -> Tensor:
        """
        logit: C
        feat: P, N, C, h, w
        Return
            score: [Batch_size x Budget]
        """
        N, P, _ = np.array(states).shape
        deltas = torch.zeros([N, P, 2]).long().to(logits.device)
        for i in range(N):
            for j in range(P):
                dx, dy = self.state2dxy(states[i][j])
                deltas[i, j, 0] = dx
                deltas[i, j, 1] = dy
        
        match self.criterion:
            case 'min_delta':
                return deltas.sum(-1)
            case 'max_delta':
                return -deltas.sum(-1)
            case 'min_entropy':
                return torch_normalized_entropy(logits, dim=-1)
            case 'random':
                return torch.rand(N, P).cuda()
            case 'random-max_learn':
                scores1 = torch.rand(N, P).cuda()
                scores2 = 1. / self.agg_layer(feats, logits, deltas)[1]
                return scores1 * scores2
            case 'random-min_learn':
                scores1 = torch.rand(N, P).cuda()
                scores2 = self.agg_layer(feats, logits, deltas)[1]
                return scores1 * scores2
            case 'max_learn':
                return 1. / self.agg_layer(feats, logits, deltas)[1]
            case 'min_learn':
                return self.agg_layer(feats, logits, deltas)[1]
            case _:
                raise ValueError(self.criterion)

    def get_neighboring_states(self, state, i) -> list:
        """
        neighboring states including itself
        """
        neighboring_states = []
        for j in range(self.downsample_modules[i].num_expand):
            neighboring_states.append(state[:i] + [j] + state[i+1:])
        return neighboring_states

    def forward_aggregate_logits(self, all_candidates):
        """
        :params P N K
            - inputs: 
                [P N K]
                [P N K H W]
                [P N K L]
        """
        all_logits = torch.stack([ torch.stack([ n.logit for n in c ], 0) for c in all_candidates], 0)
        all_states = np.stack([ np.stack([ n.state for n in c ], 0) for c in all_candidates], 0)
        
        N, P = all_logits.shape[:2]

        deltas = torch.zeros([N, P, 2]).long().to(all_logits.device)
        for i in range(N):
            for j in range(P):
                dx, dy = self.state2dxy(all_states[i][j])
                deltas[i, j, 0] = dx
                deltas[i, j, 1] = dy

        match self.aggregate_fn:
            case 'learn':
                logits, score = self.agg_layer(all_logits, all_logits, deltas)
                return logits
            case 'avg':
                return all_logits.mean(1)
            case 'entropy':
                entropy = torch_entropy(logits, dim=-1) / np.log(self.num_classes)
                weights = 1. - entropy
                weights = weights / torch.sum(weights, dim=1, keepdims=True)
                return torch.sum(weights * all_logits, dim=1)
            case _:
                raise ValueError(self.aggregate_fn)
        
    def state2dxy(self, state):
        return s2dxy(state, [module.R for module in self.downsample_modules])
