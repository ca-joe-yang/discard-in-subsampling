import math
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision
import torchvision.transforms as T

from utils.math_helper import torch_normalized_entropy, torch_entropy
from models.base import PoolSearchModelBase
from search.progress import PoolSearchProgress
from .attention import MultiHeadAttention

class PoolSearchModelCls(PoolSearchModelBase):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def build(self, cfg):
        super().build(cfg.SEARCH)

        self.agg_layer = MultiHeadAttention(
            in_dim=self.model.feat_dim, 
            num_heads=1,
            dropout=0.1
        )

    def train(self, mode: bool=True) -> None:
        self.agg_layer.train(mode)
        self.model.eval()

    def forward(self, images: Tensor) -> Tensor:
        """
        Inputs:
            I: [N, C, H, W]
        Return:
            logits: [N, P, C]
        """
        if self.budget == 1:
            self.set_state(np.zeros([len(images), self.num_downsample]))
            return self.model.forward_head(self.model.forward_features(images))

        if self.cfg.ONE_BY_ONE:
            raise
            return self.forward_search_single_img(img)
        
        return self.foward_search_whole_batch(images)

    def foward_search_whole_batch(self, imgs: Tensor) -> Tensor:
        # Initialize starting conditions for all imgs
        batch_size = len(imgs)
        all_search_progress = []
        for _ in range(batch_size):
            all_search_progress.append(
                PoolSearchProgress(self.budget, self.num_downsample))
        
        first = True
        while True:
            if np.alltrue([p.is_end() for p in all_search_progress]): break
            all_states = []
            all_images = []
            all_expidx = []
            all_expanded_states = []
            for e in range(self.num_downsample):
                all_images.append([])
                all_states.append([])
                all_expidx.append({})
                all_expanded_states.append([])
            # Finding the next node to expand

            if not first:
                all_candidates = [ s.candidates for s in all_search_progress ]
                all_feats = torch.stack(
                    [ torch.stack([ n.feat for n in c ], 0) for c in all_candidates], 0)
                all_logits = torch.stack(
                    [ torch.stack([ n.logit for n in c ], 0) for c in all_candidates], 0)
                all_cstates = np.stack(
                    [ np.stack([ n.state for n in c ], 0) for c in all_candidates], 0)
                
                scores = self.search_criterion(all_cstates, all_feats, all_logits)
                # print(scores.shape)
                for i in range(scores.shape[0]):
                    for j in range(scores[i].shape[0]):
                        all_search_progress[i].candidates[j].score = scores[i][j]

            for i in range(len(imgs)):
                search_progress = all_search_progress[i]
                # print(i, search_progress)
                search_progress.next()
                if search_progress.expand_node is None:
                    print(search_progress.is_end())
                    print(search_progress)
                    raise
                search_progress.get_expand_neighbors(
                    self.downsample_modules[search_progress.expand_idx].num_expand)
                
                expand_idx = search_progress.expand_idx
                all_expidx[expand_idx][len(all_images[expand_idx])] = i
                all_images[expand_idx].append(imgs[i])
                state = search_progress.expand_node.state.copy()
                state[expand_idx] = search_progress.num_expand
                all_states[expand_idx].append(state)
                all_expanded_states[expand_idx].append(search_progress.neighboring_states)
            # Forward pass for all images
            for e in range(self.num_downsample):
                if len(all_images[e]) != 0:
                    e_feats, e_logits = self.forward_state(
                        torch.stack(all_images[e], 0), np.array(all_states[e]))
                    # e_scores = self.search_criterion(
                    #     all_expanded_states[e], e_feats, e_logits)
                    for j in range(e_logits.shape[0]):
                        i = all_expidx[e][j]
                        # Expand the node given the feat and logit computed
                        search_progress = all_search_progress[i]
                        search_progress.expand(
                            e_feats[j], e_logits[j]
                        )
            first = False
        all_candidates = [ p.candidates for p in all_search_progress ]
        match self.aggregate_mode:
            case 'features':
                feats = self.forward_aggregate_features(all_candidates)
                logits = self.model.forward_head(feats)
                return logits
            case 'logits':
                logits = self.forward_aggregate_logits(all_candidates) 
                return logits
            case _:
                raise ValueError(self.aggregate_mode) 

    @torch.no_grad
    def forward_state(self, 
        imgs: Tensor, 
        state
    ) -> tuple[Tensor, Tensor]:
        """
        Input
        : NCHW
        : ND
        Return
        : NDK
        or
        : NDChw
        """
        N = imgs.shape[0]
        state = np.array(state)
        if len(state.shape) == 1:
            state = np.repeat(np.expand_dims(state, 0), N, axis=0)
        self.set_state(state)
        feats = self.model.forward_features(imgs)
        
        match len(feats.shape):
            case 4:
                # CNN
                logits = self.model.forward_head(feats)
                
                NE, C, H, W = feats.shape
                feats = feats.view(N, -1, C, H, W)#.permute(1, 0, 2, 3, 4)
                
                NE, K = logits.shape
                logits = logits.view(N, -1, K)#.permute(1, 0, 2)
                return feats, logits
            case 3:
                raise
                # Transformers
                NE, L, C = feats.shape
                feats = feats.view(N, -1, L, C).permute(1, 0, 2, 3)
                logits = torch.stack([self.model.forward_head(feat) for feat in feats ], 0)
                return feats, logits
            case _:
                raise NotImplementedError

    def forward_aggregate_features(self, all_candidates: list) -> Tensor:
        r"""
        - Args:
            - all_candidates: {N, P} {feat[CHW]; logit [CK]; state;}
        """
        # device = all_candidates[0][0].feat.device
        all_feats = torch.stack(
            [ torch.stack([ n.feat for n in c ], 0) for c in all_candidates], 0)
        all_logits = torch.stack(
            [ torch.stack([ n.logit for n in c ], 0) for c in all_candidates], 0)
        all_states = np.stack(
            [ np.stack([ n.state for n in c ], 0) for c in all_candidates], 0)
        N, P = all_states.shape[:2]
        
        deltas = torch.zeros([N, P, 2]).long().to(all_feats.device)
        for i in range(N):
            for j in range(P):
                dx, dy = self.state2dxy(all_states[i][j])
                deltas[i, j, 0] = dx
                deltas[i, j, 1] = dy

        if self.aggregate_align:
            all_feats = self.forward_align(all_feats, all_states)

        match self.aggregate_fn:
            case 'learn':
                feats, score = self.agg_layer(all_feats, all_logits, deltas)
                return feats.mean(1)
            case 'entropy':
                # print(all_logits.shape)
                entropy = torch_normalized_entropy(all_logits, dim=-1) #NP1
                weights = 1. - entropy
                weights = weights[..., None, None]#.unsqueeze(-1).unsqueeze(-1)
                # print(weights.shape, all_feats.shape)
                # raise
                return torch.sum(weights * all_feats, dim=1) / torch.sum(weights, dim=1)
            case 'avg':
                return all_feats.mean(1)
            case _:
                raise ValueError(self.aggregate_op)

    def forward_align(self, 
        all_feats: Tensor, 
        all_states: np.ndarray
    ):
        """
        - Args:
            - `all_feats`: [batch_size, channel, height, width] or [batch_size, token_length, channel]
        """
        N = all_feats.shape[0]
        device = all_feats.device
        is_transformer = len(all_feats.shape) == 3
        if is_transformer:
            L, C = all_feats.shape[-2:]
            h = w = int(np.sqrt(L))
        else:
            C, h, w = all_feats.shape[-3:]
        
        # if is_transformer:
        #     all_feats = all_feats.reshape(B, h, w, C).permute(0, 3, 1, 2) # NChw
        
        total_R = self.max_shift[0]
        H = total_R * h
        W = total_R * w
        ret_feat = torch.zeros([N, self.budget, C, h, w]).to(device)
        # count = torch.zeros([1, H, W]).to(device)
        for i in range(N):
            resize_feats = T.functional.resize(
                all_feats[i], (H, W), antialias=True,
                interpolation=T.InterpolationMode.NEAREST)
            for j in range(self.budget):
                dx, dy = self.state2dxy(all_states[i][j])
                resize_feat = F.pad(resize_feats[j], (dx, 0, dy, 0))[..., :H, :W]
                ret_feat[i, j] = F.avg_pool2d(
                    resize_feat, kernel_size=total_R, stride=total_R)

        return ret_feat
