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
# from search.node import PoolSearchStateNode
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

    def forward_search_single_img(self, img):
        # Initialize starting conditions for all imgs
        N = len(imgs)
        all_search_progress = []
        for _ in range(N):
            all_search_progress.append(
                PoolSearchProgress(self.budget, self.num_downsample))
        
        while True:
            if np.alltrue([p.is_end() for p in all_search_progress]): break
            all_states = np.full([N, self.num_downsample], -1)
            
            # Finding the next node to expand
            all_feats = torch.stack(
                [ torch.stack([ n.feat for n in c ], 0) for c in all_candidates], 0)
            all_logits = torch.stack(
                [ torch.stack([ n.logit for n in c ], 0) for c in all_candidates], 0)
            all_states = np.stack(
                [ np.stack([ n.state for n in c ], 0) for c in all_candidates], 0)
            
            scores = self.search_criterion(all_feats, all_logits, all_states)

            for i in range(len(imgs)):
                search_progress = all_search_progress[i]
                # print(i, search_progress)
                search_progress.next()
                # print(search_progress.expand_node, search_progress.expand_idx)
                if search_progress.expand_node is None:
                    all_states[i, :] == 0
                    continue
                
            num_expand = self.downsample_modules[search_progress.expand_idx].num_expand
            for j in range(num_expand):
                all_states[i, search_progress.expand_idx] = j

            # Forward pass for all images
            feats, logits = self.forward_state(imgs, all_states)
            # Expand the node given the feat and logit computed
            for i in range(len(imgs)):
                search_progress = all_search_progress[i]
                search_progress.expand(
                    self.get_neighboring_states, self.search_criterion,
                    feats[i][0], logits[i][0]
                )
        if self.use_fmap:
            feats = []
            for i in range(len(imgs)):
                candidates = all_search_progress[i].candidates[0:1]
                feat = self.aggregate_feats(candidates, method=self.aggregate_fn)
                feats.append(feat)
                # print(feat.shape)
            feats = torch.stack(feats, 0)
            # print(feats.shape)
            logits = self.model.forward_head(feats)
            return logits
        else:
            # logits = [ all_search_progress[i].candidates.logit for i in N ]
            all_logits = []
            for i in range(len(imgs)):
                candidates = all_search_progress[i].candidates
                logits = torch.stack([node.logit for node in candidates], 0)
                all_logits.append(logits)
            all_logits = torch.cat(all_logits, 0)
            all_logits = self.aggregate_logits(all_logits, method=self.aggregate_fn) 
            return all_logits

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


    # def _aggregate_feats(self, candidates, method=None):
    #     """
    #     Inputs
    #         queue: List of [Node(feats: [NCHW], logits: [NK]) ], length=B
    #         or
    #         queue: List of [Node(feats: [NLC], logits: [NK]) ], length=B
    #     """
    #     device = candidates[0].feat.device
    #     all_feats = torch.stack([ node.feat for node in candidates], 0) # B N C H W
    #     all_logits = torch.stack([ node.logit for node in candidates], 0) # B N K
    #     B = self.budget
    #     do_align = 'align' in method
    #     is_transformer = len(all_feats.shape) == 3
    #     if is_transformer:
    #         _, L, C = all_feats.shape
    #         h = w = int(np.sqrt(L))
    #     else:
    #         _, C, h, w = all_feats.shape
        
    #     if is_transformer:
    #         all_feats = all_feats.reshape(B, h, w, C).permute(0, 3, 1, 2) # NChw
        
    #     if 'weight' in method:
    #         entropy = torch_normalized_entropy(all_logits * self.logits_scale, dim=-1) #PN1
    #         if torch.isnan(entropy).any():
    #             print(all_logits.max())
    #         weights = (1. / entropy).unsqueeze(-1).unsqueeze(-1)
    #     elif 'cyclic' in method:
    #         entropy = torch_entropy(all_logits * self.logits_scale, dim=-1) #PN1
    #         if torch.isnan(entropy).any():
    #             print(all_logits.max())
    #         weights = (1. / F.softmax(entropy, dim=0)).unsqueeze(-1).unsqueeze(-1)
    #     elif 'avg' in method:
    #         weights = torch.ones([B, 1, 1, 1]).to(all_feats.device)
    #     elif 'best' in method:
    #         entropy = torch_normalized_entropy(all_logits * self.logits_scale, dim=-1) # NP1
    #         pred = torch.argmin(entropy, dim=0)
    #         weights = torch.zeros_like(entropy).scatter_(0, pred.unsqueeze(0), 1.).unsqueeze(-1).unsqueeze(-1)
    #     elif 'learn' in method:
    #         # print(all_feats.shape) # P, C, H, W
    #         # f = all_feats.view(B, C, -1).permute(2, 0, 1) # hw, P, C
    #         # print(f.shape)
    #         f, score = self.agg_layer(all_feats)#.unsqueeze(-1) # P, 1, 1
    #         # print(score[0,0,0])

    #         # weights = torch.ones([B, 1, 1, 1]).to(all_feats.device)
    #         # weights = weights.repeat(1, 1, h, w)
    #         # ret_feat = (all_feats * weights).sum(0) / torch.clamp(weights.sum(0), min=1e-12)

    #         # print(f.mean(1).view(h, w, C).shape)
    #         # f = f.mean(1).view(h, w, C).permute(2, 0, 1).unsqueeze(0)
    #         # print(w.shape)
    #         # w = F.softmax(w)
    #         # print(w[:,0, 0])
    #         # r = (w * all_feats).sum(0) / w.sum(0)
    #         return f
    #     else:
    #         raise ValueError(method)
    #     if do_align:
    #         total_R = self.max_shift[0]
    #         H = total_R * h
    #         W = total_R * w
    #         weights = weights.repeat(1, 1, H, W)
    #         ret_feat = torch.zeros([C, H, W]).to(device)
    #         count = torch.zeros([1, H, W]).to(device)
    #         for i in range(B):
    #             dx, dy = self.state2dxy(candidates[i].state)
    #             weight = F.pad(weights[i], (dx, 0, dy, 0))[..., :H, :W]
    #             resize_feat = T.functional.resize(
    #                 all_feats[i], (H, W), antialias=True,
    #                 interpolation=T.InterpolationMode.NEAREST)
    #             resize_feat = F.pad(resize_feat, (dx, 0, dy, 0))[..., :H, :W]
                
    #             ret_feat += resize_feat * weight
    #             count += weight
            
    #         ret_feat = ret_feat / torch.clamp(count, min=1e-12) # NCHW
    #         # ret_feat = T.functional.resize(
    #         #     ret_feat, (h, w), antialias=True)
    #         ret_feat = F.avg_pool2d(ret_feat, kernel_size=total_R, stride=total_R)
    #     else:
    #         weights = weights.repeat(1, 1, h, w)
    #         ret_feat = (all_feats * weights).sum(0) / torch.clamp(weights.sum(0), min=1e-12)

    #     if is_transformer:
    #         ret_feat = ret_feat.reshape(N, C, L).permute(0, 2, 1)

    #     return ret_feat

