import numpy as np
import torch
import torch.nn.functional as F

def numpy_entropy(x, keepdims=True):
    p = numpy_softmax(x, axis=-1)
    return np.sum(-p * np.log(p), axis=-1, keepdims=keepdims)

def numpy_softmax(x, axis=None):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def torch_entropy(x, dim=-1):
    p = F.softmax(x, dim=dim)
    return torch.sum(-p * torch.log(p), dim=dim, keepdims=True)

def torch_normalized_entropy(x, dim=-1):
    p = F.softmax(x, dim=dim)
    return torch.sum(-p * torch.log(p), dim=dim, keepdims=True) / np.log(x.shape[-1])

def majority_vote(x, topk=None):
    if topk is None:
        topk = x.shape[-1]
    x = x.permute(1, 0, 2)
    a = torch.argsort(-x, dim=-1) # [... P K]
    v = torch.mode(a, -2)[0] # [... K]
    x = torch.zeros_like(v) - 1.
    if len(x.shape) == 1: # [K]
        for i, s in enumerate(v[:topk]):
            x[s] = (topk-i)/topk
    else: # [... K]
        for b in range(x.shape[0]):
            for i, s in enumerate(v[b][:topk]):
                x[b][s] = (topk-i)/topk
    return x
