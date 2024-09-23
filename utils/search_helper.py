import numpy as np
import torch
import torch.nn.functional as F
import random

def aggregate_majority(x, dim=None, topk=5):
    a = torch.argsort(-x, dim=-1) # [... K 1000]
    v = torch.mode(a, -2)[0] # [... 1000]
    x = torch.zeros_like(v) - 1.
    if len(x.shape) == 1: # [1000]
        for i, s in enumerate(v[:topk]):
            x[s] = (topk-i)/topk
    else: # [... 1000]
        for b in range(x.shape[0]):
            for i, s in enumerate(v[b][:topk]):
                x[b][s] = (topk-i)/topk
    return x

def get_shift_mask(x, factor=2, shift=[0, 0]):
    """

    """
    H, W = x.shape[-2:]
    m = torch.ones([1, H, W])
    L = [m] + [torch.zeros([1, H, W])]*(factor*factor-1)
    L = torch.stack(L, dim=1).reshape(1, factor*factor, H, W)
    y = F.pixel_shuffle(L, factor)
    dy, dx = shift
    y = F.pad(y, (dx, 0, dy, 0))[..., :H*factor, :W*factor]
    return y

def shift_img(img, dx: int, dy: int):
    H, W = img.shape[-2:]
    if dx > 0:
        if dy > 0:
            return F.pad(img, (dx, 0, dy, 0))[..., :H, :W]
        else:
            return F.pad(img, (dx, 0, 0, dy))[..., -H:, :W]
    else:
        if dy > 0:
            return F.pad(img, (0, dx, dy, 0))[..., :H, -W:]
        else:
            return F.pad(img, (0, dx, 0, dy))[..., -H:, -W:]

def dxy2s(dx, dy, L=[2,2,2,2,2]):
    assert dx < np.prod(L)
    assert dy < np.prod(L)

    X, Y = [], []
    for l in L[::-1]:
        r = dx % l
        q = dx // l
        dx = q
        X.append(r)

        r = dy % l
        q = dy // l
        dy = q
        Y.append(r)

    s = [ y*l+x for (x, y, l) in zip(X, Y, L) ]
    return s

def s2dxy(s, L=[2,2,2,2,2]):
    assert len(s) == len(L)

    dx, dy = 0, 0
    b = 1
    for i, (idx, l) in enumerate(zip(s, L)):
        if idx == -1: idx = 0
        x = idx % l
        y = idx // l
        dx += x * b
        dy += y * b
        b *= l
        
    return dx, dy




def set_dilation_padding(layer, multiply=1, divide=1):
    if isinstance(layer.dilation, int):
        layer.dilation = int(multiply*layer.dilation/divide)
    else:
        layer.dilation = (
            int(multiply*layer.dilation[0]/divide), 
            int(multiply*layer.dilation[1]/divide))
    if isinstance(layer.padding, int):
        layer.padding = int(multiply*layer.padding/divide)
    else:
        layer.padding = (
            int(multiply*layer.padding[0]/divide),
            int(multiply*layer.padding[1]/divide))

def yield_choices(lst):
    if lst:
        for el in lst[0]:
            for combo in yield_choices(lst[1:]):
                yield [el] + combo
    else:
        yield []

