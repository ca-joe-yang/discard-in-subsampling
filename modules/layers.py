import math
import numpy as np

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModule
from .register import register

@register
class PoolSearchConv2d(BaseModule):

    def __init__(self, conv2d, new_stride=(1, 1)):
        super().__init__()

        if isinstance(conv2d.stride, int):
            conv2d.stride = (conv2d.stride, conv2d.stride)
        if isinstance(conv2d.padding, int):
            conv2d.padding = (conv2d.padding, conv2d.padding)     
        assert(conv2d.stride[0] == conv2d.stride[1])
        assert(new_stride[0] == new_stride[1])
        assert(conv2d.stride[0] % new_stride[0] == 0)
        
        assert(conv2d.padding[0] == conv2d.padding[1])
        assert(conv2d.padding_mode == 'zeros')
        
        self.old_stride = conv2d.stride[0]
        self.old_padding = conv2d.padding
        self.new_stride = new_stride
        
        self.R = conv2d.stride[0] // new_stride[0]
        self.num_expand = self.R * self.R
        self.pixel_unshuffle = nn.PixelUnshuffle(self.R)

        self.conv2d = conv2d
        self.conv2d.padding = (0, 0)

    def forward_single(self, b_x, select_idx):
        if select_idx == 0 or select_idx == -1: # Original
            self.conv2d.stride = (self.old_stride, self.old_stride)
            b_y = self.conv2d(b_x)
        elif select_idx == self.num_expand: # Expand all
            self.conv2d.stride = self.new_stride
            new_shape_x = (b_x.shape[-1] - self.conv2d.kernel_size[1]) // self.new_stride[1] + 1
            new_shape_y = (b_x.shape[-2] - self.conv2d.kernel_size[0]) // self.new_stride[0] + 1
            remainder_x = int(new_shape_x) % self.R
            remainder_y = int(new_shape_y) % self.R
            pad_x = self.R - remainder_x if remainder_x > 0 else 0
            pad_y = self.R - remainder_y if remainder_y > 0 else 0
            b_x = F.pad(b_x, [0, pad_x, 0, pad_y])
            b_y = self.conv2d(b_x)
            b_y = self.pixel_unshuffle(b_y.unsqueeze(2))
            N, C, P, H, W = b_y.shape
            b_y = b_y.permute(0, 2, 1, 3, 4).reshape(N*P, C, H, W)
        else: #1, 2,3
            self.conv2d.stride = (self.old_stride, self.old_stride)
            dw = select_idx % self.R
            dh = select_idx // self.R
            H, W = b_x.shape[-2:]
            b_x = F.pad(b_x, [0, dw, 0, dh])[..., -H:, -W:]
            b_y = self.conv2d(b_x)
        return b_y

    def forward(self, x: torch.Tensor):
        
        x = F.pad(x, 
            [self.old_padding[0], self.old_padding[1], 
             self.old_padding[0], self.old_padding[1]])
        
        return self._forward(x)
        
@register
class PoolSearchMaxPool2d(BaseModule):

    def __init__(self, maxpool, new_stride=1):
        super().__init__()

        self.new_stride = new_stride
        if isinstance(maxpool.kernel_size, int):
            maxpool.kernel_size = (maxpool.kernel_size, maxpool.kernel_size)
        self.old_stride = maxpool.stride
        self.old_padding = maxpool.padding

        assert(self.old_stride % self.new_stride == 0)
        self.R = self.old_stride // self.new_stride
        self.num_expand = self.R * self.R
        self.pixel_unshuffle = nn.PixelUnshuffle(self.R)

        self.maxpool = maxpool        
        self.maxpool.padding = 0

    def forward_single(self, b_x, select_idx):
        if select_idx == 0 or select_idx == -1:
            self.maxpool.stride = self.old_stride
            b_y = self.maxpool(b_x)
        elif select_idx == self.num_expand: #all
            self.maxpool.stride = self.new_stride
            new_shape_x = (b_x.shape[-1] - self.maxpool.kernel_size[1]) // self.maxpool.stride + 1
            new_shape_y = (b_x.shape[-2] - self.maxpool.kernel_size[0]) // self.maxpool.stride + 1
            remainder_x = int(new_shape_x) % self.R
            remainder_y = int(new_shape_y) % self.R
            pad_x = self.R - remainder_x if remainder_x > 0 else 0
            pad_y = self.R - remainder_y if remainder_y > 0 else 0
            b_x = F.pad(b_x, [0, pad_x, 0, pad_y], value=-float('inf'))
            b_y = self.maxpool(b_x)
            b_y = self.pixel_unshuffle(b_y.unsqueeze(2))
            N, C, P, H, W = b_y.shape
            b_y = b_y.permute(0, 2, 1, 3, 4).reshape(N*P, C, H, W)
        else: #1,2,3
            self.maxpool.stride = self.old_stride
            dw = select_idx % self.R
            dh = select_idx // self.R
            H, W = b_x.shape[-2:]
            b_x = F.pad(b_x, [0, dw, 0, dh], value=-float('inf'))[..., -H:, -W:]
            b_y = self.maxpool(b_x)
        return b_y

    def forward(self, x: torch.Tensor):
        x = F.pad(x, [
            self.old_padding, self.old_padding, 
            self.old_padding, self.old_padding], value=-float('inf'))

        return self._forward(x)


@register
class PoolSearchAvgPool2d(BaseModule):

    # TODO: take care edge cases
    def __init__(self, avgpool, new_stride=1):
        super().__init__()

        self.new_stride = new_stride
        if isinstance(avgpool.kernel_size, int):
            avgpool.kernel_size = (avgpool.kernel_size, avgpool.kernel_size)
        self.old_stride = avgpool.stride
        self.old_padding = avgpool.padding

        assert(self.old_stride % self.new_stride == 0)
        self.R = self.old_stride // self.new_stride
        self.num_expand = self.R * self.R
        self.pixel_unshuffle = nn.PixelUnshuffle(self.R)

        self.avgpool = avgpool        
        self.avgpool.padding = 0

    def forward_single(self, b_x, select_idx):
        if select_idx == 0 or select_idx == -1:
            self.avgpool.stride = self.old_stride
            b_y = self.avgpool(b_x)
        elif select_idx == self.num_expand: #all
            self.avgpool.stride = self.new_stride
            new_shape_x = (b_x.shape[-1] - self.avgpool.kernel_size[1]) // self.avgpool.stride + 1
            new_shape_y = (b_x.shape[-2] - self.avgpool.kernel_size[0]) // self.avgpool.stride + 1
            remainder_x = int(new_shape_x) % self.R
            remainder_y = int(new_shape_y) % self.R
            pad_x = self.R - remainder_x if remainder_x > 0 else 0
            pad_y = self.R - remainder_y if remainder_y > 0 else 0
            m_x = torch.ones_like(b_x).to(b_x)
            b_x = F.pad(b_x, [0, pad_x, 0, pad_y], value=float(0))
            m_x = F.pad(m_x, [0, pad_x, 0, pad_y], value=float(0))
            b_y = self.avgpool(b_x) / self.avgpool(m_x)
            b_y = self.pixel_unshuffle(b_y.unsqueeze(2))
            N, C, P, H, W = b_y.shape
            b_y = b_y.permute(0, 2, 1, 3, 4).reshape(N*P, C, H, W)
        else: #1,2,3
            self.avgpool.stride = self.old_stride
            dw = select_idx % self.R
            dh = select_idx // self.R
            H, W = b_x.shape[-2:]
            m_x = torch.ones_like(b_x).to(b_x)
            b_x = F.pad(b_x, [0, dw, 0, dh], value=float(0))[..., -H:, -W:]
            m_x = F.pad(m_x, [0, dw, 0, dh], value=float(0))[..., -H:, -W:]
            b_y = self.avgpool(b_x) / self.avgpool(m_x)
        return b_y

    def forward(self, x: torch.Tensor):
        x = F.pad(x, [
            self.old_padding, self.old_padding, 
            self.old_padding, self.old_padding], value=-float('inf'))

        return self._forward(x)

@register
class PoolSearchPatchMerging(BaseModule):
    
    def __init__(self, patch_merging, version='torchvision'):
        super().__init__()

        self.R = 2
        self.num_expand = self.R * self.R
        self.pixel_unshuffle = nn.PixelUnshuffle(self.R)

        self.patch_merging = patch_merging

        self.version = version

    def forward(self, x: torch.Tensor, input_size=None):
        return self._forward(x, input_size=input_size)

    def forward_original(self, x, input_size=None):
        if input_size is None:
            return self.patch_merging(x)
        return self.patch_merging(x, input_size)

    def shift_idx(self, b_x, select_idx):
        dw = select_idx % self.R
        dh = select_idx // self.R
        H, W, C = b_x.shape[-3:]
            
        b_x = F.pad(b_x, [0, 0, 0, dw, 0, dh])[..., -H:, -W:, :]
        if self.version == 'timm':
            b_x = b_x.reshape(-1, H*W, C)
        return b_x

    def forward_single(self, b_x, select_idx, input_size=None):
        if select_idx == 0 or select_idx == -1:
            return self.forward_original(b_x, input_size)

        if self.version == 'timm':
            if input_size is not None:
                H, W = input_size
            else:
                H, W = self.patch_merging.input_resolution
            B, L, C = b_x.shape
            b_x = b_x.view(B, H, W, C)
        else:
            B, H, W, C = b_x.shape

        if select_idx == self.num_expand: #all        
            X = []
            for idx in range(self.num_expand):
                xi = self.shift_idx(b_x, idx)
                X.append(xi)
            X = torch.stack(X, dim=1)
            if self.version == 'timm':
                X = X.reshape(B*self.num_expand, L, C)
            else:
                X = X.reshape(B*self.num_expand, H, W, C)
            return self.forward_original(X, input_size)
        
        b_x = self.shift_idx(b_x, select_idx)
        return self.forward_original(b_x, input_size)


def get_padding_mode(pad_layer):
    if isinstance(pad_layer, nn.ReflectionPad2d):
        return 'reflect'
    elif isinstance(pad_layer, nn.ReplicationPad2d):
        return 'replicate'
    elif isinstance(pad_layer, nn.ZeroPad2d):
        return 'zero'
    raise ValueError('Pad type [%s] not recognized'%pad_layer)

class PoolSearchBlurPool(BaseModule):
    def __init__(self, blurpool, downsample_id=0, new_stride=(1, 1), **kwargs):
        super().__init__()

        self.downsample_id = downsample_id
        self.fix = kwargs.get('fix', False)

        if isinstance(blurpool.stride, int):
            blurpool.stride = (blurpool.stride, blurpool.stride)        
        assert(blurpool.stride[0] == blurpool.stride[1])
        assert(new_stride[0] == new_stride[1])
        assert(blurpool.stride[0] % new_stride[0] == 0)
        
        self.old_stride = blurpool.stride
        self.new_stride = new_stride
        
        self.R = ratio = self.old_stride[0] // self.new_stride[0]
        self.num_expand = self.R * self.R
        self.downsample_ratio = (self.R, self.R)
        self.pixel_unshuffle = nn.PixelUnshuffle(self.R)

        self.blurpool = blurpool
        self.padding_mode = get_padding_mode(self.blurpool.pad)

    def forward(self, x: Tensor):
        config_idx = self.config.state[self.downsample_id]

        if config_idx == 0:
            self.blurpool.stride = self.old_stride[0]
            y = self.blurpool(x)
            return y   
        elif config_idx == 'all':
            self.blurpool.stride = self.new_stride[0]
            new_shape_w = x.shape[-1]
            new_shape_h = x.shape[-2]
            remainder_w = int(new_shape_w) % self.R
            remainder_h = int(new_shape_h) % self.R
            if remainder_w > 0:
                x = F.pad(x, [0, self.R - remainder_w, 0, 0], mode=self.padding_mode)
            if remainder_h > 0:
                x = F.pad(x, [0, 0, 0, self.R - remainder_h], mode=self.padding_mode)
            y = self.blurpool(x)
            y = self.pixel_unshuffle(y.unsqueeze(2))
            B, C, P, H, W = y.shape
            y = y.permute(0, 2, 1, 3, 4).reshape(B*P, C, H, W)
            return y
        elif isinstance(config_idx, int) and config_idx in range(self.num_expand):
            self.blurpool.stride = self.old_stride[0]
            dx = config_idx % self.R
            dy = config_idx // self.R
            H, W = x.shape[-2:]
            x = F.pad(x, [0, dx, 0, dy], mode=self.padding_mode)[..., -H:, -W:]
            y = self.blurpool(x)
            return y
        else:
            raise ValueError(config_idx)