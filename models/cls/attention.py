import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 32):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        d_model = d_model // 2

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.) / d_model))
        pe = torch.zeros(max_len, 2, d_model)
        self.pe = nn.Parameter(pe)

    def forward(self, deltas):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        pe1 = self.pe[deltas[..., 0], 0]
        pe2 = self.pe[deltas[..., 1], 1]
        pe = torch.cat([pe1, pe2], -1)
        return pe

def init_layer(layer, w, b):
    nn.init.normal_(layer.bias, std=b)
    if w == 'eye':
        nn.init.eye_(layer.weight)
    elif w == 'one':
        nn.init.ones_(layer.weight)
    elif w == 'kaiming':
        nn.init.kaiming_normal_(layer.weight)
    else:
        nn.init.xavier_normal_(layer.weight, gain=w)

class MultiHeadAttention(nn.Module):
    
    def __init__(self, in_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.pe = PositionalEncoding(in_dim)
        self.in_dim = in_dim
        self.num_heads = num_heads
        assert(self.in_dim % self.num_heads == 0)
        self.head_dim = self.in_dim // self.num_heads

        self.w_q = nn.Linear(in_dim, 1)
        self.w_k = nn.Linear(in_dim, 1)
        self.w_o = nn.Parameter(torch.zeros(in_dim))

        c = 1e-2

        init_layer(self.w_q, 0.0, 0.0)
        init_layer(self.w_k, c, c)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, feats, logits, deltas):
        feats = feats.detach()
        logits = logits.detach()
        # 1. dot product with weight matrices
        N, P, K = logits.shape
        
        N, P, C, H, W = feats.shape
        pe = self.pe(deltas)
        x = feats
        x = x + pe.unsqueeze(-1).unsqueeze(-1)
        
        
        x = x.permute(0, 3, 4, 1, 2).reshape(-1, P, C) # N, H, W, P, C
        
        q = self.w_q(x)
        k = self.w_k(x)
        v = x

        q = q.view(N, H, W, P, self.num_heads, -1).transpose(-2, -3) # N, head, P, K
        k = k.view(N, H, W, P, self.num_heads, -1).transpose(-2, -3) # N, head, P, K

        # batch x num_heads x query_len x key_len
        scores = torch.matmul(q, k.transpose(-1, -2)) # N H W head P P

        # apply attention dropout and compute context vectors.
        attention = self.softmax(scores) # N, H, W, head P, P
        s = attention.mean(1).mean(1).mean(1).mean(1)
        attention = self.dropout(attention)#.unsqueeze(1).unsqueeze(1) # N, 1, 1, head, P, P

        v = v.view(N, H, W, P, self.num_heads, -1).transpose(-2, -3) # N, H, W, head, P, C
        context = torch.matmul(attention, v) # N H W, head, P, C
        context = context.transpose(-2, -3).contiguous().view(
            N, H, W, P, -1)


        output = context
        output = self.w_o.expand_as(output) * output
        output = self.dropout(output)
        output = output.reshape(N, H, W, P, -1).permute(0, 3, 4, 1, 2) # N P C H W

        return output + feats, s
