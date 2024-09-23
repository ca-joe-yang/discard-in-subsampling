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
        # pe[:, 0, 0::2] = torch.sin(position * div_term)
        # pe[:, 0, 1::2] = torch.cos(position * div_term)
        # pe[:, 1, 0::2] = torch.sin(position * div_term)
        # pe[:, 1, 1::2] = torch.cos(position * div_term)
        self.pe = nn.Parameter(pe)
        # self.register_buffer('pe', pe)

    def forward(self, deltas):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        # self.pe.to(x.device)
        # print(self.pe.device, deltas.device)
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
    # elif w == 'kaiming':
    #     nn.init.kaiming_normal_(layer.weight)
    else:
        # nn.init.constant_(layer.weight, w)
        nn.init.xavier_normal_(layer.weight, gain=w)

class MultiHeadAttention(nn.Module):
    
    def __init__(self, in_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.pe = PositionalEncoding(in_dim)
        self.in_dim = in_dim
        self.num_heads = num_heads
        assert(self.in_dim % self.num_heads == 0)
        self.head_dim = self.in_dim // self.num_heads
        # self.layer_norm = nn.LayerNorm([in_dim, 7, 7], eps=1e-6)

        self.w_q = nn.Linear(in_dim, 1)
        self.w_k = nn.Linear(in_dim, 1)
        # self.w_v = nn.Linear(in_dim, 512)#nn.Linear(in_dim, 1)
        # self.w_v = nn.Parameter(torch.ones(in_dim))
        # self.w_o = nn.Linear(512, in_dim)
        self.w_o = nn.Parameter(torch.zeros(in_dim))

        c = 1e-2

        init_layer(self.w_q, 0.0, 0.0)
        init_layer(self.w_k, c, c)
        # init_layer(self.w_v, 'eye', 0.0)
        # init_layer(self.w_o, 0.0, 0.0)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, feats, logits, deltas):
        feats = feats.detach()
        logits = logits.detach()
        # 1. dot product with weight matrices
        N, P, K = logits.shape
        
        # orig_x = x
        # if len(orig_x.shape) == 3: # N P K
        #     x = x.unsqueeze(-1).unsqueeze(-1)
        N, P, C, H, W = feats.shape
        pe = self.pe(deltas)
        x = feats
        x = x + pe.unsqueeze(-1).unsqueeze(-1)
        
        # x = self.layer_norm(x.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        #[N, P, C, H, W]
        
        x = x.permute(0, 3, 4, 1, 2).reshape(-1, P, C) # N, H, W, P, C
        # y = y.permute(0, 3, 4, 1, 2).reshape(-1, P, C)
        # x = self.layer_norm(x)
        
        q = self.w_q(x)
        k = self.w_k(x)
        v = x#self.w_v(x)

        # reshape q, k, v for our computation to [batch_size, L, num_heads, ..]
        q = q.view(N, H, W, P, self.num_heads, -1).transpose(-2, -3) # N, head, P, K
        k = k.view(N, H, W, P, self.num_heads, -1).transpose(-2, -3) # N, head, P, K

        # batch x num_heads x query_len x key_len
        scores = torch.matmul(q, k.transpose(-1, -2)) # N H W head P P
        # if mask is not None:
        #     scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(1), float('-inf'))

        # apply attention dropout and compute context vectors.
        attention = self.softmax(scores) # N, H, W, head P, P
        s = attention.mean(1).mean(1).mean(1).mean(1)
        # print(s[0])
        attention = self.dropout(attention)#.unsqueeze(1).unsqueeze(1) # N, 1, 1, head, P, P

        v = v.view(N, H, W, P, self.num_heads, -1).transpose(-2, -3) # N, H, W, head, P, C
        context = torch.matmul(attention, v) # N H W, head, P, C
        context = context.transpose(-2, -3).contiguous().view(
            N, H, W, P, -1)

        # print(attention.shape, context.shape)

        output = context
        # output = self.w_o(output)
        output = self.w_o.expand_as(output) * output
        output = self.dropout(output)
        output = output.reshape(N, H, W, P, -1).permute(0, 3, 4, 1, 2) # N P C H W
        # print()
        # if len(orig_x.shape) == 3:
        #     output = output.squeeze(-1).squeeze(-1)

        return output + feats, s

    def forward1(self, feats, logits, deltas):
        logits = logits.detach()
        # 1. dot product with weight matrices
        N, P, K = logits.shape
        
        x = logits
        # pe = self.pe(deltas)
        # x = x + pe
        
        q = self.w_q(x)
        k = self.w_k(x)
        v = x#self.w_v(x)

        # reshape q, k, v for our computation to [batch_size, L, num_heads, ..]
        q = q.view(N, P, self.num_heads, -1).transpose(-2, -3) # N, head, P, K
        k = k.view(N, P, self.num_heads, -1).transpose(-2, -3) # N, head, P, K

        # batch x num_heads x query_len x key_len
        scores = torch.matmul(q, k.transpose(-1, -2)) # N head P P
        s = scores.reshape(N, self.num_heads, P, P).mean(1)
        s = torch.stack([ torch.diag(_) for _ in s], 0) # N P

        # apply attention dropout and compute context vectors.
        attention = self.softmax(scores)
        attention = self.dropout(attention) # N, head, P, P

        v = v.view(N, P, self.num_heads, -1).transpose(-2, -3) # N, head, P, C
        context = torch.matmul(attention, v) # N head, P, C
        print(context.shape)
        context = context.transpose(-2, -3).contiguous().view(
            N, P, -1)

        output = context
        output = self.w_o(output)
        output = self.dropout(output)

        return output + logits, s
        
# class GELU(nn.Module):
#     """
#     Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
#     """

#     def forward(self, x):
#         return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

# class PositionwiseFeedForward(nn.Module):
#     """
#     Position-wise Feed-forward layer
#     Projects to ff_size and then back down to input_size.
#     """

#     def __init__(self, input_size, ff_size, dropout=0.1):
#         """
#         Initializes position-wise feed-forward layer.
#         :param input_size: dimensionality of the input.
#         :param ff_size: dimensionality of intermediate representation
#         :param dropout:
#         """
#         super(PositionwiseFeedForward, self).__init__()
#         self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)
#         self.pwff_layer = nn.Sequential(
#             nn.Linear(input_size, ff_size),
#             GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(ff_size, input_size),
#             nn.Dropout(dropout),
#         )

#     def forward(self, x):
#         x_norm = self.layer_norm(x)
#         return self.pwff_layer(x_norm) + x

# class TransformerEncoderLayer(nn.Module):
#     """
#     One Transformer encoder layer has a Multi-head attention layer plus
#     a position-wise feed-forward layer.
#     """

#     def __init__(self,
#                  size: int = 0,
#                  ff_size: int = 0,
#                  num_heads: int = 0,
#                  dropout: float = 0.1):
#         """
#         A single Transformer layer.
#         :param size:
#         :param ff_size:
#         :param num_heads:
#         :param dropout:
#         """
#         super().__init__()

#         self.layer_norm = nn.LayerNorm(size, eps=1e-6)
#         self.src_src_att = MultiHeadedAttention(num_heads, size,
#                                                 dropout=dropout)
#         self.feed_forward = PositionwiseFeedForward(size, ff_size=ff_size)
#         self.dropout = nn.Dropout(dropout)
#         self.size = size

#     # pylint: disable=arguments-differ
#     def forward(self, x: Tensor, mask: Tensor) -> Tensor:
#         """
#         Forward pass for a single transformer encoder layer.
#         First applies layer norm, then self attention,
#         then dropout with residual connection (adding the input to the result),
#         and then a position-wise feed-forward layer.
#         :param x: layer input
#         :param mask: input mask
#         :return: output tensor
#         """
#         x_norm = self.layer_norm(x)
#         h = self.src_src_att(x_norm, x_norm, x_norm, mask)
#         h = self.dropout(h) + x
#         o = self.feed_forward(h)
#         return o
