'''
Author: BHM-Bob 2262029386@qq.com
Date: 2023-03-23 21:50:21
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2024-07-17 09:14:33
Description: some Basic Blocks implements for some paper
'''

import math
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class NonLocalBlock(nn.Module):
    """Non-local Neural Networks (CVPR 2018)\n
    arXiv:1711.07971v3 [cs.CV] 13 Apr 2018\n
    Embedded Gaussian\n
    """
    def __init__(self, inc, hid_dim, outc, **kwargs):
        super().__init__()
        self.inc = inc
        self.hid_dim = hid_dim
        self.outc = outc
        self.q = nn.Conv2d(self.inc, self.hid_dim, kernel_size=1, **kwargs)
        self.k = nn.Conv2d(self.inc, self.hid_dim, kernel_size=1, **kwargs)
        self.v = nn.Conv2d(self.inc, self.hid_dim, kernel_size=1, **kwargs)
        self.o = nn.Conv2d(self.hid_dim, self.outc, kernel_size=1, **kwargs)
    def forward(self, x:torch.Tensor):
        # x:[b, c, h, w]
        shape = list(x.shape)
        # Q => [batch size, query len, hid dim]
        # K => [batch size, hid dim, query len]
        # V => [batch size, query len, hid dim]
        Q = self.q(x).reshape(shape[0], self.hid_dim, -1).permute(0, 2, 1)
        K = self.k(x).reshape(shape[0], self.hid_dim, -1)
        V = self.v(x).reshape(shape[0], self.hid_dim, -1).permute(0, 2, 1)
        # attention = [batch size, query len, query len]
        attention = Q.matmul(K).softmax(dim=-1)
        shape[1] = self.outc
        # x = [batch size, query len, hid dim] => [batch size, hid dim, query len] => [b, hid dim, h, w]
        return self.o(attention.matmul(V).permute(0, 2, 1).reshape(*shape))
    
try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None
"""
FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré
Paper: https://arxiv.org/abs/2205.14135
"""

class HydraAttention(nn.Module):
    """Hydra Attention:Efficient Attention with Many Heads
    arXiv:2209.07484v1 [cs.CV] 15 Sep 2022
    cosine similarity kernel
    modified from https://github.com/robflynnyh/hydra-linear-attention
    """
    def __init__(self, inc, output_layer='linear', dropout=0.3, **kwargs):
        super(HydraAttention, self).__init__()
        self.inc = inc
        self.out = nn.Linear(self.inc, self.inc) if output_layer == 'linear' else nn.Identity()
        self.dropout = nn.Dropout(dropout)
    def forward(self, q, k, v):
        '''x:[b, l, c]'''
        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
        kv = k * v
        kv = self.dropout(kv.transpose(-1, -2)).transpose(-1, -2) # dropout in seq dimension 
        out = kv.sum(dim=-2, keepdim=True) * q
        return self.out(out)