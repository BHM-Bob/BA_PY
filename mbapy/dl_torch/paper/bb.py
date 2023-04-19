'''
Author: BHM-Bob 2262029386@qq.com
Date: 2023-03-23 21:50:21
LastEditors: BHM-Bob
LastEditTime: 2023-04-19 16:28:07
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
        self.q = nn.Conv2d(self.inc, self.hid_dim, kernel_size=1)
        self.k = nn.Conv2d(self.inc, self.hid_dim, kernel_size=1)
        self.v = nn.Conv2d(self.inc, self.hid_dim, kernel_size=1)
        self.o = nn.Conv2d(self.hid_dim, self.outc, kernel_size=1)
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
    
