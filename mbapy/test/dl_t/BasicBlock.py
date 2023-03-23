'''
Author: BHM-Bob 2262029386@qq.com
Date: 2022-11-04 12:33:19
LastEditors: BHM-Bob
LastEditTime: 2023-03-23 22:57:48
Description: 
'''
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(r'../../../')

import dl_torch.bb as bb
x = torch.arange(16, dtype = torch.float32, device = 'cuda').reshape([1, 1, 4, 4])
t = F.unfold(x, 3, 1, 1, 1)
t = (
    t.reshape(1, 1, 9, 16)
    .permute(0, 1, 3, 2)
    .reshape(1, 1 * 16, 9)
)
print(t)
net = bb.SCANN(4, 1, 1, 1, 1, 3,'linear', 0.3).to('cuda')
print(net(x).shape, "torch.Size([1, 1, 4, 4])")
