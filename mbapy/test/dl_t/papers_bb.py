'''
Author: BHM-Bob 2262029386@qq.com
Date: 2023-04-19 16:30:02
LastEditors: BHM-Bob
LastEditTime: 2023-04-19 16:34:22
Description: 
'''

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(r'../../../')

import dl_torch.paper.bb as bb

x = torch.rand([16, 8, 64, 64], device = 'cuda')
net = bb.NonLocalBlock(8, 8, 8).to('cuda')
print(net(x).shape, "torch.Size([16, 8, 64, 64])")