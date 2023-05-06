'''
Author: BHM-Bob 2262029386@qq.com
Date: 2023-04-19 16:30:02
LastEditors: BHM-Bob
LastEditTime: 2023-05-06 16:51:57
Description: 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import dl_torch.paper.bb as bb

x = torch.rand([16, 8, 64, 64], device = 'cuda', dtype = torch.float16)
net = bb.NonLocalBlock(8, 8, 8, dtype = torch.float16).to('cuda')
print(net(x).shape, "torch.Size([16, 8, 64, 64])")

x = torch.rand([64, 256, 128], device = 'cuda', dtype = torch.float16)
net = bb.FlashMHA(embed_dim=128, num_heads=8, device='cuda', dtype=torch.float16)
print(net(x)[0].shape, "torch.Size([64, 256, 128])")

x = torch.rand([64, 256, 128], device = 'cuda')
net = bb.HydraAttention(128).to('cuda')
print(net(x, x, x).shape, "torch.Size([64, 256, 128])")
