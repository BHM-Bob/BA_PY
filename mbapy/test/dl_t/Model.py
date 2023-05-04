'''
Author: BHM-Bob 2262029386@qq.com
Date: 2022-11-04 12:33:19
LastEditors: BHM-Bob
LastEditTime: 2023-05-04 21:27:46
Description: Test for Model
'''
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(r'../../../')
import dl_torch as dt

import dl_torch.bb as bb
import dl_torch.m as m
from dl_torch.utils import Mprint, GlobalSettings

x = torch.rand([32, 32, 1024], device = 'cuda')

net = m.COneDLayer(m.LayerCfg(32, 64, 3, 2, 'SABlock1D', avg_size=4)).to('cuda')
print(net(x).shape, "torch.Size([32, 64, 256])")
net = m.COneDLayer(m.LayerCfg(32, 64, 3, 2, 'SABlock1DR', avg_size=4)).to('cuda')
print(net(x).shape, "torch.Size([32, 64, 256])")


x = torch.rand([32, 32, 64, 64], device = 'cuda')

net = m.MAlayer(m.LayerCfg(32, 64, 3, 2, 'ResBlock', 'SABlock')).to('cuda')
print(net(x).shape, "torch.Size([32, 64, 32, 32])")
net = m.MAlayer(m.LayerCfg(32, 64, 3, 2, 'ResBlockR', 'SABlockR')).to('cuda')
print(net(x).shape, "torch.Size([32, 64, 32, 32])")

net = m.MAvlayer(m.LayerCfg(32, 64, 2, 2, 'ResBlock', 'SABlock', avg_size=2)).to('cuda')
print(net(x).shape, "torch.Size([32, 64, 32, 32])")
net = m.MAvlayer(m.LayerCfg(32, 64, 2, 2, 'ResBlockR', 'SABlockR', avg_size=2)).to('cuda')
print(net(x).shape, "torch.Size([32, 64, 32, 32])")

# net = m.SCANlayer(m.LayerCfg(32, 64, 3, 2),
#                   layer = bb.ResBlock, device='cuda').to('cuda')
# print(net(x).shape, "torch.Size([32, 64, 32, 32])")

args = GlobalSettings(Mprint(), '')

x = torch.rand([32, 3, 128, 128], device = 'cuda')
cfg = [
    m.LayerCfg( 3,  8, 3, 2, 'ResBlockR', 'SABlockR'),
    m.LayerCfg( 8, 16, 3, 2, 'ResBlockR', 'SABlockR'),
    m.LayerCfg(16, 32, 3, 2, 'ResBlockR', 'SABlockR'),
    m.LayerCfg(32, 64, 3, 2, 'ResBlockR', 'SABlockR'),
    ]
net = m.MATTPBase(args, cfg, m.MAlayer).to('cuda')
print(net(x).shape, "torch.Size([32, 64, 8, 8])")

x = torch.rand([32, 8, 1024], device = 'cuda')
cfg = [
    m.LayerCfg( 8,  32, 7, 1, 'SABlock1D', avg_size=4, use_trans=False),
    m.LayerCfg(32,  64, 5, 1, 'SABlock1D', avg_size=2, use_trans=False),
    m.LayerCfg(64,  64, 3, 1, 'SABlock1D', avg_size=2, use_trans=True,
               trans_layer='EncoderLayer', trans_cfg=m.TransCfg(64)),
    m.LayerCfg(64, 128, 3, 1, 'SABlock1D', avg_size=2, use_trans=True,
               trans_layer='EncoderLayer', trans_cfg=m.TransCfg(128)),
    ]
net = m.COneD(args, cfg, m.COneDLayer).to('cuda')
print(net(x).shape, "torch.Size([32, 128, 32])")

x = torch.rand([32, 3, 128, 128], device = 'cuda')
cfg = [
    m.LayerCfg( 3,  32, 3, 2, 'ResBlockR', 'SABlockR', use_trans=False),
    m.LayerCfg(32,  64, 3, 2, 'ResBlockR', 'SABlockR', use_trans=False),
    m.LayerCfg(64, 128, 3, 2, 'ResBlockR', 'SABlockR', use_trans=True,
               trans_layer='EncoderLayer', trans_cfg=m.TransCfg(128)),
    ]
net = m.MATTP(args, cfg, m.MAlayer, m.TransCfg(128, n_layers=2)).to('cuda')
print(net(x).shape, "torch.Size([32, 256, 128])")

x = torch.rand([32, 3, 128, 128], device = 'cuda')
cfg = [
    m.LayerCfg( 3,  32, 7, 1, 'ResBlockR', 'SABlockR', avg_size=2, use_SA=False),
    m.LayerCfg(32,  64, 7, 1, 'ResBlockR', 'SABlockR', avg_size=2, use_SA=True),
    m.LayerCfg(64, 128, 5, 1, 'ResBlockR', 'SABlockR', avg_size=2, use_SA=True,
               trans_layer='EncoderLayer', trans_cfg=m.TransCfg(128)),
    ]
net = m.MAvTTP(args, cfg, m.MAvlayer, m.TransCfg(128, n_layers=2)).to('cuda')
print(net(x).shape, "torch.Size([32, 256, 128])")

x = torch.rand([32, 3, 128, 128], device = 'cuda')
cfg = [
    m.LayerCfg( 3,  32, 7, 1, 'ResBlockR', 'SABlockR', avg_size=2, use_SA=False),
    m.LayerCfg(32,  64, 7, 1, 'ResBlockR', 'SABlockR', avg_size=2, use_SA=True),
    m.LayerCfg(64, 128, 5, 1, 'ResBlockR', 'SABlockR', avg_size=2, use_SA=True,
               trans_layer='EncoderLayer', trans_cfg=m.TransCfg(128)),
    ]
net = m.MATTPE(args, cfg, m.MAvlayer, m.TransCfg(128, n_layers=2)).to('cuda')
print(net(x).shape, "torch.Size([32, 256, 128])")

x = torch.rand([32, 3, 128, 128], device = 'cuda')
cfg = [
    m.LayerCfg( 3,  32, 7, 1, 'ResBlockR', 'SABlockR', avg_size=2, use_SA=False),
    m.LayerCfg(32,  64, 7, 1, 'ResBlockR', 'SABlockR', avg_size=2, use_SA=True),
    m.LayerCfg(64, 128, 5, 1, 'ResBlockR', 'SABlockR', avg_size=2, use_SA=True,
               trans_layer='EncoderLayer', trans_cfg=m.TransCfg(128)),
    ]
net = m.SCANNTTP(args, cfg, m.SCANlayer, m.TransCfg(128, n_layers=2)).to('cuda')
print(net(x).shape, "torch.Size([32, 256, 128])")

