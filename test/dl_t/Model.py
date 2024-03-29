'''
Author: BHM-Bob 2262029386@qq.com
Date: 2022-11-04 12:33:19
LastEditors: BHM-Bob
LastEditTime: 2023-05-10 18:13:08
Description: Test for Model
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import mbapy.dl_torch as dt

import mbapy.dl_torch.bb as bb
import mbapy.dl_torch.m as m
from mbapy.dl_torch.utils import Mprint, GlobalSettings

criterion = nn.CrossEntropyLoss().cuda('cuda')
label = torch.rand(size = (32, 128)).to('cuda')
        
x = torch.rand([32, 32, 1024], device = 'cuda')

net = m.COneDLayer(m.LayerCfg(32, 64, 3, 2, 'SABlock1D', avg_size=4)).to('cuda')
print(net(x).shape, "torch.Size([32, 64, 256])")
net = m.COneDLayer(m.LayerCfg(32, 64, 3, 2, 'SABlock1DR', avg_size=4)).to('cuda')
print(net(x).shape, "torch.Size([32, 64, 256])")
net = m.COneDLayer(m.LayerCfg(32, 64, 3, 2, 'SABlock1DR', avg_size=4,
                              use_trans=True, trans_layer='EncoderLayer',
                              trans_cfg=m.TransCfg(64, use_FastMHA = True))).to('cuda')
print(net(x).shape, "torch.Size([32, 64, 256])\n")


x = torch.rand([32, 32, 64, 64], device = 'cuda')

net = m.MAlayer(m.LayerCfg(32, 64, 3, 2, 'ResBlock', 'SABlock')).to('cuda')
print(net(x).shape, "torch.Size([32, 64, 32, 32])\n")
net = m.MAlayer(m.LayerCfg(32, 64, 3, 2, 'ResBlockR', 'SABlockR')).to('cuda')
print(net(x).shape, "torch.Size([32, 64, 32, 32])\n")

net = m.MAvlayer(m.LayerCfg(32, 64, 2, 2, 'ResBlock', 'SABlock', avg_size=2)).to('cuda')
print(net(x).shape, "torch.Size([32, 64, 32, 32])\n")
net = m.MAvlayer(m.LayerCfg(32, 64, 2, 2, 'ResBlockR', 'SABlockR', avg_size=2)).to('cuda')
print(net(x).shape, "torch.Size([32, 64, 32, 32])\n")

net = m.SCANlayer(m.LayerCfg(32, 64, 3, 2, ''),
                  layer = bb.ResBlock, device='cuda').to('cuda')
print(net(x).shape, "torch.Size([32, 64, 32, 32])\n")

args = GlobalSettings(Mprint(), '')

x = torch.rand([32, 3, 128, 128], device = 'cuda')
cfg = [
    m.LayerCfg( 3,  8, 3, 2, 'ResBlockR', 'SABlockR'),
    m.LayerCfg( 8, 16, 3, 2, 'ResBlockR', 'SABlockR'),
    m.LayerCfg(16, 32, 3, 2, 'ResBlockR', 'SABlockR'),
    m.LayerCfg(32, 64, 3, 2, 'ResBlockR', 'SABlockR',
               use_trans = True, trans_layer='EncoderLayer',
               trans_cfg=m.TransCfg(64, n_layers=2, q_len = 64, class_num=128,
                                    out_layer='OutEncoderLayer',
                                    ))
    ]
net = m.MATTPBase(args, cfg, m.MAlayer).to('cuda')
logits = net(x)
# criterion(logits, label).backward()
print('MATTPBase', logits.shape, "torch.Size([32, 128])\n")


x = torch.rand([32, 8, 1024], device = 'cuda')
cfg = [
    m.LayerCfg( 8,  32, 7, 1, 'SABlock1D', avg_size=4, use_trans=False),
    m.LayerCfg(32,  64, 5, 1, 'SABlock1D', avg_size=2, use_trans=False),
    m.LayerCfg(64,  64, 3, 1, 'SABlock1D', avg_size=2, use_trans=True,
               trans_layer='EncoderLayer', trans_cfg=m.TransCfg(64)),
    m.LayerCfg(64, 128, 3, 1, 'SABlock1D', avg_size=2, use_trans=True,
               trans_layer='EncoderLayer',
               trans_cfg=m.TransCfg(128, q_len = 32, class_num=128,
                                    out_layer='OutEncoderLayer')),
    ]
net = m.COneD(args, cfg, m.COneDLayer).to('cuda')
logits = net(x)
criterion(logits, label).backward()
print('COneD', logits.shape, "torch.Size([32, 128])\n")

x = torch.rand([32, 3, 128, 128], device = 'cuda')
cfg = [
    m.LayerCfg( 3,  16, 3, 2, 'ResBlockR', 'SABlockR', use_trans=False),
    m.LayerCfg(16,  32, 3, 2, 'ResBlockR', 'SABlockR', use_trans=False),
    m.LayerCfg(32,  64, 3, 2, 'ResBlockR', 'SABlockR', use_trans=False),
    m.LayerCfg(64, 128, 3, 2, 'ResBlockR', 'SABlockR',
               use_trans = True, trans_layer='EncoderLayer',
               trans_cfg=m.TransCfg(128, n_layers=2, q_len = 64, class_num=128,
                                    out_layer='OutEncoderLayer')),
    ]
net = m.MATTP(args, cfg, m.MAlayer).to('cuda')
logits = net(x)
criterion(logits, label).backward()
print('MATTP', net(x).shape, "torch.Size([32, 256, 128])\n")

x = torch.rand([32, 3, 128, 128], device = 'cuda')
cfg = [
    m.LayerCfg( 3,  32, 7, 1, 'ResBlockR', 'SABlockR', avg_size=2, use_trans=False, use_SA=False),
    m.LayerCfg(32,  64, 7, 1, 'ResBlockR', 'SABlockR', avg_size=2, use_trans=False, use_SA=True),
    m.LayerCfg(64, 128, 5, 1, 'ResBlockR', 'SABlockR', avg_size=2, use_trans=True, use_SA=True,
               trans_layer='EncoderLayer',
               trans_cfg=m.TransCfg(128, q_len = 256, class_num=128, out_layer='OutEncoderLayer')),
    ]
net = m.MAvTTP(args, cfg, m.MAvlayer).to('cuda')
print('MAvTTP', net(x).shape, "torch.Size([32, 128])\n")

x = torch.rand([32, 3, 128, 128], device = 'cuda')
cfg = [
    m.LayerCfg( 3,  32, 7, 1, 'ResBlockR', 'SABlockR', avg_size=2, use_SA=False),
    m.LayerCfg(32,  64, 7, 1, 'ResBlockR', 'SABlockR', avg_size=2, use_SA=True),
    m.LayerCfg(64, 128, 5, 1, 'ResBlockR', 'SABlockR', avg_size=2, use_SA=True,
               trans_layer='EncoderLayer', trans_cfg=m.TransCfg(128)),
    ]
net = m.MATTPE(args, cfg, m.MAvlayer,
               m.TransCfg(128, n_layers=2, q_len = 256, class_num=128, out_layer='OutEncoderLayer')).to('cuda')
print('MATTPE', net(x).shape, "torch.Size([32, 128])\n")

x = torch.rand([32, 3, 128, 128], device = 'cuda')
cfg = [
    m.LayerCfg( 3,  32, 7, -1, 'ResBlockR', 'SABlockR', avg_size=2, use_SA=False),
    m.LayerCfg(32,  64, 7, -1, 'ResBlockR', 'SABlockR', avg_size=2, use_SA=True),
    m.LayerCfg(64, 128, 5, -1, 'ResBlockR', 'SABlockR', avg_size=2, use_SA=True,
               trans_layer='EncoderLayer', trans_cfg=m.TransCfg(128)),
    ]
net = m.SCANNTTP(args, cfg, m.SCANlayer, m.TransCfg(128, n_layers=2)).to('cuda')
print('SCANNTTP', net(x).shape, "torch.Size([32, 256, 128])\n")

x = torch.rand([32, 3, 128, 128], device = 'cuda')
cfg = [
    m.LayerCfg( 3,  32, 3, 2, 'ResBlockR', 'SABlockR', avg_size=2, use_SA=False),
    m.LayerCfg(32,  64, 3, 2, 'ResBlockR', 'SABlockR', avg_size=2, use_SA=True),
    m.LayerCfg(64, 128, 3, 2, 'ResBlockR', 'SABlockR', avg_size=2, use_SA=True,
               trans_layer='EncoderLayer', trans_cfg=m.TransCfg(128)),
    ]
net = m.MATTP_ViT(args, cfg, m.MAlayer, m.TransCfg(128, n_layers=2)).to('cuda')
print('MATTP_ViT', net(x).shape, "torch.Size([32, 258, 128])\n")

