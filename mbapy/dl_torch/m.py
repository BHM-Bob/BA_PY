'''
Author: BHM-Bob 2262029386@qq.com
Date: 2023-03-23 21:50:21
LastEditors: BHM-Bob
LastEditTime: 2023-05-04 18:15:27
Description: Model
'''

import math
from typing import Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import GlobalSettings
from . import bb
from .bb import CnnCfg

class TransCfg:
    def __init__(self, hid_dim:int, pf_dim:int = None, n_heads:int = 8, n_layers:int = 3,
                 dropout:float = 0.3, q_len:int = -1, class_num:int = -1):
        self.hid_dim = hid_dim
        self.pf_dim = pf_dim if pf_dim is not None else 2*hid_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.q_len = q_len
        self.class_num = class_num
        self._str_ = f'hid_dim={self.hid_dim:d}, pf_dim={self.pf_dim:d}, n_heads={self.n_heads:d}, n_layers={self.n_layers:d}, dropout={self.dropout:f}, q_len={self.q_len:d}, class_num={self.class_num:d}'
    def __str__(self):
        return self._str_
    def toDict(self):
        d = {}
        for attr in vars(self):
            d[attr] = getattr(self,attr)
        return d
    def gen(self, layer, **kwargs):
        """generate a transformer like layer using cfg, unnecessary args will be the kwargs"""
        return layer(**(self.toDict()), **kwargs)

class LayerCfg(CnnCfg):
    def __init__(self, inc:int, outc:int, kernel_size:int, stride:int,
                 layer:str, sa_layer:str = None, trans_layer:str = None,
                 avg_size:int = -1, trans_cfg:TransCfg = None,
                 use_SA:bool = False, use_trans:bool = False):
        super().__init__(inc, outc, kernel_size, stride)
        self.layer = layer
        self.sa_layer = sa_layer
        self.trans_layer = trans_layer
        self.avg_size = avg_size
        self.trans_cfg = trans_cfg
        self.use_SA = use_SA
        self.use_trans = use_trans
        self._str_ += f', layer={layer:}, sa_layer={sa_layer:}, trans_layer={trans_layer:}, avg_size={avg_size:}, use_SA={use_SA:}, use_trans={use_trans:}, trans_cfg={trans_cfg:}'

def calcu_q_len(input_size:int, cfg:list[LayerCfg], dims:int = 1):
    """
    calcu q_len for Conv model
    strides: list of each layers' stride
    input_size : when dim is 2, it must be set as img 's size (w==h)
    """
    ret = input_size
    for s in cfg:
        while not ret % s.stride == 0:
            ret += 1
        ret /= s.stride
    return int(ret**dims)

class COneDLayer(nn.Module):
    """[b, c, l] => [b, c', l']"""
    def __init__(self, cfg:LayerCfg, device = 'cuda', **kwargs):
        super().__init__()
        self.cfg = cfg
        self.t_cfg = cfg.trans_cfg
        self.layer = str2net[cfg.layer](CnnCfg(cfg.inc, cfg.outc, cfg.kernel_size))
        self.avg = nn.AvgPool1d(cfg.avg_size, cfg.avg_size)
        if self.cfg.use_trans:
            self.trans = cfg.trans_cfg.gen(str2net[cfg.trans_layer], **kwargs)
    def forward(self, x):
        """[b, c, l] => [b, c', l']"""
        x = self.layer(x)
        x = self.avg(x)
        if self.cfg.use_trans:
            # x: [b, c', l'] => [b, l', c'] => [b, c', l']
            x = self.trans(x.permute(0, 2, 1)).permute(0, 2, 1)
        # [b, c', l']
        return x

class MAlayer(nn.Module):
    """[b, c, w, h]"""
    def __init__(self, cfg:LayerCfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        if self.cfg.use_SA:
            self.SA = str2net[cfg.sa_layer](CnnCfg(cfg.inc, cfg.outc, cfg.kernel_size))
            self.layer = nn.Sequential(str2net[cfg.layer](CnnCfg(cfg.outc, cfg.outc, cfg.kernel_size, stride=2)),
                                       str2net[cfg.layer](CnnCfg(cfg.outc, cfg.outc, cfg.kernel_size, stride=1)))
        else:
            self.layer = nn.Sequential(str2net[cfg.layer](CnnCfg(cfg.inc, cfg.outc, cfg.kernel_size, stride=2)),
                                       str2net[cfg.layer](CnnCfg(cfg.outc, cfg.outc, cfg.kernel_size, stride=1)))
    def forward(self, x):
        if self.cfg.use_SA:
            x = self.SA(x)
        return self.layer(x)

class MAvlayer(MAlayer):
    def __init__(self, cfg:LayerCfg, **kwargs):
        super().__init__(cfg, device = 'cuda', **kwargs)
        if self.cfg.use_SA:
            self.layer = nn.Sequential(nn.AvgPool2d((cfg.avg_size, cfg.avg_size), cfg.avg_size),
                                       str2net[cfg.layer](CnnCfg(cfg.inc, cfg.inc, 1, 1, 0)))
            self.SA = str2net[cfg.sa_layer](CnnCfg(cfg.inc, cfg.outc, cfg.kernel_size))
        else:
            self.layer = nn.Sequential(nn.AvgPool2d((cfg.avg_size, cfg.avg_size), cfg.avg_size),
                                       str2net[cfg.layer](CnnCfg(cfg.inc, cfg.outc, 1, 1, 0)))
    def forward(self, x):
        x = self.layer(x)
        if self.cfg.use_SA:
            x = self.SA(x)
        return x
    
class SCANlayer(nn.Module):
    def __init__(self, cfg:LayerCfg,
                 layer = bb.SCANN, SA_layer = bb.SABlockR,
                 device = 'cuda', **kwargs):
        super().__init__()
        self.cfg = cfg
        if self.use_SA:
            self.layer = SA_layer(CnnCfg(cfg.inc, cfg.outc, cfg.kernel_size))
        else:
            self.layer = nn.Conv2d(cfg.inc, cfg.outc, 1, 1, 0)
        self.scann = layer(self.pic_size, cfg.inc, padding="half", outway="avg")
        self.shoutCut = nn.AvgPool2d((2, 2), stride=2, padding=0)
    def forward(self, x):
        t = self.shoutCut(x)
        x = self.scann(x)
        x = self.layer(t + x)
        return x

str2net = {
    'EncoderLayer':bb.EncoderLayer,
    'OutEncoderLayer':bb.OutEncoderLayer,
    'OutEncoderLayerAvg':bb.OutEncoderLayerAvg,
    'Trans':bb.Trans,
    'TransPE':bb.TransPE,
    'TransAvg':bb.TransAvg,
    
    'SeparableConv2d':bb.SeparableConv2d,
    'ResBlock':bb.ResBlock,
    'ResBlockR':bb.ResBlockR,
    'SABlock':bb.SABlock,
    'SABlockR':bb.SABlockR,
    'SABlock1D':bb.SABlock1D,
    'SABlock1DR':bb.SABlock1DR,
    
    'COneDLayer':COneDLayer,
    'MAlayer':MAlayer,
    'MAvlayer':MAvlayer,
    'SCANlayer':SCANlayer,
}

class MATTPBase(nn.Module):#MA TT with permute
    """ x: [b, c, w, h] => [b, c', w', h'] or [b, l, c']"""
    def __init__(self, args: GlobalSettings, cfg:list[LayerCfg], layer:MAlayer,
                 tail_trans_cfg:TransCfg = None, **kwargs):
        super().__init__()
        self.args = args
        self.cfg = cfg
        self.tail_trans_cfg = tail_trans_cfg
        args.mp.mprint('cfg:list[LayerCfg] =\n',
                       "\n".join([f'LAYER {i:d}: '+str(c) for i, c in enumerate(cfg)]))
        self.main_layers = nn.ModuleList([ layer(c) for c in self.cfg ])
        if self.tail_trans_cfg is not None:
            # TODO : add out trans
            self.tail_trans = nn.ModuleList(
                    [
                        bb.EncoderLayer(**(self.tail_trans_cfg.toDict()))
                        for _ in range(self.tail_trans_cfg.n_layers)
                    ]
                )
    def forward(self, x):
        """ x: [b, c, w, h] => [b, c', w', h'] or [b, l, c']"""
        batch_size = x.shape[0]
        # x: [b, c, w, h] => [b, c', w', h']
        for layer in self.main_layers:
            x = layer(x)
        if self.tail_trans_cfg is not None:
            # x: [b, c', w', h'] => [b, c', l] => [b, l, c']
            x = x.reshape(batch_size, self.cfg[-1].outc, -1).permute(0, 2, 1)
            for layer in self.tail_trans:
                x = layer(x)
        return x

class COneD(MATTPBase):#MA TT with permute
    def __init__(self, args: GlobalSettings, cfg:list[LayerCfg], layer:COneDLayer,
                 tail_trans_cfg:TransCfg = None, **kwargs):
        super(COneD, self).__init__(args, cfg, layer, tail_trans_cfg, **kwargs)
    
class MATTP(MATTPBase):#MA TT with permute
    def __init__(self, args: GlobalSettings, cfg:list[LayerCfg], layer:MAlayer,
                 tail_trans_cfg:TransCfg = None, **kwargs):
        super().__init__(args, cfg, layer, tail_trans_cfg, **kwargs)
         
class MAvTTP(MATTPBase):#MA TT with permute
    def __init__(self, args: GlobalSettings, cfg:list[LayerCfg], layer:MAvlayer,
                 tail_trans_cfg:TransCfg = None, **kwargs):
        super().__init__(args, cfg, layer, tail_trans_cfg, **kwargs)
         
class MATTPE(MATTPBase):#MA TT with permute
    def __init__(self, args: GlobalSettings):
        self.stemconvnum = 32
        self.convnum = [
            [args.channels       , 1 * self.stemconvnum, False],  # [b,32,40,h]
            [1 * self.stemconvnum, 2 * self.stemconvnum,  True],  # [b,64,20,h]
            [2 * self.stemconvnum,             args.dim,  True],
        ]  
        super(MATTPE, self).__init__(args, self.convnum, MAvlayer)
        self.pos_embedding = bb.PositionalEncoding(args.dim, self.q_len)
    def forward(self, x):#x: [b, c, L]
        batchSize = x.shape[0]
        # x: [b, c, L] => [b, c, w, h]
        x = x.reshape(batchSize, self.channels, self.picsize, self.picsize)
        # x: [b, c, w, h] => [b, c', w', h']
        for maLayer in self.MAlayers:
            x = maLayer(x)
        # x: [b, c', w', h'] => [b, c', l]
        x = x.reshape(batchSize,self.convnum[-1][1],self.q_len)
        # x: [b, c', l] => [b, l, c']
        x = x.permute(0,2,1)
        # positional_encoding
        x = self.pos_embedding(x)
        # # x: [b, l, c'] => [b, l, c']
        # for layer in self.tailTrans:
        #     x = layer(x)
        return x
    
class SCANNTTP(MATTPBase):#MA TT with permute
    def __init__(self, args: GlobalSettings):
        self.stemconvnum = 32
        self.convnum = [
            [args.channels       , 1 * self.stemconvnum, 40, True],  # [b,32,20,h]
            [1 * self.stemconvnum, 2 * self.stemconvnum, 20, True],  # [b,64,10,h]
            [2 * self.stemconvnum,             args.dim, 10, True],
        ]
        super(SCANNTTP, self).__init__(args, self.convnum, SCANlayer)
        
class MATTP_ViT(nn.Module):  # MA TT with permute
    def __init__(self, classnum, device, seqLen=32, channels=3):
        super(MATTP_ViT, self).__init__()
        self.picsize = int(math.sqrt(seqLen))
        self.channels = channels
        self.classnum = classnum
        self.stemconvnum = 16
        self.pf_dim = 256
        self.n_heads = 4
        self.n_layers = 2
        self.convnum = [
            [channels, 1 * self.stemconvnum, True],  # [b,32,41,h]
            [1 * self.stemconvnum, 1 * self.stemconvnum, True],  # [b,64,21,h]
            [1 * self.stemconvnum, 8 * self.stemconvnum, True],
        ]  # [b,128,6q,h]
        self.q_len = 25

        # Uinfo: [b,2,128]
        self.Uinfo = nn.Parameter(torch.randn(1, 2, self.convnum[-1][1])).to(device)

        self.MAlayers = nn.ModuleList(
            [MAlayer(convnum[0], convnum[1], convnum[2]) for convnum in self.convnum]
        )

        if self.n_layers > 1:
            self.translayers = nn.ModuleList(
                [
                    bb.EncoderLayer(
                        self.convnum[-1][1], self.n_heads, self.pf_dim, 0.2, device
                    )
                    for _ in range(self.n_layers)
                ]
            )
        else:
            self.translayers = None

        self.fc = nn.Linear(self.convnum[-1][1], classnum)

    def forward(self, x):  # x: [b,3,128,128]
        batchSize = x.shape[0]
        # x: [b,3,128,128]
        x = x.reshape(batchSize, self.channels, self.picsize, self.picsize)
        # x: [b,128,5,5]
        for maLayer in self.MAlayers:
            x = maLayer(x)
        # x: [b,128,5,5] => [b,128,25] => [b,25,128]
        x = x.reshape(batchSize, self.convnum[-1][1], self.q_len).permute(0, 2, 1)
        # x: [b,2+25,128]
        x = torch.cat([self.Uinfo.repeat(batchSize, 1, 1), x], dim=1)

        if not self.translayers is None:
            for layer in self.translayers:
                x = layer(x)

        # x: [b,2,128] => [b,2,class_num] => [b,class_num,2] => [b*class_num,2] => [b, cn*2]
        x = (
            self.fc(x[:, 0:2, :])
            .permute(0, 2, 1)
            .reshape(batchSize * self.classnum, 2)
            .reshape(batchSize, -1)
        )
        return x