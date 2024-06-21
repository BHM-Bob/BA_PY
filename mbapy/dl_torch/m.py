'''
Author: BHM-Bob 2262029386@qq.com
Date: 2023-03-23 21:50:21
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2024-06-21 15:10:30
Description: Model, most of models outputs [b, c', w', h'] or [b, l', c'] or [b, D]\n
you can add tail_trans as normal transformer or out_transformer in LayerCfg of model.__init__()
'''

import math
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import autoparse
from . import bb
from .bb import CnnCfg
from .utils import GlobalSettings

# str2net合法性前置声明
str2net = {}

class TransCfg:
    @autoparse
    def __init__(self, hid_dim:int, pf_dim:Optional[int] = None, n_heads:int = 8,
                 n_layers:int = 3, dropout:float = 0.3,
                 trans_layer:str = 'EncoderLayer', out_layer:Optional[str] = None,
                 q_len:int = -1, class_num:int = -1,
                 **kwargs):
        self.pf_dim: int = pf_dim if pf_dim is not None else 2*hid_dim
        self.kwargs: Dict[str, Union[int, str, bool]] = kwargs if kwargs is not None else {}
        self._str_:str = ','.join([attr+'='+str(getattr(self, attr)) for attr in vars(self)])
    def __str__(self):
        return self._str_
    def toDict(self):
        d: Dict[str, Union[int, str, bool]] = {}
        for attr in vars(self):
            if attr not in ['_str_', 'kwargs']:
                d[attr] = getattr(self, attr)
        return d
    def gen(self, layer:str = None, **kwargs):
        """
        generate a transformer like layer using cfg, unnecessary args will be the kwargs\n
        support out_layer
        """
        kwargs.update(self.kwargs)
        kwargs.update(self.toDict())
        if layer is None:
            layer = str2net[kwargs['trans_layer']]
        if kwargs['out_layer'] is not None:
            return nn.Sequential(*([layer(**kwargs) for _ in range(self.n_layers - 1)]+\
                [str2net[kwargs['out_layer']](**kwargs)]))
        else:
            return nn.Sequential(*([layer(**kwargs) for _ in range(self.n_layers)]))
    
class LayerCfg:
    @autoparse
    def __init__(self, inc:int, outc:int, kernel_size:int, stride:int,
                 layer:str, sa_layer:Optional[str] = None, trans_layer:Optional[str] = None,
                 avg_size:int = -1, trans_cfg:Optional[TransCfg] = None,
                 use_SA:bool = False, use_trans:bool = False):
        if isinstance(self.sa_layer, str) and use_SA == True:
            self.use_SA = True            
        self._str_:str = ','.join([attr+'='+str(getattr(self, attr)) for attr in vars(self)])
    def __str__(self):
        return self._str_
    
def calcu_q_len(input_size:int, cfg:List[LayerCfg], dims:int = 1):
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
        """[b, c, l] => [b, c', l']"""
        super().__init__()
        self.cfg = cfg
        self.t_cfg = cfg.trans_cfg
        self.layer = str2net[cfg.layer](CnnCfg(cfg.inc, cfg.outc, cfg.kernel_size))
        if cfg.avg_size > 1:
            self.avg = nn.AvgPool1d(cfg.avg_size, cfg.avg_size)
        else:
            self.avg = nn.Identity()
        if self.cfg.use_trans:
            self.trans = self.t_cfg.gen(str2net[cfg.trans_layer], **kwargs)
    def forward(self, x):
        """[b, c, l] => [b, c', l']"""
        x = self.layer(x)
        x = self.avg(x)
        if self.cfg.use_trans:
            # x: [b, c', l'] => [b, l', c'] => [b, l', c'] or [b, D]
            x = self.trans(x.permute(0, 2, 1))
            if self.cfg.trans_cfg.out_layer is None:
                # x: [b, l', c'] => [b, c', l']
                x = x.permute(0, 2, 1)
        # [b, l', c'] or [b, D]
        return x

class MAlayer(nn.Module):
    """[b, c, w, h] or [b, D]"""
    def __init__(self, cfg:LayerCfg, **kwargs):
        """[b, c, w, h] or [b, D]"""
        super().__init__()
        self.cfg = cfg
        if self.cfg.use_SA:
            self.SA = str2net[cfg.sa_layer](CnnCfg(cfg.inc, cfg.outc, cfg.kernel_size))
            self.layer = nn.Sequential(str2net[cfg.layer](CnnCfg(cfg.outc, cfg.outc, cfg.kernel_size, stride=2)),
                                       str2net[cfg.layer](CnnCfg(cfg.outc, cfg.outc, cfg.kernel_size, stride=1)))
        else:
            self.layer = nn.Sequential(str2net[cfg.layer](CnnCfg(cfg.inc, cfg.outc, cfg.kernel_size, stride=2)),
                                       str2net[cfg.layer](CnnCfg(cfg.outc, cfg.outc, cfg.kernel_size, stride=1)))
        if self.cfg.use_trans:
            self.trans = cfg.trans_cfg.gen(str2net[cfg.trans_layer], **kwargs)
    def forward(self, x):
        if self.cfg.use_SA:
            x = self.SA(x)
        x = self.layer(x)
        if self.cfg.use_trans:
            batch_size, c, w, h = x.shape
            # x: [b, c', w', h'] => [b, c', l'] => [b, l', c']
            x = x.reshape(batch_size, c, -1).permute(0, 2, 1)
            # x: [b, l', c'] => [b, l', c'] or [b, D]
            x = self.trans(x)
            if self.cfg.trans_cfg.out_layer is None:
                # x: [b, l', c'] => [b, c', l'] => [b, c', w', h']
                x = x.permute(0, 2, 1).reshape(batch_size, c, w, h)
        return x

class MAvlayer(MAlayer):
    """[b, c, w, h] or [b, D]"""
    def __init__(self, cfg:LayerCfg, **kwargs):
        """[b, c, w, h] or [b, D]"""
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
        if self.cfg.use_trans:
            # x: [b, c', w', h'] => [b, c', l'] => [b, l', c'] => [b, c', l']
            batch_size, c, w, h = x.shape
            x = x.reshape(batch_size, c, -1)
            if self.cfg.use_trans:
                # x: [b, c', l'] => [b, l', c'] => [b, l', c'] or [b, D]
                x = self.trans(x.permute(0, 2, 1))
                if self.cfg.trans_cfg.out_layer is None:
                    # x: [b, l', c'] => [b, c', l']
                    x = x.permute(0, 2, 1)
                else:# x: [b, D], has self.cfg.trans_cfg.out_layer
                    return x
            x = x.reshape(batch_size, c, w, h)
        return x
    
class SCANlayer(nn.Module):
    def __init__(self, cfg:LayerCfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        if self.cfg.use_SA:
            self.layer = str2net[cfg.sa_layer](CnnCfg(cfg.inc, cfg.outc, cfg.kernel_size))
        else:
            self.layer = nn.Conv2d(cfg.inc, cfg.outc, 1, 1, 0)
        self.scann = bb.SCANN(cfg.inc, padding=1, outway="avg")
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
    """ x: [b, c, w, h] => [b, c', w', h'] or [b, c', l]"""
    def __init__(self, args: GlobalSettings, cfg:List[LayerCfg], layer:MAlayer,
                 tail_trans_cfg:TransCfg = None, **kwargs):
        """ x: [b, c, w, h] => [b, c', w', h'] or [b, c', l] or [b, D]"""
        super().__init__()
        self.args = args
        self.cfg = cfg
        self.tail_trans_cfg = tail_trans_cfg
        args.mp.mprint('cfg:List[LayerCfg] =\n',
                       "\n".join([f'LAYER {i:d}: '+str(c) for i, c in enumerate(cfg)]))
        self.main_layers = nn.ModuleList([ layer(c) for c in self.cfg ])
        if self.tail_trans_cfg is not None:
            self.tail_trans = tail_trans_cfg.gen(str2net[tail_trans_cfg.trans_layer], **kwargs)
    def forward(self, x):
        """ x: [b, c, w, h] => [b, c', w', h'] or [b, l, c']"""
        batch_size = x.shape[0]
        # x: [b, c, w, h] => [b, c', w', h']
        for layer in self.main_layers:
            x = layer(x)
        if self.tail_trans_cfg is not None:
            # x: [b, c', w', h'] => [b, c', l]
            x = x.reshape(batch_size, self.cfg[-1].outc, -1).permute(0, 2, 1)
            for layer in self.tail_trans:
                x = layer(x)
        return x

class COneD(MATTPBase):#MA TT with permute
    def __init__(self, args: GlobalSettings, cfg:List[LayerCfg], layer:COneDLayer,
                 tail_trans_cfg:TransCfg = None, **kwargs):
        """ x: [b, c', l] or [b, D]"""
        super(COneD, self).__init__(args, cfg, layer, tail_trans_cfg, **kwargs)
    
class MATTP(MATTPBase):#MA TT with permute
    def __init__(self, args: GlobalSettings, cfg:List[LayerCfg], layer:MAlayer,
                 tail_trans_cfg:TransCfg = None, **kwargs):
        """ x: [b, c, w, h] => [b, c', w', h'] or [b, c', l] or [b, D]"""
        super().__init__(args, cfg, layer, tail_trans_cfg, **kwargs)
         
class MAvTTP(MATTPBase):#MA TT with permute
    def __init__(self, args: GlobalSettings, cfg:List[LayerCfg], layer:MAvlayer,
                 tail_trans_cfg:TransCfg = None, **kwargs):
        """ x: [b, c, w, h] => [b, c', w', h'] or [b, c', l] or [b, D]"""
        super().__init__(args, cfg, layer, tail_trans_cfg, **kwargs)
         
class MATTPE(MATTPBase):#MA TT with permute
    def __init__(self, args: GlobalSettings, cfg:List[LayerCfg], layer:MAvlayer,
                 tail_trans_cfg:TransCfg = None, **kwargs):
        """ x: [b, c, w, h] => [b, c', w', h'] or [b, c', l] or [b, D]"""
        super().__init__(args, cfg, layer, tail_trans_cfg, **kwargs)
        self.pos_embedding = bb.PositionalEncoding(cfg[-1].outc)
    def forward(self, x):
        """ x: [b, c, w, h] => [b, c', w', h'] or [b, l, c']"""
        batch_size = x.shape[0]
        # x: [b, c, w, h] => [b, c', w', h']
        for layer in self.main_layers:
            x = layer(x)
        if self.tail_trans_cfg is not None:
            # x: [b, c', w', h'] => [b, c', l] => [b, l, c']
            x = x.reshape(batch_size, self.cfg[-1].outc, -1).permute(0, 2, 1)
            # positional_encoding
            x = self.pos_embedding(x)
            for layer in self.tail_trans:
                x = layer(x)
        return x
    
class SCANNTTP(MATTPBase):#MA TT with permute
    def __init__(self, args: GlobalSettings, cfg:List[LayerCfg], layer:SCANlayer,
                 tail_trans_cfg:TransCfg = None, **kwargs):
        """ x: [b, c, w, h] => [b, c', w', h'] or [b, c', l] or [b, D]"""
        super().__init__(args, cfg, layer, tail_trans_cfg, **kwargs)
        
class MATTP_ViT(MATTPBase):  # MA TT with permute
    def __init__(self, args: GlobalSettings, cfg:List[LayerCfg], layer:MAvlayer,
                 tail_trans_cfg:TransCfg = None, **kwargs):
        """ x: [b, c, w, h] => [b, c', w', h'] or [b, c', l] or [b, D]"""
        super().__init__(args, cfg, layer, tail_trans_cfg, **kwargs)
        # Uinfo: [b,2,128]
        self.uni_info = nn.Parameter(torch.randn(1, 2, cfg[-1].outc))
    def forward(self, x):  # x: [b,3,128,128]
        # x: [b, c, w, h] => [b, c', w', h']
        for layer in self.main_layers:
            x = layer(x)
        batch_size, c, w, h = x.shape
        # x: [b, c', w', h'] => [b, c', l] => [b, l, c']
        x = x.reshape(batch_size, c, w*h).permute(0, 2, 1)
        # x: [b, l+2, c']
        x = torch.cat([self.uni_info.repeat(batch_size, 1, 1), x], dim=1)
        if self.tail_trans_cfg is not None:
            # x: [b, c', w', h'] => [b, c', l] => [b, l, c']
            x = x.reshape(batch_size, self.cfg[-1].outc, -1).permute(0, 2, 1)
            for layer in self.tail_trans:
                x = layer(x)
        return x