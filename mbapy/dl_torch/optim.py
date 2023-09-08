'''
Date: 2023-05-26 09:02:43
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-05-30 09:58:26
FilePath: /BA_PY/mbapy/dl_torch/optim.py
Description: 
'''
import math

import torch
import torch.optim

from ..base import autoparse


def _ConsineDown(lr_0:float, now_epoch:int, T_0:int, sum_epoch:int):
    return lr_0 * 0.5 * (1. + math.cos(math.pi * now_epoch / sum_epoch))

def _ConsineAnnealing(lr_0:float, now_epoch:int, T_0:int, sum_epoch:int):
    return lr_0 * 0.5 * (1. + math.cos(math.pi * now_epoch / T_0))

def _DownConsineAnnealing(lr_0:float, now_epoch:int, T_0:int, sum_epoch:int):
    return lr_0 * 0.5 * (1. + math.cos(math.pi * now_epoch/sum_epoch)) * 0.5 * (1. + math.cos(math.pi * now_epoch / T_0))

def _DownScaleConsineAnnealing(lr_0:float, now_epoch:int, T_0:int, sum_epoch:int):
    return lr_0 * 0.5 * (1. + math.cos(math.pi * now_epoch/sum_epoch)) * 0.5 * (1. + math.cos(math.pi * now_epoch / (0.1 * now_epoch + T_0)))

def _DownScaleRConsineAnnealing(lr_0:float, now_epoch:int, T_0:int, sum_epoch:int):
    scale = T_0 / (T_0 + now_epoch)
    lr = 0.5 * scale * math.cos(math.pi * now_epoch / (now_epoch * 0.05 + T_0)) + scale
    return lr_0 * max(1e-5, min(lr, 1))

_str2scheduleF = {
    'ConsineDown':_ConsineDown,
    'ConsineAnnealing':_ConsineAnnealing,
    'DownConsineAnnealing':_DownConsineAnnealing,
    'DownScaleConsineAnnealing':_DownScaleConsineAnnealing,
    'DownScaleRConsineAnnealing':_DownScaleRConsineAnnealing,
}

class LrScheduler:
    r"""
    Step method could be called after every batch update
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lr_0: origin lr or max lr
        now_epoch: now epoch, 0 or a positive number when loaded a checkpoint
        T_0: min T
        sum_epoch: sum epoch
    method:
        ConsineDown: lr_t = lr_0 * 0.5 * (1. + cos(pi * now_epoch / epochs))
        ConsineAnnealing: lr_t = lr_0 * 0.5 * (1. + cos(pi * now_epoch / T_0))
        DownConsineAnnealing: lr_t = lr_0 * 0.5 * (1. + cos(pi * now_epoch / epochs)) * 0.5 * (1. + cos(pi * now_epoch / T_0))
        DownScaleConsineAnnealing: lr_t = lr_0 * 0.5 * (1. + cos(pi * now_epoch / epochs)) * 0.5 * (1. + cos(pi * now_epoch / (now_epoch + T_0))) 
        DownScaleRConsineAnnealing: lr_t = lr_0 * CLIP[ 0.5 * (T_0 / (T_0 + now_epoch)) * cos(pi * now_epoch / (0.05 * now_epoch + T_epoch))) + T_0 / (T_0 + now_epoch), 1e-5, 1]
    """
    @autoparse
    def __init__(self, optimizer:torch.optim.Optimizer, lr_0:float,
                 now_epoch:int = 0, T_0:int = 100, sum_epoch:int = 5000,
                 method = '_ConsineDown'):
        assert lr_0 > 0, r'lr_0 <= 0'
        assert now_epoch >= 0, r'now_epoch < 0'
        assert T_0 > 0, 'T_0 <= 0'
        assert sum_epoch >= now_epoch, 'sum_epoch < now_epoch'
        assert method in _str2scheduleF.keys(), 'method not in _str2scheduleF.keys()'
        self.lr = lr_0
        self._get_lr_ = _str2scheduleF[method]
    
    def add_epoch(self, n:int):
        self.sum_epoch += n
    
    def edited_ext_epoch(self, n:int):
        self.sum_epoch  = self.now_epoch + n

    def step(self, epoch:float):
        """Step could be called after every batch update
        Example:
            >>> iters = len(dataloader)
            >>> for epoch in range(20):
            >>>     for i, sample in enumerate(dataloader):
            >>>         ...
            >>>         outputs = net(inputs)
            >>>         ...
            >>>         optimizer.step()
            >>>         scheduler.step(epoch + i / iters)
        """
        self.now_epoch = epoch
        self.lr = self._get_lr_(self.lr_0, self.now_epoch, self.T_0, self.sum_epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        return self.lr