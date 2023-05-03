'''
Author: BHM-Bob 2262029386@qq.com
Date: 2023-05-02 20:40:37
LastEditors: BHM-Bob
LastEditTime: 2023-05-02 23:35:23
Description: 
'''
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(r'../../../')

import dl_torch as dt
dt._Params['USE_VIZDOM'] =False
from dl_torch.utils import Mprint, GlobalSettings
from dl_torch.data import DataSetRAM

# global settings
mp = Mprint()
args = GlobalSettings(mp, 'model/')
args.add_arg('read', {'path':r"D:\AI\DataSet\Seq2ImgFluently\seq\Seq\wordSeqs_int16_.seq"})
args.add_arg('load_shape', [64*64])

# load data
# x: [[novel_name:str, seq:tensor], ...]
ds = DataSetRAM(args, x = args.read['path'],
                x_transfer_origin=lambda path : torch.load(path),
                x_transfer_gather=lambda x : x[0])
train_loader, test_loader = ds.split([0, 0.7, 1],
                                     x_transformer=lambda x : x[1][0:args.load_shape[0]],
                                     y_transformer=lambda y : 0)

# training
for x, _ in train_loader:
    print(x.shape)
    break