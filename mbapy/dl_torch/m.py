'''
Author: BHM-Bob 2262029386@qq.com
Date: 2023-03-23 21:50:21
LastEditors: BHM-Bob
LastEditTime: 2023-03-30 14:07:27
Description: Model
'''

import math
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import bb

def calcu_q_len(input_size:int, strides:list[int], dims:int = 1):
    """
    calcu q_len for Conv model
    strides: list of each layers' stride
    input_size : when dim is 2, it must be set as img 's size (w==h)
    """
    ret = input_size
    for s in range(strides):
        while not ret % s == 0:
            ret += 1
        ret /= 2
    return int(ret**dims)

class COneDLayer(nn.Module):
    def __init__(self, inc:int, outc:int, cnn_kernel:int, layer = bb.SABlock1DR,
                 use_trans:bool = True, trans_layer = bb.EncoderLayer, device = 'cuda', **kwargs):
        self.layer = layer(inc, outc, cnn_kernel)
        self.use_trans = use_trans
        if self.use_trans:
            self.trans = trans_layer(outc, 8, 2 * self.outc, 0.3, device, **kwargs)
    def forward(self, x):
        # [b, c', l']
        x = self.layer(x)
        if self.use_trans:
            # x: [b, c', l'] => [b, l', c'] => [b, c', l']
            x = self.trans(x.permute(0, 2, 1)).permute(0, 2, 1)
        # [b, c', l']
        return x

class MAlayer(nn.Module):
    def __init__(self, inc:int, outc:int, cnn_kernel:int, layer = bb.ResBlock,
                 use_SA:bool = True, SA_layer = bb.SABlockR, device = 'cuda', **kwargs):
        super(MAlayer, self).__init__()
        self.use_SA = use_SA
        if self.use_SA:
            self.SA = SA_layer(inc, outc, cnn_kernel)
            self.layer = nn.Sequential(layer(outc, outc, 2),
                                       layer(outc, outc, 1))
        else:
            self.layer = nn.Sequential(layer(inc, outc, 2),
                                       layer(outc, outc, 1))
    def forward(self, x):
        if self.use_SA:
            x = self.SA(x)
        return self.layer(x)

class MAvlayer(MAlayer):
    def __init__(self, inc:int, outc:int, cnn_kernel:int, layer = bb.ResBlock,
                 use_SA:bool = True, SA_layer = bb.SABlockR, device = 'cuda', **kwargs):
        super().__init__(inc, outc, cnn_kernel, layer, use_SA, SA_layer, device = 'cuda', **kwargs)
        if self.use_SA:
            self.SA = SA_layer(inc, outc, minCnnKSize=self.minCnnKSize)
            self.layer = nn.Sequential(nn.AvgPool2d((2, 2), 2),
                                       layer(outc, outc, 1))
        else:
            self.layer = nn.Sequential(nn.AvgPool2d((2, 2), 2),
                                       layer(inc, outc, 1))

class SCANlayer(nn.Module):
    def __init__(self, inc:int, outc:int, cnn_kernel:int, layer = bb.SCANN,
                 use_SA:bool = True, SA_layer = bb.SABlockR, device = 'cuda', **kwargs):
        super(SCANlayer, self).__init__()
        if self.isSA:
            self.layer = SA_layer(self.inc, self.outc)
        else:
            self.layer = nn.Conv2d(self.inc, self.outc, 1, 1, 0)
        self.scann = layer(self.pic_size, self.inc, padding="half", outway="avg")
        self.shoutCut = nn.AvgPool2d((2, 2), stride=2, padding=0)
    def forward(self, x):
        t = self.shoutCut(x)
        x = self.scann(x)
        x = self.layer(t + x)
        return x



class MATTPBase(nn.Module):#MA TT with permute
    def __init__(self, args: MyArgs, convnum:list, ConvLayer):
        super(MATTPBase, self).__init__()
        self.picsize = int(math.sqrt(args.seqLen))
        assert args.seqLen % self.picsize == 0
        self.channels = args.channels
        self.classnum = args.sumClass
        self.stemconvnum = 32
        self.pf_dim = 256
        self.n_heads = 4
        self.n_layers = 2
        self.convnum = convnum
        self.q_len = CacuQLenTwoD(self.picsize, len(self.convnum))
        mp.mprint('\nconvnum = {''}    \nq_len = {:2d}'.format(str(self.convnum), self.q_len))
        self.MAlayers = nn.ModuleList([ ConvLayer(convnum) for convnum in self.convnum ])
        # self.tailTrans = nn.ModuleList(
        #         [
        #             EncoderLayer(args.dim, self.n_heads, self.pf_dim, 0.3, args.gpu)
        #             for _ in range(self.fusionLayers)
        #         ]
        #     )
    def forward(self, x):#x: [b, c, L]
        batchSize = x.shape[0]
        # x: [b, c, L] => [b, c, w, h]
        x = x.reshape(batchSize, self.channels, self.picsize, self.picsize)
        # x: [b, c, w, h] => [b, c', w', h']
        for maLayer in self.MAlayers:
            x = maLayer(x)
        # x: [b, c', w', h'] => [b, c', l] => [b, l, c']
        x = x.reshape(batchSize,self.convnum[-1][1],self.q_len).permute(0,2,1)
        # # x: [b, l, c'] => [b, l, c']
        # for layer in self.tailTrans:
        #     x = layer(x)
        return x





class COneD(MATTPBase):#MA TT with permute
    def __init__(self, args: MyArgs):
        self.c = 32
        self.coAt = [
            [args.channels, 1 * self.c, 4, False, 7],  # [b, 16,  10*40]
            [1 * self.c   , 2 * self.c, 2, False, 5],  # [b, 16,  5*40]
            [2 * self.c   , 2 * self.c, 2, True,  3],  # [b, 16,  5*20]
            [2 * self.c   ,   args.dim, 2, True,  3],  # [b, 16,  5*10]
        ]  
        super(COneD, self).__init__(args, self.coAt, COneDLayer)
        self.q_len = int(
            args.seqLen / CacuQLenOneD([cnn[2] for cnn in self.coAt])
        )
        mp.mprint('\ncnn = {''}    \nq_len = {:2d}'.format(str(self.coAt), self.q_len))
    def forward(self, x):  # x: [512, 3, 1600]
        batchSize = x.shape[0]
        # x: [b, c, L] => [b, c', l]
        for maLayer in self.MAlayers:
            x = maLayer(x)
        # x: [b, c', l] => [b, l, c']
        x = x.permute(0, 2, 1)
        # # x: [b, l, c'] => [b, l, c']
        # for layer in self.tailTrans:
        #     x = layer(x)
        # #[b, l, c']
        return x
    
class MATTP(MATTPBase):#MA TT with permute
    def __init__(self, args: MyArgs):
        self.stemconvnum = 32
        self.convnum = [
            [args.channels       , 1 * self.stemconvnum, True],  # [b,32, 20,20]
            [1 * self.stemconvnum, 2 * self.stemconvnum, True],  # [b,64, 10,10]
            [2 * self.stemconvnum,             args.dim, True],  # [b,128, 5, 5]
        ]  
        super(MATTP, self).__init__(args, self.convnum, MAlayer)
         
class MAvTTP(MATTPBase):#MA TT with permute
    def __init__(self, args: MyArgs):
        self.stemconvnum = 32
        self.convnum = [
            [args.channels       , 1 * self.stemconvnum, False, 7],  # [b,32,40,h]
            [1 * self.stemconvnum, 2 * self.stemconvnum,  True, 7],  # [b,64,20,h]
            [2 * self.stemconvnum,             args.dim,  True, 5],
        ]  
        super(MAvTTP, self).__init__(args, self.convnum, MAvlayer)
         
class MATTPE(MATTPBase):#MA TT with permute
    def __init__(self, args: MyArgs):
        self.stemconvnum = 32
        self.convnum = [
            [args.channels       , 1 * self.stemconvnum, False],  # [b,32,40,h]
            [1 * self.stemconvnum, 2 * self.stemconvnum,  True],  # [b,64,20,h]
            [2 * self.stemconvnum,             args.dim,  True],
        ]  
        super(MATTPE, self).__init__(args, self.convnum, MAvlayer)
        self.pos_embedding = PositionalEncoding(args.dim, self.q_len)
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
    def __init__(self, args: MyArgs):
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
                    EncoderLayer(
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