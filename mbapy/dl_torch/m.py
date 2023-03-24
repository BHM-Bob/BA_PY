'''
Author: BHM-Bob 2262029386@qq.com
Date: 2023-03-23 21:50:21
LastEditors: BHM-Bob
LastEditTime: 2023-03-24 22:14:33
Description: Model
'''

import math
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def CacuQLenOneD(a:list[int]):
    """Cacu q_len for Conv1D model
    a: list of each stride
    """
    ret = 1
    for i in a:
        ret *= i
    return ret

def CacuQLenTwoD(picSize:int, a:int):
    """Cacu q_len for Conv2D model
    picSize: W or H
    a: num of sum downsample cnn layers(size = 3, stride = 2, padding = 1)
    """
    ret = picSize
    for i in range(a):
        while not ret % 2 == 0:
            ret += 1
        ret /= 2
    return int(ret**2)


class COneDLayer(nn.Module):
    def __init__(self, conv):
        super(COneDLayer, self).__init__()
        self.inc = conv[0]
        self.outc = conv[1]
        self.avgK = conv[2]
        self.isTrans = conv[3]
        self.cnnKSize = conv[4]
        self.downSampler = SABlock1DR2(self.inc, self.outc, self.avgK, self.cnnKSize)
        if self.isTrans:
            self.trans = EncoderLayer(self.outc, 8, 2 * self.outc, 0.3, 'cuda')
            self.extra = nn.LeakyReLU()
    def forward(self, x):
        batchSize = x.shape[0]
        # [b, c', l']
        x = self.downSampler(x)

        if self.isTrans:
            # shortCut: [b, c', l']
            shortCut = self.extra(x)
            # x: [b, c', l'] => [b, l', c'] => [b, c', l']
            x = self.trans(x.permute(0, 2, 1)).permute(0, 2, 1)
            # Res
            x = x + shortCut
        # [b, c', l']
        return x


class MAlayer(nn.Module):  # Cifar10 [b,3,32,32] => [b,10]
    def __init__(self, conv):
        super(MAlayer, self).__init__()
        self.inc = conv[0]
        self.outc = conv[1]
        self.isSA = conv[2]
        if self.isSA:
            self.SA = SABlock(self.inc, self.outc)
            self.layer = nn.Sequential(ResBlock(self.outc, self.outc, 2),
                                       ResBlock(self.outc, self.outc, 1))
        else:
            self.layer = nn.Sequential(ResBlock(self.inc, self.outc, 2),
                                       ResBlock(self.outc, self.outc, 1))
    def forward(self, x):
        if self.isSA:
            x = self.SA(x)
        return self.layer(x)


class MAvlayer(nn.Module):  # Cifar10 [b,3,32,32] => [b,10]
    def __init__(self, conv):
        super(MAvlayer, self).__init__()
        self.inc = conv[0]
        self.outc = conv[1]
        self.isSA = conv[2]
        self.minCnnKSize = conv[3]
        if self.isSA:
            self.layer = nn.Sequential(nn.AvgPool2d((2, 2), 2),
                                       nn.Conv2d(self.inc, self.inc, 1, 1, 0))
            self.SA = SABlockR(self.inc, self.outc, minCnnKSize=self.minCnnKSize)
        else:
            self.layer = nn.Sequential(nn.AvgPool2d((2, 2), 2),
                                       nn.Conv2d(self.inc, self.outc, 1, 1, 0))
    def forward(self, x):
        x = self.layer(x)
        if self.isSA:
            x = self.SA(x)
        return x


class SCANlayer(nn.Module):  # Cifar10 [b,3,32,32] => [b,10]
    def __init__(self, conv):
        super(SCANlayer, self).__init__()
        self.inc = conv[0]
        self.outc = conv[1]
        self.pic_size = conv[2]
        self.isSA = conv[3]
        if self.isSA:
            self.layer = SABlock(self.inc, self.outc)
        else:
            self.layer = nn.Conv2d(self.inc, self.outc, 1, 1, 0)
        self.SCAN = SCANN(self.pic_size, self.inc, padding="half", outway="avg")
        self.shoutCut = nn.AvgPool2d((2, 2), stride=2, padding=0)
    def forward(self, x):
        t = self.shoutCut(x)
        x = self.SCAN(x)
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