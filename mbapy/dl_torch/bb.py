'''
Author: BHM-Bob 2262029386@qq.com
Date: 2023-03-23 21:50:21
LastEditors: BHM-Bob
LastEditTime: 2023-03-24 12:41:33
Description: Basic Blocks
'''

import math
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScannCore(nn.Module):
    def __init__(self, inc, s, way="linear", dropout=0.2):
        super(ScannCore, self).__init__()
        self.inc = inc
        self.s = s
        self.fc_q = nn.Linear(s, s)
        self.fc_k = nn.Linear(s, s)
        self.fc_v = nn.Linear(s, s)
        self.dropout = nn.Dropout(dropout)
        if way == "linear":
            self.out = nn.Linear(s, 1)
        else:
            self.out = nn.AvgPool1d(s, 1, 0)
    def forward(self, x):
        # x = [b,inc,s]
        Q = self.fc_q(x)  # Q = [b,inc,s]
        K = self.fc_k(x).permute(0, 2, 1)  # K = [b,s,inc]
        V = self.fc_v(x).permute(0, 2, 1)  # V = [b,s,inc]
        # energy = [b,s,s]
        energy = torch.matmul(K, Q)
        # attention = [b,s,s]
        attention = torch.softmax(energy, dim=-1)
        # x = [b,s,inc]
        x = torch.matmul(self.dropout(attention), V)
        # x = [b,s,inc]=>[b,inc,s]
        x = x.permute(0, 2, 1)
        # x = [b,inc,s]=>[b,inc,1]
        return self.out(x)

class SCANN(nn.Module):
    """params:
        img_size : int,# means img must be (w == h)
        inc : int, # input channle
        group : int=1,# means how many channels are in a group to get into ScannCore
        stride : int=2,
        padding : int=1,# F.unflod的padding是两边(四周)均pad padding个0
        kernel_size : int=3,
        outway : str="linear", # linear or avg
    """
    def __init__( self,
        img_size : int,# means img must be (w == h)
        inc : int, # input channle
        group : int=1,# means how many channels are in a group to get into ScannCore
        stride : int=2,
        padding : int=1,# F.unflod的padding是两边（四周）均pad padding个0
        kernel_size : int=3,
        outway : str="linear", # linear or avg
        dropout : float=0.2,
    ):
        super(SCANN, self).__init__()
        assert inc % group == 0
        self.inc = inc
        self.group = group
        assert kernel_size < img_size
        assert (img_size + 2*padding - kernel_size) % stride == 0
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.side_patch_num = (img_size + 2*padding - kernel_size) // stride + 1
        self.patch_num = (self.side_patch_num) ** 2
        self.patch_size = kernel_size**2

        self.SAcnn = nn.ModuleList(
            [
                ScannCore(self.patch_num * self.group, self.patch_size, outway)
                for _ in range(inc // group)
            ]
        )
    def ScannCoreMiniForward(self, x, i):
        # x = [b, group, h, w]
        batch_size = x.shape[0]
        # t = [b,self.group*self.patch_size,self.patch_num]
        t = F.unfold(x, self.kernel_size, 1, self.padding, self.stride)
        # t = [b,self.patch_num*self.group,self.patch_size]
        t = (
            t.reshape(batch_size, self.group, self.patch_size, self.patch_num)
            .permute(0, 1, 3, 2)
            .reshape(batch_size, self.group * self.patch_num, self.patch_size)
        )
        # t = [b,self.patch_num*self.group,1]
        t = self.SAcnn[i](t)
        # t = [b,self.group,self.side_patch_num,self.side_patch_num]
        t = (
            t.reshape( batch_size, self.side_patch_num, self.side_patch_num, self.group)
            .permute(0, 3, 1, 2)
            )
        return t
    def forward(self, x):
        # x = [b, inc, h, w]
        if self.group == self.inc:
            return self.ScannCoreMiniForward(x, 0)
        else:
            return torch.cat([ self.ScannCoreMiniForward(xi, i) for i, xi in
                          enumerate(torch.split(x, self.group, dim=1))], dim=1)

class PositionalEncoding(nn.Module):
# from https://zhuanlan.zhihu.com/p/338592312
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        #pe.requires_grad = False
        self.register_buffer('pe', pe)
    def forward(self, x):
        return self.pe.repeat(x.shape[0], 1, 1).add(x)

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(hid_dim, pf_dim),
            nn.Dropout(dropout),
            nn.ReLU(True),
            nn.Linear(pf_dim, hid_dim),
        )
    def forward(self, x):
        # x = [batch size, seq len, hid dim]
        return self.nn(x)

class MultiHeadAttentionLayer(nn.Module):
    """MultiHeadAttentionLayer\n
    if kwargs['use_enhanced_fc_q'] and 'q_len' in kwargs and 'out_len' in kwargs\n
    use fc_q mlp like PositionwiseFeedforwardLayer to output a tensor with out_len\n
    if 'out_dim' in kwargs\n
    self.fc_o = nn.Linear(hid_dim, kwargs['out_dim'])
    """
    def __init__(self, hid_dim, n_heads, dropout, device = 'cuda', **kwargs):
        super().__init__()
        assert hid_dim % n_heads == 0
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        if 'out_dim' in kwargs:
            self.fc_o = nn.Linear(hid_dim, kwargs['out_dim'])
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
    def forward(self, query, key, value):
        batch_size = query.shape[0]
        # Q = [batch size, query len, hid dim] => [batch size, n heads, query len, head dim]
        # K = [batch size, key len,   hid dim] => [batch size, n heads, key len  , head dim]
        # V = [batch size, value len, hid dim] => [batch size, n heads, value len, head dim]
        Q = self.fc_q(query).\
            reshape(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.fc_k(key).\
            reshape(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.fc_v(value).\
            reshape(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)        
        # attention = [batch size, n heads, query len, key len]
        attention = Q.matmul(K.permute(0, 1, 3, 2)).multiply(self.scale). softmax(dim=-1)
        # x = [batch size, query len, hid dim]
        x = self.dropout(attention).matmul(V).\
            permute(0, 2, 1, 3).contiguous().\
            reshape(batch_size, -1, self.hid_dim)
        # x = [batch size, query len, hid dim]
        return self.fc_o(x)

class OutMultiHeadAttentionLayer(MultiHeadAttentionLayer):
    """OutMultiHeadAttentionLayer\n
    if kwargs['use_enhanced_fc_q'] and 'q_len' in kwargs and 'out_len' in kwargs\n
    use fc_q mlp like PositionwiseFeedforwardLayer to output a tensor with out_len\n
    if 'out_dim' in kwargs\n
    self.fc_o = nn.Linear(hid_dim, kwargs['out_dim'])
    """
    def __init__(self, q_len, class_num, hid_dim, n_heads, dropout, device = 'cuda', **kwargs):
        super().__init__(hid_dim, n_heads, dropout, device, **kwargs)
        if kwargs['use_enhanced_fc_q']:
            self.fc_q = nn.Sequential(nn.Linear(q_len, hid_dim),
                                      nn.GELU(),
                                      nn.Dropout(dropout),
                                      nn.Linear(hid_dim, class_num))
    def forward(self, query, key, value):
        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]
        batch_size = query.shape[0]
        # Q = [batch size, n heads, class num, head dim]
        # K = [batch size, n heads, key len  , head dim]
        # V = [batch size, n heads, value len, head dim]
        Q = self.fc_q(query.permute(0, 2, 1)).permute(0, 2, 1).\
            reshape(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.fc_k(key).\
            reshape(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.fc_v(value).\
            reshape(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # attention = [batch size, n heads, class num, key len]
        attention = torch.matmul(Q, K.permute(0, 1, 3, 2)).multiply(self.scale).softmax(dim=-1)
        # x = [batch size, n heads, class num, head dim]
        # x = [batch size, class num, n heads, head dim]
        x = self.dropout(attention).matmul(V).\
            permute(0, 2, 1, 3).contiguous().\
            reshape(batch_size, -1, self.hid_dim)
        # x = [batch size, class num, hid dim]
        return self.fc_o(x)
    
class EncoderLayer(nn.Module):
    def __init__(self, q_len, class_num, hid_dim, n_heads, pf_dim, dropout, device = 'cuda', **kwargs):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        if not 'do_not_ff' in kwargs:
            self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)
    def forward(self, src):
        # src = [batch size, src len, hid dim]
        # self attention
        _src = self.self_attention(src, src, src)
        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        # src = [batch size, src len, hid dim]
        # positionwise feedforward
        _src = self.positionwise_feedforward(src)
        # dropout, residual and layer norm
        # ret = [batch size, src len, hid dim]
        return self.ff_layer_norm(src + self.dropout(_src))

class OutEncoderLayer(EncoderLayer):
    def __init__(self, q_len, class_num, hid_dim, n_heads, pf_dim, dropout, device = 'cuda', **kwargs):
        kwargs.update({'do_not_ff':True})
        super().__init__(q_len, class_num, hid_dim, n_heads, pf_dim, dropout, device, **kwargs)
        self.class_num = class_num
        self.self_attention = OutMultiHeadAttentionLayer(
            q_len, class_num, hid_dim, n_heads, dropout, device, **kwargs
        )
        self.fc_out = nn.Linear(hid_dim, 1)
    def forward(self, src):
        # src = [batch size, src len, hid dim]
        # self attention
        # src = [batch size, class num, hid dim]
        src = self.self_attention(src, src, src)
        # dropout, and layer norm
        # src = [batch size, class num, hid dim]
        src = self.self_attn_layer_norm(self.dropout(src))
        # ret = [batch size, class num]
        return self.fc_out(src).reshape(-1, self.class_num)

class Trans(nn.Module):
    """[batch size, src len, hid dim]"""
    def __init__(self, q_len:int, class_num:int, hid_dim:int, n_layers:int, n_heads:int, pf_dim:int,
                 dropout:float, device:str, out_encoder_layer = OutEncoderLayer, **kwargs):
        super().__init__()
        assert n_layers > 0
        self.nn = nn.Sequential(
            *([EncoderLayer(q_len, class_num, hid_dim, n_heads, pf_dim, dropout, device, **kwargs) \
                for _ in range(n_layers - 1)]+\
                [out_encoder_layer(q_len, class_num, hid_dim, n_heads, pf_dim, dropout, device, **kwargs)]))
    def forward(self, src):
        return self.nn(src)
    
class OutEncoderLayerRAvg(nn.Module):
    def __init__(self, k1, k2, new_shape):
        super().__init__()
        self.nn = nn.Sequential(
            nn.AvgPool1d(k1, 1, 0),
            lambda x : x.reshape(new_shape),
            nn.AvgPool1d(k2, k2, 0),
        )
    def forward(self, x):
        return self.nn(x)
class OutEncoderLayerR(OutEncoderLayer):
    def __init__(self, q_len, class_num, hid_dim, n_heads, pf_dim, dropout, device, **kwargs):
        super().__init__(q_len, class_num, hid_dim, n_heads, pf_dim, dropout, device, **kwargs)
        self.q_len = q_len
        self.class_num = class_num
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device, **kwargs)
        #self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,pf_dim,dropout)
        self.dropout = nn.Dropout(dropout)
        if q_len > class_num:
            assert q_len % class_num == 0
            self.avgOut = OutEncoderLayerRAvg(hid_dim, int(q_len/class_num),
                                               newShape = (-1, q_len))
        elif q_len < class_num:
            assert class_num % q_len == 0
            ks = int(q_len * hid_dim / class_num)
            self.avgOut = nn.AvgPool1d(ks, ks, 0)
        elif q_len == class_num:
            self.avgOut = nn.AvgPool1d(hid_dim, 1, 0)
    def forward(self, src):
        batchSize = src.shape[0]
        #src = [batch size, src len, hid dim]  
        src = self.self_attention(src, src, src)
        #src = [batch size, src len, hid dim] 
        src = self.self_attn_layer_norm(self.dropout(src))      
        #x: [b, l', c'//avgKernelSize]
        src = self.avgOut(src).reshape(batchSize, self.class_num)
        #retrun [batch size, class_num]   
        return src
class TransR(nn.Module):
    def __init__(self, q_len:int, class_num:int, hid_dim:int, n_layers:int, n_heads:int, pf_dim:int,
                 dropout:float, device:str, **kwargs):
        super(Trans).__init__(q_len,class_num, hid_dim, n_layers, n_heads, pf_dim,
                              dropout, device, OutEncoderLayerR, kwargs)

class TransPE(Trans):
    def __init__(self, q_len:int, class_num:int, hid_dim:int, n_layers:int, n_heads:int, pf_dim:int,
                 dropout:float, device:str, **kwargs):
        super(Trans).__init__(q_len,class_num, hid_dim, n_layers, n_heads, pf_dim, dropout, device, kwargs)
        self.pos_embedding = PositionalEncoding(hid_dim, q_len)
    def forward(self, src):
        # src = [batch size, src len, hid dim]
        src = self.pos_embedding(src)
        # src = [batch size, class_num]
        return self.nn(src)


class SeparableConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size, stride, padding, depth = 1):
        super(SeparableConv2d, self).__init__()
        self.nn = (
            nn.Conv2d(inc, outc*depth, kernel_size, stride, padding, groups=inc),# depthwise 
            nn.Conv2d(outc*depth, outc, kernel_size=1),# pointwise 
        )
    def forward(self, x):
        return self.nn(x)

class ResBlock(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = SeparableConv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bnl = nn.BatchNorm2d(ch_out)
        self.conv2 = SeparableConv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out),
            )
        else:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out),
            )

    def forward( self, inputs ):  # [b,ch_in,w,h] => [b,ch_out,w/2,h/2]  (stride = 2,w and h +1 %3 ==0)
        out = F.leaky_relu(self.bnl(self.conv1(inputs)))
        out = self.bn2(self.conv2(out))
        t = self.extra(inputs)
        out = t + out
        return out


class SABlock(nn.Module):
    def __init__(self, inc, outc):
        super(SABlock, self).__init__()
        # [ B , C , H , W ]
        self.convh1 = nn.Conv2d(inc, outc // 4, (7, 5), stride=1, padding="same")
        self.convh1_ = nn.Conv2d(
            inc + outc // 4, outc // 4, (7, 5), stride=1, padding="same"
        )
        self.bnh1 = nn.BatchNorm2d(inc + outc // 4)
        self.bnh2 = nn.BatchNorm2d(inc + outc // 4)
        self.convh2 = nn.Conv2d(inc, outc // 4, (5, 3), stride=1, padding="same")
        self.convh2_ = nn.Conv2d(
            inc + outc // 4, outc // 4, (5, 3), stride=1, padding="same"
        )

        self.convw1 = nn.Conv2d(inc, outc // 4, (5, 7), stride=1, padding="same")
        self.convw1_ = nn.Conv2d(
            inc + outc // 4, outc // 4, (5, 7), stride=1, padding="same"
        )
        self.bnw1 = nn.BatchNorm2d(inc + outc // 4)
        self.bnw2 = nn.BatchNorm2d(inc + outc // 4)
        self.convw2 = nn.Conv2d(inc, outc // 4, (3, 5), stride=1, padding="same")
        self.convw2_ = nn.Conv2d(
            inc + outc // 4, outc // 4, (3, 5), stride=1, padding="same"
        )

    def forward(self, inputs):  # [b,inc,h,w] => [b,outc,h,w]
        outh1 = self.convh1(inputs)
        outh1 = self.convh1_(F.leaky_relu(self.bnh1(torch.cat([outh1, inputs], dim=1))))
        outh2 = self.convh2(inputs)
        outh2 = self.convh2_(F.leaky_relu(self.bnh2(torch.cat([outh2, inputs], dim=1))))

        outw1 = self.convw1(inputs)
        outw1 = self.convw1_(F.leaky_relu(self.bnw1(torch.cat([outw1, inputs], dim=1))))
        outw2 = self.convw2(inputs)
        outw2 = self.convw2_(F.leaky_relu(self.bnw2(torch.cat([outw2, inputs], dim=1))))

        return torch.cat([outh1, outh2, outw1, outw2], dim=1)

class SABlockR(nn.Module):
    def __init__(self, inc, outc, minCnnKSize = 3):
        super(SABlockR, self).__init__()
        mS = minCnnKSize
        self.cnnK = [(mS+4, mS+2), (mS+2, mS+0), (mS+2, mS+4), (mS+0, mS+2)]
        def GenCnn(inChannles: int, outChannles: int, cnnKernel):
            ret = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(inChannles, outChannles, k, stride=1, padding="same"),
                    nn.BatchNorm2d(outChannles),
                    nn.LeakyReLU()
                )
                for k in cnnKernel
            ])
            return ret if len(cnnKernel) > 1 else ret[0]
        self.cnn1 = GenCnn(inc, outc // 4, self.cnnK)
        self.cnn2 = GenCnn(outc, outc // 4, self.cnnK)
        self.shortCut = GenCnn(inc, outc, [(1, 1)])
    def forward(self, inputs):  # [b,inc,h,w] => [b,outc,h,w]
        out = torch.cat([ cnn(inputs) for cnn in self.cnn1 ], dim=1)
        out = torch.cat([ cnn(out) for cnn in self.cnn2 ], dim=1)
        shortCut = self.shortCut(inputs)
        return out + shortCut


class SABlock1D(nn.Module):
    def __init__(self, inc, outc, avgK, minCnnKSize):
        super(SABlock1D, self).__init__()
        # [ B , L , C]
        self.convh1 = nn.Conv1d(inc, outc // 4, 9, stride=1, padding="same")
        self.convh2 = nn.Conv1d(inc, outc // 4, 7, stride=1, padding="same")
        self.convh3 = nn.Conv1d(inc, outc // 4, 5, stride=1, padding="same")
        self.convh4 = nn.Conv1d(inc, outc // 4, 3, stride=1, padding="same")
        self.bnh1 = nn.BatchNorm1d(outc // 4)
        self.bnh2 = nn.BatchNorm1d(outc // 4)
        self.bnh3 = nn.BatchNorm1d(outc // 4)
        self.bnh4 = nn.BatchNorm1d(outc // 4)

        self.shortCut = nn.Sequential(
            nn.Conv1d(inc, outc, 1, stride=1, padding="same"),
            nn.BatchNorm1d(outc),
            nn.LeakyReLU(),
        )

        self.out = nn.AvgPool1d(avgK, avgK)

    def forward(self, inputs):  # [b, c, l] => [b, c', l']
        out1 = F.leaky_relu(self.bnh1(self.convh1(inputs)))
        out2 = F.leaky_relu(self.bnh2(self.convh2(inputs)))
        out3 = F.leaky_relu(self.bnh3(self.convh3(inputs)))
        out4 = F.leaky_relu(self.bnh4(self.convh4(inputs)))
        out = torch.cat([out1, out2, out3, out4], dim=1)

        shortCut = self.shortCut(inputs)

        return self.out(out + shortCut)
    
class SABlock1DR(nn.Module):
    def __init__(self, inc, outc, avgK, minCnnKSize):
        super(SABlock1DR, self).__init__()
        # [ B , L , C]
        self.cnn = nn.ModuleList([
            nn.Sequential(
            nn.Conv1d(inc, outc // 4, k, stride=1, padding="same"),
            nn.BatchNorm1d(outc // 4),
            nn.LeakyReLU()
            )
            for k in range(minCnnKSize, minCnnKSize+2*4, 2)
        ])
        self.shortCut = nn.Sequential(
            nn.Conv1d(inc, outc, 1, stride=1, padding="same"),
            nn.BatchNorm1d(outc),
            nn.LeakyReLU(),
        )

        self.out = nn.AvgPool1d(avgK, avgK)
    def forward(self, inputs):  # [b, c, l] => [b, c', l']
        out = torch.cat([ cnn(inputs) for cnn in self.cnn ], dim=1)
        shortCut = self.shortCut(inputs)
        return self.out(out + shortCut)
    
class SABlock1DR2(nn.Module):
    """SABlock Res Construction"""
    def __init__(self, inc, outc, avgK, minCnnKSize):
        super(SABlock1DR2, self).__init__()
        mS = minCnnKSize
        self.cnnK = [(mS+4, mS+2), (mS+2, mS+0), (mS+2, mS+4), (mS+0, mS+2)]
        def GenCnn(inChannles: int, outChannles: int, cnnKernel):
            ret = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(inChannles, outChannles, k, stride=1, padding="same"),
                    nn.BatchNorm1d(outChannles),
                    nn.LeakyReLU()
                )
                for k in cnnKernel
            ])
            return ret if len(cnnKernel) > 1 else ret[0]
        self.cnn = GenCnn(inc, outc // 4, range(minCnnKSize, minCnnKSize+2*4, 2))
        self.cnn2 = GenCnn(outc, outc // 4, range(minCnnKSize, minCnnKSize+2*4, 2))
        self.shortCut = GenCnn(inc, outc, [1, ])
        self.out = nn.AvgPool1d(avgK, avgK)
    def forward(self, inputs):  # [b, c, l] => [b, c', l']
        out = torch.cat([ cnn(inputs) for cnn in self.cnn ], dim=1)
        out = torch.cat([ cnn(out) for cnn in self.cnn2 ], dim=1)
        shortCut = self.shortCut(inputs)
        return self.out(out + shortCut)
