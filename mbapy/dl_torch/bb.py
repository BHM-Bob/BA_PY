'''
Author: BHM-Bob 2262029386@qq.com
Date: 2023-03-23 21:50:21
LastEditors: BHM-Bob
LastEditTime: 2023-03-24 22:11:07
Description: Basic Blocks
'''

import math
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

class reshape(nn.Module):
    def __init__(self, *args, **kwargs):
        super(reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.reshape(self.shape)

class ScannCore(nn.Module):
    """MHSA 单头版"""
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
                ScannCore(self.patch_num * self.group, self.patch_size, outway, dropout)
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
        self.fc_q = nn.Linear(q_len, class_num)
        if 'use_enhanced_fc_q' in kwargs and kwargs['use_enhanced_fc_q']:
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
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        if not 'do_not_ff' in kwargs:
            self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
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
        self.ff_layer_norm = nn.LayerNorm(class_num)
        self.fc_out = nn.Linear(hid_dim, 1)
    def forward(self, src):
        # src = [batch size, src len, hid dim]
        # self attention
        # src = [batch size, class num, hid dim]
        src = self.self_attention(src, src, src)
        # dropout, residual connection and layer norm
        # src = [batch size, class num, hid dim]
        src = self.self_attn_layer_norm(src + self.dropout(src))
        # src = [batch size, class num]
        _src = self.fc_out(src).reshape(-1, self.class_num)
        # ret = [batch size, class num]
        return self.ff_layer_norm(self.dropout(_src))

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
    
class OutEncoderLayerAvg(OutEncoderLayer):
    def __init__(self, q_len, class_num, hid_dim, n_heads, pf_dim, dropout, device, **kwargs):
        super().__init__(q_len, class_num, hid_dim, n_heads, pf_dim, dropout, device, **kwargs)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        if q_len > class_num:
            assert q_len % class_num == 0
            self.fc_out = nn.Sequential(
                nn.AvgPool1d(hid_dim, 1, 0),
                reshape(-1, q_len),
                nn.AvgPool1d(int(q_len/class_num), int(q_len/class_num), 0),
            )
        elif q_len < class_num:
            assert class_num % q_len == 0
            ks = int(q_len * hid_dim / class_num)
            self.fc_out = nn.AvgPool1d(ks, ks, 0)
        elif q_len == class_num:
            self.fc_out = nn.AvgPool1d(hid_dim, 1, 0)
class TransAvg(Trans):
    def __init__(self, q_len:int, class_num:int, hid_dim:int, n_layers:int, n_heads:int, pf_dim:int,
                 dropout:float, device:str, **kwargs):
        super().__init__(q_len, class_num, hid_dim, n_layers, n_heads, pf_dim,
                         dropout, device, OutEncoderLayerAvg, **kwargs)


class SeparableConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size, stride, padding, depth = 1):
        super(SeparableConv2d, self).__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(inc, outc*depth, kernel_size, stride, padding, groups=inc),# depthwise 
            nn.Conv2d(outc*depth, outc, kernel_size=1),# pointwise
            )
    def forward(self, x):
        return self.nn(x)

class ResBlock(nn.Module):
    """Identity Mappings in Deep Residual Networks : proposed"""
    def __init__(self, ch_in, ch_out, stride=1):
        super(ResBlock, self).__init__()
        self.nn = nn.Sequential(# full pre-activation
            nn.BatchNorm2d(ch_in),
            nn.ReLU(True),
            SeparableConv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(True),
            SeparableConv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
        )
        if ch_out != ch_in:
            self.extra = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride)
        else:
            self.extra = lambda x : x
    def forward(self, x):  # [b,ch_in,w,h] => [b,ch_out,w/2,h/2]  (stride = 2,w and h +1 %3 ==0)
        return self.nn(x)+self.extra(x)
    
class ResBlockR(ResBlock):
    """Identity Mappings in Deep Residual Networks : exclusive gating"""
    def __init__(self, ch_in, ch_out, stride=1):
        super().__init__(ch_in, ch_out, stride)
        self.extra = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride)
    def forward(self, x):  # [b,ch_in,w,h] => [b,ch_out,w/2,h/2]  (stride = 2,w and h +1 %3 ==0)
        t = self.extra(x)
        return t.mul(self.nn(x))+(1.-torch.sigmoid_(t)).mul(self.extra(x))

class SABlock(nn.Module):
    """异形卷积核的并行，外加残差结构"""
    def __init__(self, inc, outc, minCnnKSize = 3):
        super().__init__()
        mS = minCnnKSize
        self.cnnK = [(mS+4, mS+2), (mS+2, mS+0), (mS+2, mS+4), (mS+0, mS+2)]
        def GenCnn(inChannles: int, outChannles: int, cnnKernel):
            return nn.ModuleList([
                nn.Sequential(
                    nn.BatchNorm2d(inChannles),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(inChannles, outChannles, k,
                              stride=1, padding="same"),
                )
                for k in cnnKernel
            ])
        self.cnn1 = GenCnn(inc, outc // 4, self.cnnK)
        self.cnn2 = GenCnn(outc, outc // 4, self.cnnK)
        if inc != outc:
            self.extra = nn.Conv2d(inc, outc, kernel_size=1, stride=1, padding="same")
        else:
            self.extra = lambda x : x
    def forward(self, x):  # [b,inc,h,w] => [b,outc,h,w]
        out = torch.cat([ cnn(x) for cnn in self.cnn1 ], dim=1)
        out = torch.cat([ cnn(out) for cnn in self.cnn2 ], dim=1)
        return out + self.extra(x)
    
class SABlockR(SABlock):
    def __init__(self, inc, outc, minCnnKSize = 3):
        super().__init__(inc, outc, minCnnKSize)
        self.extra = nn.Conv2d(inc, outc, kernel_size=1, stride=1, padding="same")
    def forward(self, x):  # [b,inc,h,w] => [b,outc,h,w]
        out = torch.cat([ cnn(x) for cnn in self.cnn1 ], dim=1)
        out = torch.cat([ cnn(out) for cnn in self.cnn2 ], dim=1)
        t = self.extra(x)
        return t.mul(out)+(1.-torch.sigmoid_(t)).mul(self.extra(x))
    
class SABlock1D(SABlock):
    """[b, c, l] => [b, c', l']"""
    def __init__(self, inc, outc, minCnnKSize = 1):
        super().__init__(inc, outc)
        def GenCnn(inc: int, outc: int, minCnnKSize:int):
            return nn.ModuleList([
                nn.Sequential(
                    nn.BatchNorm1d(inc),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv1d(inc, outc // 4, k, stride=1, padding="same"),
                )
                for k in range(minCnnKSize, minCnnKSize+2*4, 2)
            ])
        self.cnn1 = GenCnn(inc, outc, minCnnKSize)
        self.cnn2 = GenCnn(outc, outc, minCnnKSize)
        if inc != outc:
            self.extra = nn.Conv1d(inc, outc, kernel_size=1, stride=1, padding="same")
        else:
            self.extra = lambda x : x
    
class SABlock1DR(SABlockR):
    """[b, c, l] => [b, c', l']"""
    def __init__(self, inc, outc, minCnnKSize = 3):
        super().__init__(inc, outc)
        def GenCnn(inc: int, outc: int, minCnnKSize:int):
            return nn.ModuleList([
                nn.Sequential(
                    nn.BatchNorm1d(inc),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv1d(inc, outc // 4, k, stride=1, padding="same"),
                )
                for k in range(minCnnKSize, minCnnKSize+2*4, 2)
            ])
        self.cnn1 = GenCnn(inc, outc, minCnnKSize)
        self.cnn2 = GenCnn(outc, outc, minCnnKSize)
        self.extra = nn.Conv1d(inc, outc, kernel_size=1, stride=1, padding="same")