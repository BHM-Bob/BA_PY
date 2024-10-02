'''
Author: BHM-Bob 2262029386@qq.com
Date: 2023-03-23 21:50:21
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2024-10-02 18:17:27
Description: Basic Blocks
'''

import math
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == '__main__':
    from mbapy.base import autoparse, get_default_for_None
    from mbapy.dl_torch import paper
else:
    from ..base import autoparse, get_default_for_None
    from . import paper


class CnnCfg:
    @autoparse
    def __init__(self, inc:int, outc:int, kernel_size:int = 3, stride:int = 1, padding:int = 1):
        self.inc:int = inc
        self.outc:int = outc
        self.kernel_size:int = kernel_size 
        self.stride:int = stride
        self.padding:int = padding
        self._str_:str = ','.join([attr+'='+str(getattr(self, attr)) for attr in vars(self)])
    def __str__(self):
        return self._str_
        
class reshape(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.shape = args
    def forward(self, x):
        return x.reshape(self.shape)
    
class permute(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.idxs = args
    def forward(self, x):
        return x.permute(self.idxs)

class ScannCore(nn.Module):
    """MHSA 单头版"""
    def __init__(self, s, way="linear", dropout=0.2):
        super(ScannCore, self).__init__()
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
        inc : int, # input channle
        group : int=1,# means how many channels are in a group to get into ScannCore
        stride : int=2,
        padding : int=1,# F.unflod的padding是两边（四周）均pad padding个0
        kernel_size : int=3,
        outway : str="linear", # linear or avg
        dropout : float=0.2,
    ):
        super(SCANN, self).__init__()
        assert inc % group == 0, r'NOT inc % group == 0'
        self.inc = inc
        self.group = group
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.patch_size = kernel_size**2

        self.SAcnn = nn.ModuleList(
            [
                ScannCore(self.patch_size, outway, dropout)
                for _ in range(inc // group)
            ]
        )
    def ScannCoreMiniForward(self, x, i):
        # x = [b, group, h, w]
        batch_size = x.shape[0]
        # t = [b, self.group*self.patch_size, patch_num]
        t = F.unfold(x, self.kernel_size, 1, self.padding, self.stride)
        b, g_ps, patch_num = t.shape
        side_patch_num = int(math.sqrt(patch_num))
        # t = [b, patch_num*self.group, self.patch_size]
        t = (
            t.reshape(batch_size, self.group, self.patch_size, patch_num)
            .permute(0, 1, 3, 2)
            .reshape(batch_size, self.group * patch_num, self.patch_size)
        )
        # t = [b,self.patch_num*self.group,1]
        t = self.SAcnn[i](t)
        # t = [b,self.group,self.side_patch_num,self.side_patch_num]
        t = (
            t.reshape( batch_size, side_patch_num, side_patch_num, self.group)
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
    def __init__(self, dim: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        #pe.requires_grad = False
        self.register_buffer('pe', pe)
    def forward(self, x):
        b, l, c = x.shape
        return self.pe.repeat(x.shape[0], 1, 1)[:,:l, :].add(x)
    
class RoPE(nn.Module):
    """
    from Meta LLAMA
    edit from https://mp.weixin.qq.com/s/LH1leSGJSloxQXPYM1wzog
    """
    def __init__(self, dim: int, max_len: int, theta: float = 10000.0) -> None:
        super().__init__()
        # 计算词向量元素两两分组之后，每组元素对应的旋转角度\theta_i
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
        t = torch.arange(max_len, device=freqs.device)
        # freqs.shape = [seq_len, dim // 2] 
        freqs = torch.outer(t, freqs).float()  # 计算m * \theta
        # 计算结果是个复数向量
        # 假设 freqs = [x, y]
        # 则 freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
        self.register_buffer('freqs_cis',
                             torch.polar(torch.ones_like(freqs), freqs))
    def forward(self, xq:torch.Tensor, xk:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # xq.shape = [batch_size, seq_len, dim]
        xq_seq_len = xq.shape[1]
        xk_seq_len = xk.shape[1]
        # xq_.shape = [batch_size, seq_len, dim // 2, 2]
        xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
        xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
        # 转为复数域
        xq_ = torch.view_as_complex(xq_)
        xk_ = torch.view_as_complex(xk_)
        # 应用旋转操作，然后将结果转回实数域
        # xq_out.shape = [batch_size, seq_len, dim]
        xq_out = torch.view_as_real(xq_ * self.freqs_cis[:xq_seq_len, :]).flatten(2)
        xk_out = torch.view_as_real(xk_ * self.freqs_cis[:xk_seq_len, :]).flatten(2)
        return xq_out.type_as(xq), xk_out.type_as(xk)

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(hid_dim, pf_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(pf_dim, hid_dim),
        )
    def forward(self, x):
        # x = [batch size, seq len, hid dim]
        return self.nn(x)

class MultiHeadAttentionLayer(nn.Module):
    """
    MultiHeadAttentionLayer
    
        - if kwargs['use_enhanced_fc_q'] and 'q_len' in kwargs and 'out_len' in kwargs, use fc_q mlp like PositionwiseFeedforwardLayer to output a tensor with out_len\n
        - if 'out_dim' in kwargs, self.fc_o = nn.Linear(hid_dim, kwargs['out_dim'])
    NOTE: 
        - energy = energy.masked_fill(mask==0, -1e10)
    """
    def __init__(self, hid_dim, n_heads, dropout, device = 'cuda', **kwargs):
        super().__init__()
        assert hid_dim % n_heads == 0, 'NOT hid_dim % n_heads == 0'
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.input_dim = kwargs.get('input_dim', hid_dim)
        self.kv_input_dim = kwargs.get('kv_input_dim', self.input_dim)
        self.fc_q = nn.Linear(self.input_dim, hid_dim)
        self.fc_k = nn.Linear(self.kv_input_dim, hid_dim)
        self.fc_v = nn.Linear(self.kv_input_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        if 'out_dim' in kwargs:
            self.fc_o = nn.Linear(hid_dim, kwargs['out_dim'])
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
    def forward(self, query, key, value, RoPE: RoPE = None,
                mask = None, pad_id = None):
        batch_size = query.shape[0]
        # Q = [batch size, query len, input dim] => [batch size, query len, hid_dim][batch size, n heads, query len, head dim]
        # K = [batch size, key len,   input dim] => [batch size, key len,   hid_dim][batch size, n heads, key len  , head dim]
        # V = [batch size, value len, input dim] => [batch size, value len, hid_dim][batch size, n heads, value len, head dim]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        # RoPE
        if RoPE is not None:
            Q, K = RoPE(Q, K)
        # reshape and permute
        # Q = [batch size, query len, hid_dim] => [batch size, n heads, query len, head dim]
        # K = [batch size, key len,   hid_dim] => [batch size, n heads, key len  , head dim]
        # V = [batch size, value len, hid_dim] => [batch size, n heads, value len, head dim]
        Q = Q.reshape(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.reshape(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.reshape(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # energy & attention: [batch size, n heads, query len, key len]
        energy = Q.matmul(K.permute(0, 1, 3, 2)).multiply(self.scale)
        if mask is not None:
            energy = energy.masked_fill(mask==0, -1e10)
        elif pad_id is not None:
            energy = energy.masked_fill(energy==pad_id, -1e10)
        attention = energy.softmax(dim=-1)
        # x = [batch size, query len, hid dim]
        x = self.dropout(attention).matmul(V).permute(0, 2, 1, 3).contiguous()\
            .reshape(batch_size, -1, self.hid_dim)
        # x = [batch size, query len, hid dim]
        return self.fc_o(x)
    
class FastMultiHeadAttentionLayer(nn.Module):
    """wrapper for FlashAttention, just import flash_attn and adjust dtype"""
    def __init__(self, hid_dim, n_heads, dropout, device = 'cuda', **kwargs):
        super().__init__()
        assert paper.bb.flash_attn_func is not None, 'mbapy::import-error: paper.bb.flash_attn_func is None'
        raise NotImplementedError
    def forward(self, query, key, value, RoPE: RoPE = None, mask = None):
        ori_type = query.dtype
        query = query.to(dtype = torch.float16)
        query = self.net(query)[0]
        return query.to(dtype = ori_type)

class OutMultiHeadAttentionLayer(MultiHeadAttentionLayer):
    """
    OutMultiHeadAttentionLayer\n
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
    def forward(self, query, key, value, RoPE: RoPE = None, mask = None):
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
    def __init__(self, q_len: None, class_num: None,
                 hid_dim: int, n_heads: int, pf_dim: int, dropout: float,
                 device = 'cuda', **kwargs):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        if not kwargs.get('do_not_ff', False):
            self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        else:
            self.positionwise_feedforward = nn.Identity()
        if kwargs.get('use_FastMHA', False):
            self.self_attention = FastMultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        elif kwargs.get('use_HydraAttention', False):
            self.self_attention = paper.bb.HydraAttention(hid_dim, **kwargs)
        elif kwargs.get('MHSA', False) and issubclass(kwargs['MHSA'], nn.Module):
            self.self_attention = kwargs['MHSA']
        else:
            self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device, **kwargs)
        self.dropout = nn.Dropout(dropout)
    def forward(self, src, k = None, v = None, RoPE: RoPE = None,
                mask = None, pad_id = None):
        # src = [batch size, src len, hid dim]
        # self attention
        k = get_default_for_None(k, src)
        v = get_default_for_None(v, src)
        _src = self.self_attention(src, k, v, RoPE = RoPE,
                                   mask = mask, pad_id = pad_id)
        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        # src = [batch size, src len, hid dim]
        # positionwise feedforward
        _src = self.positionwise_feedforward(src)
        # dropout, residual and layer norm
        # ret = [batch size, src len, hid dim]
        return self.ff_layer_norm(src + self.dropout(_src))

class OutEncoderLayer(EncoderLayer):
    def __init__(self, q_len: int, class_num: int,
                 hid_dim: int, n_heads: int, pf_dim: int, dropout: float,
                 device = 'cuda', **kwargs):
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
        assert n_layers > 0, 'n_layers < 0'
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
    
class OutEncoderLayerAvg(EncoderLayer):
    def __init__(self, q_len:int, class_num:int, hid_dim:int, n_heads:int,
                 pf_dim:int, dropout:float, device:str = 'cuda', **kwargs):
        """AvgPool1d handle [N, C, L] and output [N, C, L']"""
        super().__init__(q_len, class_num, hid_dim, n_heads, pf_dim, dropout, device, **kwargs)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        if q_len > class_num:
            assert q_len % class_num == 0, r'q_len % class_num != 0'
            self.avg = nn.Sequential(
                nn.AvgPool1d(hid_dim, 1, 0),
                reshape(-1, q_len),
                nn.AvgPool1d(int(q_len/class_num), int(q_len/class_num), 0),
            )
        elif q_len < class_num:
            assert class_num % q_len == 0, r'q_len % class_num != 0'
            ks = int(q_len * hid_dim / class_num)
            if ks == 1:
                self.avg = nn.Identity()
            else:
                self.avg = nn.AvgPool1d(ks, ks, 0)
        elif q_len == class_num:
            self.avg = nn.AvgPool1d(hid_dim, 1, 0)
        self.outc = class_num
            
    def forward(self, src):
        # src = [batch size, src len, hid dim]
        b, l, c = src.shape
        # self attention
        _src = self.self_attention(src, src, src)
        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        # src = [batch size, src len, hid dim]
        # positionwise feedforward
        _src = self.positionwise_feedforward(src)
        # dropout, residual and layer norm
        # src = [batch size, src len, hid dim] = [b, l, c]
        src = self.ff_layer_norm(src + self.dropout(_src))
        # [b, l, c] => [b, l*c'] => [b, D]
        return self.avg(src).reshape(b, self.outc)
            
class TransAvg(Trans):
    def __init__(self, q_len:int, class_num:int, hid_dim:int, n_layers:int, n_heads:int, pf_dim:int,
                 dropout:float, device:str, **kwargs):
        super().__init__(q_len, class_num, hid_dim, n_layers, n_heads, pf_dim,
                         dropout, device, OutEncoderLayerAvg, **kwargs)


class SeparableConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size, stride, padding, depth = 1):
        super(SeparableConv2d, self).__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(inc, inc, kernel_size, stride, padding, groups=inc),# depthwise 
            nn.Conv2d(inc, outc, kernel_size=1),# pointwise
            )
    def forward(self, x):
        return self.nn(x)

# TODO: need proposed or simple version?
class ResBlock(nn.Module):
    """Identity Mappings in Deep Residual Networks : proposed"""
    def __init__(self, cfg:CnnCfg):
        super(ResBlock, self).__init__()
        self.cfg = cfg
        self.nn = nn.Sequential(# full pre-activation
            SeparableConv2d(cfg.inc, cfg.outc,
                            kernel_size=cfg.kernel_size, stride=cfg.stride, padding=cfg.padding),
            nn.BatchNorm2d(cfg.outc),
            nn.ReLU(True),
            SeparableConv2d(cfg.outc, cfg.outc,
                            kernel_size=cfg.kernel_size, stride=1, padding=cfg.padding),
            nn.BatchNorm2d(cfg.outc),
            nn.ReLU(True),
        )
        self.extra = nn.Conv2d(cfg.inc, cfg.outc, kernel_size=1, stride=cfg.stride)
    def forward(self, x):  # [b,ch_in,w,h] => [b,ch_out,w/2,h/2]  (stride = 2,w and h +1 %3 ==0)
        return self.nn(x)+self.extra(x)
    
class ResBlockR(ResBlock):
    """Identity Mappings in Deep Residual Networks : exclusive gating"""
    def __init__(self, cfg:CnnCfg):
        super().__init__(cfg)
    def forward(self, x):  # [b,ch_in,w,h] => [b,ch_out,w/2,h/2]  (stride = 2,w and h +1 %3 ==0)
        t = self.extra(x)
        return t.mul(self.nn(x))+(1.-torch.sigmoid(t)).mul(self.extra(x))

class SABlock(nn.Module):
    """异形卷积核的并行，外加残差结构"""
    def __init__(self, cfg:CnnCfg):
        super().__init__()
        mS = cfg.kernel_size
        self.cnnK = [(mS+4, mS+2), (mS+2, mS+0), (mS+2, mS+4), (mS+0, mS+2)]
        def GenCnn(inChannles: int, outChannles: int, cnnKernel):
            return nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(inChannles, outChannles // 4, k, stride=1, padding="same"),
                    nn.BatchNorm2d(outChannles // 4),
                    nn.LeakyReLU(inplace=False),
                )
                for k in cnnKernel
            ])
        self.cnn1 = GenCnn(cfg.inc, cfg.outc, self.cnnK)
        self.cnn2 = GenCnn(cfg.outc, cfg.outc, self.cnnK)
        self.extra = nn.Conv2d(cfg.inc, cfg.outc, kernel_size=1, stride=1, padding="same")
    def forward(self, x):  # [b,inc,h,w] => [b,outc,h,w]
        out = torch.cat([ cnn(x) for cnn in self.cnn1 ], dim=1)
        out = torch.cat([ cnn(out) for cnn in self.cnn2 ], dim=1)
        return out + self.extra(x)
    
class SABlockR(SABlock):
    """return t.mul(out)+(1.-torch.sigmoid_(t)).mul(self.extra(x))"""
    def __init__(self, cfg:CnnCfg):
        super().__init__(cfg)
    def forward(self, x):  # [b,inc,h,w] => [b,outc,h,w]
        out = torch.cat([ cnn(x) for cnn in self.cnn1 ], dim=1)
        out = torch.cat([ cnn(out) for cnn in self.cnn2 ], dim=1)
        t = self.extra(x)
        return t.mul(out)+(1.-torch.sigmoid(t)).mul(self.extra(x))
    

class ScannBlock1d(nn.Module):
    """edited from NonLocalBlock and scann_core"""
    def __init__(self, cfg:CnnCfg, **kwargs):
        self.inc = cfg.inc
        self.hid_dim = self.inc*4 if 'hid_dim' not in kwargs else kwargs['hid_dim']
        self.outc = cfg.outc
        self.q = nn.Conv1d(self.inc, self.hid_dim, kernel_size=cfg.kernel_size, padding=cfg.padding)
        self.k = nn.Conv1d(self.inc, self.hid_dim, kernel_size=1)
        self.v = nn.Conv1d(self.inc, self.hid_dim, kernel_size=1)
        self.o = nn.Conv1d(self.hid_dim, self.outc, kernel_size=1)
    
    def forward(self, x):
        """x: [b, c, l] => [b, c', l']"""
        b, c, l = x.shape
        Q = self.q(x).permute(0, 2, 1) # Q => [b, l', hid_dim]
        K = self.k(x)                  # K => [b, hid_dim, l]
        V = self.v(x).permute(0, 2, 1) # V => [b, l, hid_dim]        
        attention = Q.matmul(K).softmax(dim=-1) # attention => [b, l', l]
        # [b, l', l] @ [b, l, hd] => [b, l', hd]
        return self.o(attention.matmul(V).permute(0, 2, 1))

def GenCnn1d(inc: int, outc: int, minCnnKSize:int):
    return nn.ModuleList([
        nn.Sequential(
            nn.Conv1d(inc, outc // 4, k, stride=1, padding="same"),
            nn.BatchNorm1d(outc // 4),
            nn.LeakyReLU(inplace=False),
        )
        for k in range(minCnnKSize, minCnnKSize+2*4, 2)
    ])
    
class SABlock1D(SABlock):
    """[b, c, l] => [b, c', l']"""
    def __init__(self, cfg:CnnCfg):
        super().__init__(cfg)
        self.cnn1 = GenCnn1d(cfg.inc, cfg.outc, cfg.kernel_size)
        # self.cnn2 = GenCnn1d(cfg.outc, cfg.outc, cfg.kernel_size)
        self.extra = nn.Conv1d(cfg.inc, cfg.outc, 1, stride = 1, padding="same")
    def forward(self, x):  # [b,inc,h,w] => [b,outc,h,w]
        out = torch.cat([ cnn(x) for cnn in self.cnn1 ], dim=1)
        return out + self.extra(x)
    
class SABlock1DR(SABlockR):
    """[b, c, l] => [b, c', l']"""
    def __init__(self, cfg:CnnCfg):
        super().__init__(cfg)
        self.cnn1 = GenCnn1d(cfg.inc, cfg.outc, cfg.kernel_size)
        self.cnn2 = GenCnn1d(cfg.outc, cfg.outc, cfg.kernel_size)
        self.extra = nn.Conv1d(cfg.inc, cfg.outc, 1, stride = 1, padding="same")