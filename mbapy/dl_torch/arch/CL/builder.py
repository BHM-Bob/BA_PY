'''
Date: 2023-06-06 23:35:10
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-06-07 10:54:18
FilePath: \BA_PY\mbapy\dl_torch\arch\CL\builder.py
Description: 
'''

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils import GlobalSettings, get_default_for_None
    
def stop_grad(model:nn.Module):
    for param_k in model.parameters():
        param_k.requires_grad = False  # not update by gradient
    return model

class QK_Encoder(nn.Module):
    """forward(self, x: torch.Tensor, compute_k:bool = True, update_encoder_k:bool = True, **kwargs)"""
    def __init__(self, args:GlobalSettings, model:nn.Module, encoder_k = None, **kwargs):
        super().__init__()
        self.args = args
        self.K = args.moco_k
        self.m = args.moco_m
        self.T = args.moco_t
        self.encoder_q = model
        self.encoder_k = get_default_for_None(encoder_k, stop_grad(copy.deepcopy(model)))
        
    @torch.no_grad()
    def _momentum_update_key_encoder(self, encoder_q:nn.Module, encoder_k:nn.Module):
        """Momentum update of the key encoder"""
        for param_q, param_k in zip(encoder_q.parameters(), encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        return encoder_q, encoder_k
    
    def forward(self, x: torch.Tensor, compute_k:bool = True, update_encoder_k:bool = True, **kwargs):
        z1 = self.encoder_q(x)
        # compute key features
        if compute_k:
            with torch.no_grad():  # no gradient to keys
                if update_encoder_k:
                        self._momentum_update_key_encoder(self.encoder_q, self.encoder_k)
                z1_ = self.encoder_k(x)
        else:
            z1_ = None
        return z1, z1_

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    edited from https://github.com/facebookresearch/moco/blob/main/moco/builder.py
    """
    def __init__(self, args:GlobalSettings, criterion:nn.Module,
                 model:QK_Encoder, encoder_k = None, **kwargs):
        super().__init__()
        self.K = args.moco_k
        self.m = args.moco_m
        self.T = args.moco_t
        self.model = QK_Encoder(args, model, encoder_k, **kwargs)
        # create the queue
        assert self.K % args.batch_size == 0  # for simplicity
        self.register_buffer("queue", torch.randn(self.dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.label = torch.zeros(args.batch_size, dtype=torch.long)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        #keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        # queue = [dim, K]
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr

    def forward(self, x1:torch.Tensor, x2:torch.Tensor):
        # compute query features
        z1, _ = self.model(x1, compute_k = False, update_encoder_k = True  ) # queries: NxC
        _, z2_ = self.model(x2, update_encoder_k = False) # queries: NxC
        # compute logits
        # positive logits: [N, 1],点积后求和
        # 单样本的正样本对向量相点积，最终得到一个数字，以batch形式：[b,1]
        l_pos = torch.einsum('nc,nc->n', [z1, z2_]).unsqueeze(-1)
        # negative logits: [N, K],矩阵相乘
        # 单样本的正向量之一（Q）与queue的每个样本向量点积，以batch形式：[b,K]
        l_neg = torch.einsum('nc,ck->nk', [z1, self.queue.clone().detach()])
        # logits: [N, (1+K)]
        logits = torch.cat([l_pos, l_neg], dim=1)
        # apply temperature
        logits /= self.T
        # dequeue and enqueue
        self._dequeue_and_enqueue(z2_)
        return logits
        
    def loss_fn(self, logits:tuple[torch.Tensor]):
        # z: [b, D]
        logits = F.normalize(torch.cat(logits, dim = 0), dim=-1, p=2)
        z2, z1_, z1, z2_ = logits.split(logits.shape[0]//4, dim = 0)
        return (4 - 2 * (z2*z1_.detach()).sum(dim=-1) - 2 * (z1*z2_.detach()).sum(dim=-1)).mean()
    
    @torch.no_grad()
    def acc(self, logits:tuple[torch.Tensor], topk: list[int] = (1, 5)):
        """
        logits: [b, 1+k]
        label: [b, ]
        Computes the batch accuracy over the k top predictions for the specified values of k
        """
        _, pred = logits.topk(max(topk), 1, True, True)
        correct = pred.eq(self.label.reshape(-1, 1)) #correct: [b, k]
        return [correct[:, :k].float().sum().mul_(100.0 / logits.shape[0]).item() for k in topk]
    
@torch.jit.script
def calcu_acc(mat:torch.Tensor, mat_label:torch.Tensor, topk: list[int]):
    """
    mat: [b, b], one of BYOL sub mat
    mat_label: [b, ]
    """
    _, pred = mat.topk(max(topk), 1, True, True)
    correct = pred.eq(mat_label.reshape(-1, 1)) #correct: [2*b, k]
    return [correct[:, :k].float().sum().mul_(100.0 / mat.shape[0]).item() for k in topk]

class Twins(nn.Module):
    """barlow twins"""
    def __init__(self, args:GlobalSettings, criterion:nn.Module,
                 model:nn.Module, **kwargs):
        super().__init__()
        self.args = args
        self.K = args.moco_k
        self.m = args.moco_m
        self.T = args.moco_t
        self.model = model
        self.label = torch.arange(args.batch_size).cuda('cuda')
        self.heatmap_shape = None
        self.criterion = criterion
    
    def forward(self, seq1:torch.Tensor, seq2:torch.Tensor):
        # compute query features
        z1 = self.model(seq1) # queries: bxD
        z2 = self.model(seq2) # queries: bxD
        logits = z1.matmul(z2.T) #[b, c] @ [c, b] => [b, b]
        return logits
        
    def loss_fn(self, logits:tuple[torch.Tensor]):
        # z: [b, b]
        return self.criterion(logits, self.label)
    
    @torch.no_grad()
    def acc(self, logits:torch.Tensor,
            acc_label:torch.Tensor = None, topk: list[int] = (1, 5)):
        """
        acc_label: [2*b, 1]
        Computes the accuracy over the k top predictions for the specified values of k
        """
        return calcu_acc(logits, self.label, topk)
    