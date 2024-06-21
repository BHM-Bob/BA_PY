'''
Date: 2023-06-06 23:35:10
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2024-06-21 15:13:23
FilePath: \BA_PY\mbapy\dl_torch\arch\CL\builder.py
Description: 
'''

import copy
from typing import List, Tuple

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
        
    def loss_fn(self, logits:Tuple[torch.Tensor]):
        # z: [b, D]
        logits = F.normalize(torch.cat(logits, dim = 0), dim=-1, p=2)
        z2, z1_, z1, z2_ = logits.split(logits.shape[0]//4, dim = 0)
        return (4 - 2 * (z2*z1_.detach()).sum(dim=-1) - 2 * (z1*z2_.detach()).sum(dim=-1)).mean()
    
    @torch.no_grad()
    def acc(self, logits:Tuple[torch.Tensor], topk: List[int] = (1, 5)):
        """
        logits: [b, 1+k]
        label: [b, ]
        Computes the batch accuracy over the k top predictions for the specified values of k
        """
        _, pred = logits.topk(max(topk), 1, True, True)
        correct = pred.eq(self.label.reshape(-1, 1)) #correct: [b, k]
        return [correct[:, :k].float().sum().mul_(100.0 / logits.shape[0]).item() for k in topk]
    
@torch.jit.script
def calcu_acc(mat:torch.Tensor, mat_label:torch.Tensor, topk: List[int]):
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
        
    def loss_fn(self, logits:Tuple[torch.Tensor]):
        # z: [b, b]
        return self.criterion(logits, self.label)
    
    @torch.no_grad()
    def acc(self, logits:torch.Tensor,
            acc_label:torch.Tensor = None, topk: List[int] = (1, 5)):
        """
        acc_label: [2*b, 1]
        Computes the accuracy over the k top predictions for the specified values of k
        """
        return calcu_acc(logits, self.label, topk)

class PTwins(nn.Module):
    """
    由于MergeMultiLoads已经使用了MergeLoads, 当作projection, 故仅使args.mlp即可作为prediction
    edited from https://github.com/lucidrains/byol-pytorch/blob/master/byol_pytorch/byol_pytorch.py
    """
    def __init__(self, args:GlobalSettings, criterion:nn.Module,
                 model:nn.Module, **kwargs):
        super().__init__()
        self.args = args
        self.K = args.moco_k
        self.m = args.moco_m
        self.T = args.moco_t
        self.model = model
        self.label = torch.arange(args.batch_size).repeat(2).view(-1).cuda('cuda')
        self.acc_label = torch.arange(args.batch_size).cuda(args.gpu, non_blocking=True) # [b, ]
        self.heatmap_shape = None
        self.criterion = criterion
    
    def forward(self, seq1:torch.Tensor, seq2:torch.Tensor):
        # compute query features
        z1, z1_ = self.model(seq1, update_encoder_k = True ) # queries: bxD
        z2, z2_ = self.model(seq2, update_encoder_k = False) # queries: bxD
        return z2, z1_, z1, z2_
        
    def loss_fn(self, logits:Tuple[torch.Tensor]):
        # z: [b, D]
        logits = F.normalize(torch.cat(logits, dim = 0), dim=-1, p=2)
        z2, z1_, z1, z2_ = logits.split(logits.shape[0]//4, dim = 0)
        return (4 - 2 * (z2*z1_.detach_()).sum(dim=-1) - 2 * (z1*z2_.detach_()).sum(dim=-1)).mean()
    
    @torch.no_grad()
    def acc(self, logits:Tuple[torch.Tensor],
            acc_label:torch.Tensor = None, topk: List[int] = (1, 5)):
        """
        acc_label: [2*b, 1]
        Computes the accuracy over the k top predictions for the specified values of k"""
        z2, z1_, z1, z2_ = logits
        acc1 = calcu_acc(z2.matmul(z1_.T), self.acc_label, topk)
        acc2 = calcu_acc(z1.matmul(z2_.T), self.acc_label, topk)
        return [sum(a)/len(a) for a in zip(acc1, acc2)]

class PTwinsQ(PTwins):
    """
    和SimCLR有些相似, 使用batch内正负样例, 但是使用了动量编码器
    """
    def __init__(self, args:GlobalSettings, criterion:nn.Module,
                 model:nn.Module, **kwargs):
        super().__init__(args, model, criterion)
        # create the queue
        assert self.K % (args.batch_size) == 0  # for simplicity
        self.register_buffer("queue", torch.randn(self.args.out_shape[-1], self.K)) # [c, k]
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        # label
        diag = torch.diag_embed(torch.ones(args.batch_size).cuda(args.gpu, non_blocking=True))
        self.label = torch.cat([torch.cat([diag, diag], dim = 0),
                                torch.zeros(2*args.batch_size, self.K).cuda(args.gpu, non_blocking=True)], dim=1)
        self.acc_label = torch.arange(args.batch_size).cuda(args.gpu, non_blocking=True) # [b, ]
        self.heatmap_shape = [2*args.batch_size, args.batch_size]

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """在移动指针前在指针位更新队列"""
        # keys: [2*N, C]
        spanSize = keys.shape[0]
        ptr = int(self.queue_ptr)
        # replace the keys at ptr (dequeue and enqueue) queue = [C, kN]
        self.queue[:, ptr:ptr + spanSize] = keys.T
        # move pointer
        newPtr = (ptr + spanSize) % self.K
        self.queue_ptr[0] = newPtr
        return ptr
        
    def forward(self, seq1:torch.Tensor, seq2:torch.Tensor):
        # compute query features
        z1, z1_ = self.model(seq1, update_encoder_k = True ) # queries: bxC
        z2, z2_ = self.model(seq2, update_encoder_k = False) # queries: bxC
        # gather features
        mat_lk_queue = torch.cat([z1, z2], dim = 0).matmul(self.queue.clone().detach()) #[2*b, c] @ [c, k] => [2*b, k]
        mat_z1_z2o = z1.matmul(z2_.T) #[b, c] @ [c, b] => [b, b]
        mat_z2_z1o = z2.matmul(z1_.T) #[b, c] @ [c, b] => [b, b]
        logits = torch.cat([
            torch.cat([mat_z1_z2o, mat_z2_z1o], dim = 0), # [2*b, b]
            mat_lk_queue], dim = 1) # [2*b, k]
        # dequeue and enqueue
        self._dequeue_and_enqueue(torch.cat([z1_,z2_], dim = 0)) # [2b, c]
        return logits # [2*b, b+k]
    
    def loss_fn(self, logits:torch.Tensor):
        return self.criterion(logits, self.label)
    
    @torch.no_grad()
    def acc(self, logits:torch.Tensor,
            acc_label:torch.Tensor = None, topk: List[int] = (1, 5)):
        """
        Computes the accuracy over the k top predictions for the specified values of k\n
        logits: [2*b, b+k]
        acc_label: [b, ]
        """
        sum_samples = int(logits.shape[0]//2)
        acc1 = calcu_acc(logits[:sum_samples, :], self.acc_label, topk)
        acc2 = calcu_acc(logits[sum_samples:, :], self.acc_label, topk)
        return [sum(a)/len(a) for a in zip(acc1, acc2)]

class PTwinsM(PTwins):
    """
    和SimCLR有些相似, 使用batch内正负样例, 但是使用了动量编码器
    """
    def __init__(self, args:GlobalSettings, criterion:nn.Module,
                 model:nn.Module, **kwargs):
        super().__init__(args, model, criterion)
        diag = torch.diag_embed(torch.ones(args.batch_size).cuda(args.gpu, non_blocking=True))
        label = torch.cat([diag,diag],dim = -1)
        self.label = torch.cat([label,label],dim = 0)
        self.acc_label = torch.arange(args.batch_size).cuda(args.gpu, non_blocking=True) # [b, ]
        self.heatmap_shape = [2*args.batch_size, 2*args.batch_size]
        
    def forward(self, seq1:torch.Tensor, seq2:torch.Tensor):       
        # compute query features
        z1, z1_ = self.model(seq1, update_encoder_k = True ) # queries: bxC
        z2, z2_ = self.model(seq2, update_encoder_k = False) # queries: bxC 
        l1 = torch.cat([z1,z1_],dim = 0) # [2*b, C]
        l2 = torch.cat([z2,z2_],dim = 0) # [2*b, C]
        logits = torch.matmul(l1, l2.T) # [2*b, 2*b]
        return logits # [2*b, 2*b]
    
    def loss_fn(self, logits:torch.Tensor):
        return self.criterion(logits, self.label)
    
    @torch.no_grad()
    def acc(self, logits:torch.Tensor,
            acc_label:torch.Tensor = None, topk: List[int] = (1, 5)):
        """
        Computes the accuracy over the k top predictions for the specified values of k\n
        logits: [2*b, 2*b]
        acc_label: [2*b, 2]
        """
        sum_samples = int(logits.shape[0]//2)
        acc1 = calcu_acc(logits[:sum_samples, :sum_samples], self.acc_label, topk)
        acc2 = calcu_acc(logits[:sum_samples, sum_samples:], self.acc_label, topk)
        acc3 = calcu_acc(logits[sum_samples:, :sum_samples], self.acc_label, topk)
        acc4 = calcu_acc(logits[sum_samples:, sum_samples:], self.acc_label, topk)
        return [sum(a)/len(a) for a in zip(acc1, acc2, acc3, acc4)]

class PTwinsMQ(PTwinsM):
    """
    和SimCLR有些相似, 使用batch内正负样例, 但是使用了动量编码器, 
    也和MoCo相似, 其不断更新一个相似度矩阵队列
    """
    def __init__(self, args:GlobalSettings, criterion:nn.Module,
                 model:nn.Module, **kwargs):
        super().__init__(args, model, criterion)
        # create the queue
        assert self.K % (2*args.batch_size) == 0  # for simplicity
        self.register_buffer("queue", torch.randn(self.args.out_shape[-1], self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        diag = torch.diag_embed(torch.ones(args.batch_size).cuda(args.gpu, non_blocking=True))
        diag = diag.repeat(2,2)
        label_lank = torch.zeros(size = [2*args.batch_size, self.K]).cuda(args.gpu, non_blocking=True)
        self.label = torch.cat([diag, label_lank], dim = 1)
        self.acc_label = torch.arange(args.batch_size).cuda(args.gpu, non_blocking=True)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """在移动指针前在指针位更新队列"""
        # keys: [2*N, C]
        spanSize = keys.shape[0]
        ptr = int(self.queue_ptr)
        # replace the keys at ptr (dequeue and enqueue) queue = [C, kN]
        self.queue[:, ptr:ptr + spanSize] = keys.T
        # move pointer
        newPtr = (ptr + spanSize) % self.K
        self.queue_ptr[0] = newPtr
        return ptr
    
    def forward(self, seq1:torch.Tensor, seq2:torch.Tensor):
        # compute query features
        z1, z1_ = self.model(seq1, update_encoder_k = True ) # queries: NxC
        z2, z2_ = self.model(seq2, update_encoder_k = False) # queries: NxC   
        # gather features
        lK = torch.cat([z1_,z2_],dim = 0) # queries: [2*N, C]
        # lQ: [2*N, C]   lQ.T: [C, 2*N]   queue: [C, K]   lQue: [C, 2*N+K]
        # logits: [2*N, 2*N+K]
        logits = torch.cat([z1, z2],dim = 0).matmul(torch.cat([lK.T, self.queue], dim = 1))
        # dequeue and enqueue
        logitsPtr = self._dequeue_and_enqueue(lK)
        return logits
    
    def loss_fn(self, logits:torch.Tensor):
        return self.criterion(logits, self.label)
    
    @torch.no_grad()
    def acc(self, logits:torch.Tensor,
            acc_label:torch.Tensor = None, topk: List[int] = (1, 5)):
        """
        Computes the accuracy over the k top predictions for the specified values of k
        # logits: [2N, 2N+k],  labelAcc: [4N, 2]
        """
        sum_samples = int(logits.shape[0]//2)
        acc1 = calcu_acc(logits[:sum_samples, :sum_samples], self.acc_label, topk)
        acc2 = calcu_acc(logits[:sum_samples, sum_samples:], self.acc_label, topk)
        acc3 = calcu_acc(logits[sum_samples:, :sum_samples], self.acc_label, topk)
        acc4 = calcu_acc(logits[sum_samples:, sum_samples:], self.acc_label, topk)
        return [sum(a)/len(a) for a in zip(acc1, acc2, acc3, acc4)]
