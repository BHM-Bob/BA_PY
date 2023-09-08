'''
Date: 2023-06-06 23:35:17
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-06-07 11:14:56
FilePath: \BA_PY\mbapy\dl_torch\arch\CL\trainer.py
Description: 
'''

import time

import torch

from ...utils import *
from .builder import *


def train(train_loader, model, optimizer, epoch, args, globalsteps):
    losses = AverageMeter('Loss', ':7.5f')
    top1 = AverageMeter('Acc@1', ':5.2f')
    top5 = AverageMeter('Acc@5', ':5.2f')
    time_spent = AverageMeter('Epoch(s)', ':4.2f')
    sumBatchs = len(train_loader)
    progress = ProgressMeter(
        sumBatchs,
        [losses, top1, top5, time_spent],
        prefix="Epoch: [{}]".format(epoch))
    progress.mp = args.mp
    # train
    model.train()
    time1 = time.time()
    for i, ((seq1, seq2), _) in enumerate(train_loader):
        # compute output
        seq1 = seq1.cuda(args.gpu, non_blocking=True)
        seq2 = seq2.cuda(args.gpu, non_blocking=True)
        logits = model(seq1=seq1, seq2=seq2)
        loss = model.loss_fn(logits)
        acc = model.acc(logits, topk = (1, 5))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #viz
        losses.update(loss.item(), args.batch_size)
        top1.update(acc[0], args.batch_size)
        top5.update(acc[1], args.batch_size)
        if (i+1) % args.print_freq == 0 or (i+1) == sumBatchs:
            if (i+1) == sumBatchs:
                time_spent.update(time.time()-time1, 1)
            if epoch % args.test_freq == 0 and model.heatmap_shape is not None:
                title = f"logits | epoch {epoch:d} | globalsteps {globalsteps:d} | batchIdx {i:d}/{sumBatchs:d}"
                logits = logits[:model.heatmap_shape[0], :model.heatmap_shape[1]]
                viz.heatmap(X=logits.cpu(), win = 'logits2', opts =  dict(title = title))
            VizLine(top1.val, globalsteps, "BatchAcc", "BatchAcc", "Top1Acc", opts={'showlegend':True}) 
            VizLine(top5.val, globalsteps, "BatchAcc", "BatchAcc", "Top5Acc", opts={'showlegend':True})        
            title = f"train_loss {losses.val:4f}({losses.avg:4f}) | top1Acc {top1.val:4f}({top1.avg:4f})"
            VizLine(losses.val, globalsteps, "BatchLoss", "BatchLoss", "trainBatchLoss") 
            progress.display(i)

        globalsteps += 1
    return losses.avg, globalsteps