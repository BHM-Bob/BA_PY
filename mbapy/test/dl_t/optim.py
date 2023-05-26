'''
Date: 2023-05-26 19:30:28
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-05-26 19:48:52
FilePath: /BA_PY/mbapy/test/dl_t/optim.py
Description: 
'''
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

from mbapy.dl_torch.optim import _str2scheduleF, LrScheduler

lr = 0.01
optimizer = optim.SGD(nn.Linear(1, 1).parameters(), lr)

for method in _str2scheduleF.keys():
    scheduler = LrScheduler(optimizer, lr, 0, method=method)
    lrs = [scheduler.step(epoch) for epoch in range(scheduler.sum_epoch)]
    plt.plot(list(range(scheduler.sum_epoch)), lrs)
    plt.title(method)
    plt.show()