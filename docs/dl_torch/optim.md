# mbapy.dl_torch.optim

This module provides a learning rate scheduler for PyTorch optimizers.  

### \_ConsineDown->float
**Calculate learning rate using ConsineDown method**

#### Params
- lr_0 (float): initial learning rate
- now_epoch (int): current epoch number
- T_0 (int): minimum T
- sum_epoch (int): total number of epochs

#### Returns
- lr (float): calculated learning rate

#### Notes
This function calculates the learning rate using the ConsineDown method, which is a variant of the cosine annealing learning rate schedule.

#### Example
```python
lr = _ConsineDown(lr_0=0.1, now_epoch=10, T_0=5, sum_epoch=100)
```

### \_ConsineAnnealing->float
**Calculate learning rate using ConsineAnnealing method**

#### Params
- lr_0 (float): initial learning rate
- now_epoch (int): current epoch number
- T_0 (int): minimum T
- sum_epoch (int): total number of epochs

#### Returns
- lr (float): calculated learning rate

#### Notes
This function calculates the learning rate using the ConsineAnnealing method, which is a variant of the cosine annealing learning rate schedule.

#### Example
```python
lr = _ConsineAnnealing(lr_0=0.1, now_epoch=10, T_0=5, sum_epoch=100)
```

### \_DownConsineAnnealing->float
**Calculate learning rate using DownConsineAnnealing method**

#### Params
- lr_0 (float): initial learning rate
- now_epoch (int): current epoch number
- T_0 (int): minimum T
- sum_epoch (int): total number of epochs

#### Returns
- lr (float): calculated learning rate

#### Notes
This function calculates the learning rate using the DownConsineAnnealing method, which is a variant of the cosine annealing learning rate schedule.

#### Example
```python
lr = _DownConsineAnnealing(lr_0=0.1, now_epoch=10, T_0=5, sum_epoch=100)
```

### \_DownScaleConsineAnnealing->float
**Calculate learning rate using DownScaleConsineAnnealing method**

#### Params
- lr_0 (float): initial learning rate
- now_epoch (int): current epoch number
- T_0 (int): minimum T
- sum_epoch (int): total number of epochs

#### Returns
- lr (float): calculated learning rate

#### Notes
This function calculates the learning rate using the DownScaleConsineAnnealing method, which is a variant of the cosine annealing learning rate schedule.

#### Example
```python
lr = _DownScaleConsineAnnealing(lr_0=0.1, now_epoch=10, T_0=5, sum_epoch=100)
```

### \_DownScaleRConsineAnnealing->float
**Calculate learning rate using DownScaleRConsineAnnealing method**

#### Params
- lr_0 (float): initial learning rate
- now_epoch (int): current epoch number
- T_0 (int): minimum T
- sum_epoch (int): total number of epochs

#### Returns
- lr (float): calculated learning rate

#### Notes
This function calculates the learning rate using the DownScaleRConsineAnnealing method, which is a variant of the cosine annealing learning rate schedule.

#### Example
```python
lr = _DownScaleRConsineAnnealing(lr_0=0.1, now_epoch=10, T_0=5, sum_epoch=100)
```

### LrScheduler
**A learning rate scheduler for optimizing learning rates during training**

#### Attrs
- optimizer (torch.optim.Optimizer): Wrapped optimizer
- lr_0 (float): initial learning rate
- now_epoch (int): current epoch number
- T_0 (int): minimum T
- sum_epoch (int): total number of epochs
- method (str): method for calculating learning rate

#### Methods
- \_\_init\_\_(optimizer:torch.optim.Optimizer, lr_0:float, now_epoch:int=0, T_0:int=100, sum_epoch:int=5000, method:str='ConsineDown'): Initializes the learning rate scheduler with the specified parameters and method.
- add_epoch(n:int): Adds the specified number of epochs to the total number of epochs.
- edited_ext_epoch(n:int): Updates the total number of epochs to the sum of the current epoch and the specified number of epochs.
- step(epoch:float): Updates the learning rate based on the current epoch and returns the updated learning rate.

#### Notes
The LrScheduler class provides functionality to dynamically adjust the learning rate during training using various methods such as ConsineDown, ConsineAnnealing, DownConsineAnnealing, DownScaleConsineAnnealing, and DownScaleRConsineAnnealing.

#### Example
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = LrScheduler(optimizer, lr_0=0.1, now_epoch=0, T_0=100, sum_epoch=5000, method='ConsineDown')
for epoch in range(20):
    for i, sample in enumerate(dataloader):
        ...
        outputs = net(inputs)
        ...
        optimizer.step()
        scheduler.step(epoch + i / iters)
```