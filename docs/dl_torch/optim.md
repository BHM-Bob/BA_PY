# mbapy.dl_torch.optim

This module provides a learning rate scheduler for PyTorch optimizers.  

## Functions

### _ConsineDown(lr_0: float, now_epoch: int, T_0: int, sum_epoch: int) -> float

Calculate the learning rate using the ConsineDown method.  

Parameters:  
- lr_0 (float): The initial learning rate.  
- now_epoch (int): The current epoch.  
- T_0 (int): The minimum T.  
- sum_epoch (int): The total number of epochs.  

Returns:  
- float: The calculated learning rate.  

### _ConsineAnnealing(lr_0: float, now_epoch: int, T_0: int, sum_epoch: int) -> float

Calculate the learning rate using the ConsineAnnealing method.  

Parameters:  
- lr_0 (float): The initial learning rate.  
- now_epoch (int): The current epoch.  
- T_0 (int): The minimum T.  
- sum_epoch (int): The total number of epochs.  

Returns:  
- float: The calculated learning rate.  

### _DownConsineAnnealing(lr_0: float, now_epoch: int, T_0: int, sum_epoch: int) -> float

Calculate the learning rate using the DownConsineAnnealing method.  

Parameters:  
- lr_0 (float): The initial learning rate.  
- now_epoch (int): The current epoch.  
- T_0 (int): The minimum T.  
- sum_epoch (int): The total number of epochs.  

Returns:  
- float: The calculated learning rate.  

### _DownScaleConsineAnnealing(lr_0: float, now_epoch: int, T_0: int, sum_epoch: int) -> float

Calculate the learning rate using the DownScaleConsineAnnealing method.  

Parameters:  
- lr_0 (float): The initial learning rate.  
- now_epoch (int): The current epoch.  
- T_0 (int): The minimum T.  
- sum_epoch (int): The total number of epochs.  

Returns:  
- float: The calculated learning rate.  

### _DownScaleRConsineAnnealing(lr_0: float, now_epoch: int, T_0: int, sum_epoch: int) -> float

Calculate the learning rate using the DownScaleRConsineAnnealing method.  

Parameters:  
- lr_0 (float): The initial learning rate.  
- now_epoch (int): The current epoch.  
- T_0 (int): The minimum T.  
- sum_epoch (int): The total number of epochs.  

Returns:  
- float: The calculated learning rate.  

## Classes

### LrScheduler

A class that represents a learning rate scheduler.  

Methods:  
- __init__(self, optimizer: torch.optim.Optimizer, lr_0: float, now_epoch: int = 0, T_0: int = 100, sum_epoch: int = 5000, method: str = '_ConsineDown'): Initializes the LrScheduler object with the given parameters.  
- add_epoch(self, n: int): Adds the specified number of epochs to the total number of epochs.  
- edited_ext_epoch(self, n: int): Edits the total number of epochs by adding the specified number of epochs to the current epoch.  
- step(self, epoch: float) -> float: Updates the learning rate based on the current epoch.  

Example:  
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
scheduler = LrScheduler(optimizer, lr_0=0.1, now_epoch=0, T_0=100, sum_epoch=5000, method='ConsineDown')
for epoch in range(20):  
    for i, sample in enumerate(dataloader):  
        ...  
        optimizer.step()
        scheduler.step(epoch + i / iters)
```

## Constants

### _str2scheduleF

A dictionary that maps the method names to the corresponding learning rate calculation functions.  

Example:  
```python
_str2scheduleF = {
    'ConsineDown': _ConsineDown,
    'ConsineAnnealing': _ConsineAnnealing,
    'DownConsineAnnealing': _DownConsineAnnealing,
    'DownScaleConsineAnnealing': _DownScaleConsineAnnealing,
    'DownScaleRConsineAnnealing': _DownScaleRConsineAnnealing,
}
```