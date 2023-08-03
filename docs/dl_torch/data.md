# mbapy.dl_torch.data

This module provides classes and functions for data loading and preprocessing.  

## Functions

### denormalize_img(x: torch.Tensor, mean: List[float] = [0.485, 0.456, 0.406], std: List[float] = [0.229, 0.224, 0.225]) -> torch.Tensor

Denormalizes an image given its pixel values, mean, and standard deviation.  

Parameters:  
- x (torch.Tensor): The tensor of pixel values of the image.  
- mean (List[float], optional): The mean pixel value for each channel. Defaults to [0.485, 0.456, 0.406].  
- std (List[float], optional): The standard deviation of pixel values for each channel. Defaults to [0.229, 0.224, 0.225].  

Returns:  
- torch.Tensor: The denormalized image tensor.  

Example:  
```python
x = torch.tensor([0.485, 0.456, 0.406])
denormalize_img(x)
```

## Classes

### RandomSeqGenerator

A class that generates random sequences and labels from a given dataset.  

Methods:  
- __init__(self, seqs: torch.Tensor, labels: torch.Tensor, args: GlobalSettings, multiLoads: int = 6, maskRatio: float = 0.3, device: str = 'cpu'): Initializes the RandomSeqGenerator object with the given arguments.  
- __call__(self, idx: int): Generates random sequences and labels.  

Example:  
```python
seqs = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
labels = torch.tensor([0, 1, 2])
args = GlobalSettings()
generator = RandomSeqGenerator(seqs, labels, args)
generator(1)
```

### SubDataSet

A class that represents a subset of a dataset.  

Methods:  
- __init__(self, args: GlobalSettings, x: list, y: list, x_transformer: list = None, y_transformer: list = None): Initializes the SubDataSet object with the given arguments.  
- __getitem__(self, idx: int): Gets the item at the specified index.  
- __len__(self): Returns the length of the dataset.  

Example:  
```python
args = GlobalSettings()
x = [1, 2, 3]
y = [4, 5, 6]
dataset = SubDataSet(args, x, y)
dataset[1]
```

### SubDataSetR

A class that represents a subset of a dataset with additional transformer functions.  

Methods:  
- __init__(self, args: GlobalSettings, x: list, y: list, x_transformer: list = None, y_transformer: list = None): Initializes the SubDataSetR object with the given arguments.  
- __getitem__(self, idx: int): Gets the item at the specified index.  
- __len__(self): Returns the length of the dataset.  

Example:  
```python
args = GlobalSettings()
x = [1, 2, 3]
y = [4, 5, 6]
dataset = SubDataSetR(args, x, y)
dataset[1]
```

### DataSetRAM

A class that loads data into RAM and distributes it into DataLoader objects.  

Methods:  
- __init__(self, args: GlobalSettings, load_part: str = 'pre', device: str = 'cpu', x = None, y = None, x_transfer_origin = None, y_transfer_origin = None, x_transfer_gather = None, y_transfer_gather = None): Initializes the DataSetRAM object with the given arguments.  
- split(self, divide: list[float], x_transformer: list = None, y_transformer: list = None, dataset = SubDataSet): Splits the dataset into multiple DataLoader objects.  

Example:  
```python
args = GlobalSettings()
x = [1, 2, 3]
y = [4, 5, 6]
dataset = DataSetRAM(args, x=x, y=y)
dataset.split([0, 0.7, 0.9, 1])
```
