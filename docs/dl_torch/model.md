# mbapy.dl_torch.m

This module provides classes for the MA TT (Multi-Attention Time-Transformer) model.  

## Classes

### TransCfg

A class that represents the configuration for the transformer layers.  

Parameters:  
- hid_dim (int): The hidden dimension size.  
- pf_dim (Optional[int]): The position-wise feedforward dimension size. Defaults to None.  
- n_heads (int): The number of attention heads. Defaults to 8.  
- n_layers (int): The number of transformer layers. Defaults to 3.  
- dropout (float): The dropout rate. Defaults to 0.3.  
- trans_layer (str): The type of transformer layer to use. Defaults to 'EncoderLayer'.  
- out_layer (Optional[str]): The type of output layer to use. Defaults to None.  
- q_len (int): The length of the query sequence. Defaults to -1.  
- class_num (int): The number of classes. Defaults to -1.  
- kwargs (Dict[str, Union[int, str, bool]]): Additional keyword arguments. Defaults to {}.  

Methods:  
- __init__(self, hid_dim:int, pf_dim:Optional[int] = None, n_heads:int = 8, n_layers:int = 3, dropout:float = 0.3, trans_layer:str = 'EncoderLayer', out_layer:Optional[str] = None, q_len:int = -1, class_num:int = -1, **kwargs): Initializes the TransCfg object with the given arguments.  
- __str__(self): Returns a string representation of the TransCfg object.  
- toDict(self): Converts the TransCfg object to a dictionary.  
- gen(self, layer:str = None, **kwargs): Generates a transformer-like layer using the configuration.  

Example:  
```python
cfg = TransCfg(hid_dim=512, n_heads=8, n_layers=6)
print(cfg)
```

### LayerCfg

A class that represents the configuration for each layer in the MA TT model.  

Parameters:  
- inc (int): The input channel size.  
- outc (int): The output channel size.  
- kernel_size (int): The kernel size.  
- stride (int): The stride size.  
- layer (str): The type of layer to use.  
- sa_layer (Optional[str]): The type of self-attention layer to use. Defaults to None.  
- trans_layer (Optional[str]): The type of transformer layer to use. Defaults to None.  
- avg_size (int): The size of the average pooling. Defaults to -1.  
- trans_cfg (Optional[TransCfg]): The configuration for the transformer layers. Defaults to None.  
- use_SA (bool): Whether to use self-attention. Defaults to False.  
- use_trans (bool): Whether to use transformer layers. Defaults to False.  

Methods:  
- __init__(self, inc:int, outc:int, kernel_size:int, stride:int, layer:str, sa_layer:Optional[str] = None, trans_layer:Optional[str] = None, avg_size:int = -1, trans_cfg:Optional[TransCfg] = None, use_SA:bool = False, use_trans:bool = False): Initializes the LayerCfg object with the given arguments.  
- __str__(self): Returns a string representation of the LayerCfg object.  

Example:  
```python
cfg = LayerCfg(3, 64, 3, 1, 'Conv2d')
print(cfg)
```

### COneDLayer

A class that represents a convolutional layer in the MA TT model.  

Parameters:  
- cfg (LayerCfg): The configuration for the layer.  
- device (str): The device to use. Defaults to 'cuda'.  

Methods:  
- __init__(self, cfg:LayerCfg, device = 'cuda', **kwargs): Initializes the COneDLayer object with the given configuration and device.  
- forward(self, x): Performs a forward pass of the COneDLayer.  

Example:  
```python
layer = COneDLayer(cfg)
output = layer(x)
```

### MAlayer

A class that represents a multi-attention layer in the MA TT model.  

Parameters:  
- cfg (LayerCfg): The configuration for the layer.  

Methods:  
- __init__(self, cfg:LayerCfg, **kwargs): Initializes the MAlayer object with the given configuration.  
- forward(self, x): Performs a forward pass of the MAlayer.  

Example:  
```python
layer = MAlayer(cfg)
output = layer(x)
```

### MAvlayer

A class that represents a multi-attention layer with average pooling in the MA TT model.  

Parameters:  
- cfg (LayerCfg): The configuration for the layer.  

Methods:  
- __init__(self, cfg:LayerCfg, **kwargs): Initializes the MAvlayer object with the given configuration.  
- forward(self, x): Performs a forward pass of the MAvlayer.  

Example:  
```python
layer = MAvlayer(cfg)
output = layer(x)
```

### MATTPE

A class that represents the MA TT model with positional encoding.  

Parameters:  
- args (GlobalSettings): The global settings for the model.  
- cfg (list[LayerCfg]): The list of layer configurations.  
- layer (MAvlayer): The type of layer to use.  
- tail_trans_cfg (TransCfg): The configuration for the transformer layers. Defaults to None.  

Methods:  
- __init__(self, args: GlobalSettings, cfg:list[LayerCfg], layer:MAvlayer, tail_trans_cfg:TransCfg = None, **kwargs): Initializes the MATTPE object with the given arguments.  
- forward(self, x): Performs a forward pass of the MATTPE model.  

Example:  
```python
model = MATTPE(args, cfg, layer)
output = model(x)
```

### SCANNTTP

A class that represents the MA TT model with SCANN layer.  

Parameters:  
- args (GlobalSettings): The global settings for the model.  
- cfg (list[LayerCfg]): The list of layer configurations.  
- layer (SCANlayer): The type of layer to use.  
- tail_trans_cfg (TransCfg): The configuration for the transformer layers. Defaults to None.  

Methods:  
- __init__(self, args: GlobalSettings, cfg:list[LayerCfg], layer:SCANlayer, tail_trans_cfg:TransCfg = None, **kwargs): Initializes the SCANNTTP object with the given arguments.  
- forward(self, x): Performs a forward pass of the SCANNTTP model.  

Example:  
```python
model = SCANNTTP(args, cfg, layer)
output = model(x)
```

### MATTP_ViT

A class that represents the MA TT model with ViT-like structure.  

Parameters:  
- args (GlobalSettings): The global settings for the model.  
- cfg (list[LayerCfg]): The list of layer configurations.  
- layer (MAvlayer): The type of layer to use.  
- tail_trans_cfg (TransCfg): The configuration for the transformer layers. Defaults to None.  

Methods:  
- __init__(self, args: GlobalSettings, cfg:list[LayerCfg], layer:MAvlayer, tail_trans_cfg:TransCfg = None, **kwargs): Initializes the MATTP_ViT object with the given arguments.  
- forward(self, x): Performs a forward pass of the MATTP_ViT model.  

Example:  
```python
model = MATTP_ViT(args, cfg, layer)
output = model(x)
```

### MATTP

A class that represents the MA TT model.  

Parameters:  
- args (GlobalSettings): The global settings for the model.  
- cfg (list[LayerCfg]): The list of layer configurations.  
- layer (MAlayer): The type of layer to use.  
- tail_trans_cfg (TransCfg): The configuration for the transformer layers. Defaults to None.  

Methods:  
- __init__(self, args: GlobalSettings, cfg:list[LayerCfg], layer:MAlayer, tail_trans_cfg:TransCfg = None, **kwargs): Initializes the MATTP object with the given arguments.  
- forward(self, x): Performs a forward pass of the MATTP model.  

Example:  
```python
model = MATTP(args, cfg, layer)
output = model(x)
```

### COneD

A class that represents the MA TT model with 1D convolutional layers.  

Parameters:  
- args (GlobalSettings): The global settings for the model.  
- cfg (list[LayerCfg]): The list of layer configurations.  
- layer (COneDLayer): The type of layer to use.  
- tail_trans_cfg (TransCfg): The configuration for the transformer layers. Defaults to None.  

Methods:  
- __init__(self, args: GlobalSettings, cfg:list[LayerCfg], layer:COneDLayer, tail_trans_cfg:TransCfg = None, **kwargs): Initializes the COneD object with the given arguments.  
- forward(self, x): Performs a forward pass of the COneD model.  

Example:  
```python
model = COneD(args, cfg, layer)
output = model(x)
```

### MAvTTP

A class that represents the MA TT model with average pooling and permute.  

Parameters:  
- args (GlobalSettings): The global settings for the model.  
- cfg (list[LayerCfg]): The list of layer configurations.  
- layer (MAvlayer): The type of layer to use.  
- tail_trans_cfg (TransCfg): The configuration for the transformer layers. Defaults to None.  

Methods:  
- __init__(self, args: GlobalSettings, cfg:list[LayerCfg], layer:MAvlayer, tail_trans_cfg:TransCfg = None, **kwargs): Initializes the MAvTTP object with the given arguments.  

Example:  
```python
model = MAvTTP(args, cfg, layer)
output = model(x)
```

### MATTPE

A class that represents the MA TT model with positional encoding and permute.  

Parameters:  
- args (GlobalSettings): The global settings for the model.  
- cfg (list[LayerCfg]): The list of layer configurations.  
- layer (MAvlayer): The type of layer to use.  
- tail_trans_cfg (TransCfg): The configuration for the transformer layers. Defaults to None.  

Methods:  
- __init__(self, args: GlobalSettings, cfg:list[LayerCfg], layer:MAvlayer, tail_trans_cfg:TransCfg = None, **kwargs): Initializes the MATTPE object with the given arguments.  
- forward(self, x): Performs a forward pass of the MATTPE model.  

Example:  
```python
model = MATTPE(args, cfg, layer)
output = model(x)
```

## Constants

### str2net

A dictionary that maps strings to network classes.  

Example:  
```python
str2net = {
    'EncoderLayer':bb.EncoderLayer,
    'OutEncoderLayer':bb.OutEncoderLayer,
    'OutEncoderLayerAvg':bb.OutEncoderLayerAvg,
    'Trans':bb.Trans,
    'TransPE':bb.TransPE,
    'TransAvg':bb.TransAvg,
    
    'SeparableConv2d':bb.SeparableConv2d,
    'ResBlock':bb.ResBlock,
    'ResBlockR':bb.ResBlockR,
    'SABlock':bb.SABlock,
    'SABlockR':bb.SABlockR,
    'SABlock1D':bb.SABlock1D,
    'SABlock1DR':bb.SABlock1DR,
    
    'COneDLayer':COneDLayer,
    'MAlayer':MAlayer,
    'MAvlayer':MAvlayer,
    'SCANlayer':SCANlayer,
}
```