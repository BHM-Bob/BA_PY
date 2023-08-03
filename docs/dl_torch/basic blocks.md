# mbapy.dl_torch.bb

This module provides various building blocks for neural network models.  

## Classes

### CnnCfg

A class that represents the configuration of a convolutional neural network (CNN) layer.  

Parameters:  
- inc (int): The number of input channels.  
- outc (int): The number of output channels.  
- kernel_size (int, optional): The size of the convolutional kernel. Defaults to 3.  
- stride (int, optional): The stride of the convolution. Defaults to 1.  
- padding (int, optional): The padding of the convolution. Defaults to 1.  

Attributes:  
- inc (int): The number of input channels.  
- outc (int): The number of output channels.  
- kernel_size (int): The size of the convolutional kernel.  
- stride (int): The stride of the convolution.  
- padding (int): The padding of the convolution.  
- _str_ (str): A string representation of the configuration.  

Methods:  
- __init__(self, inc:int, outc:int, kernel_size:int = 3, stride:int = 1, padding:int = 1): Initializes the CnnCfg object with the given configuration.  
- __str__(self): Returns a string representation of the configuration.  

Example:  
```python
cfg = CnnCfg(3, 64)
print(cfg)
```
Output:  
```
inc=3,outc=64,kernel_size=3,stride=1,padding=1
```

### reshape

A class that represents a reshape operation.  

Methods:  
- __init__(self, *args, **kwargs): Initializes the reshape object with the given arguments.  
- forward(self, x): Performs the reshape operation on the input tensor x.  

Example:  
```python
reshape_layer = reshape(1, -1)
output = reshape_layer(input)
```

### permute

A class that represents a permute operation.  

Methods:  
- __init__(self, *args, **kwargs): Initializes the permute object with the given arguments.  
- forward(self, x): Performs the permute operation on the input tensor x.  

Example:  
```python
permute_layer = permute(1, 0)
output = permute_layer(input)
```

### ScannCore

A class that represents a single-head version of the Multi-Head Self-Attention (MHSA) mechanism.  

Parameters:  
- s (int): The size of the input tensor.  
- way (str, optional): The way to calculate the output. Can be "linear" or "avg". Defaults to "linear".  
- dropout (float, optional): The dropout rate. Defaults to 0.2.  

Methods:  
- __init__(self, s, way="linear", dropout=0.2): Initializes the ScannCore object with the given parameters.  
- forward(self, x): Performs the MHSA operation on the input tensor x.  

Example:  
```python
scann_core = ScannCore(256)
output = scann_core(input)
```

### SCANN

A class that represents the SCANN (Self-Attention Convolutional Neural Network) module.  

Parameters:  
- inc (int): The number of input channels.  
- group (int, optional): The number of channels in a group to be processed by the ScannCore. Defaults to 1.  
- stride (int, optional): The stride of the convolutional operation. Defaults to 2.  
- padding (int, optional): The padding of the convolutional operation. Defaults to 1.  
- kernel_size (int, optional): The size of the convolutional kernel. Defaults to 3.  
- outway (str, optional): The way to calculate the output of the ScannCore. Can be "linear" or "avg". Defaults to "linear".  
- dropout (float, optional): The dropout rate. Defaults to 0.2.  

Methods:  
- __init__(self, inc, group=1, stride=2, padding=1, kernel_size=3, outway="linear", dropout=0.2): Initializes the SCANN object with the given parameters.  
- forward(self, x): Performs the SCANN operation on the input tensor x.  

Example:  
```python
scann = SCANN(64)
output = scann(input)
```

### PositionalEncoding

A class that represents the positional encoding layer.  

Parameters:  
- d_model (int): The dimension of the input tensor.  
- max_len (int, optional): The maximum length of the input sequence. Defaults to 5000.  

Methods:  
- __init__(self, d_model, max_len=5000): Initializes the PositionalEncoding object with the given parameters.  
- forward(self, x): Performs the positional encoding operation on the input tensor x.  

Example:  
```python
pos_encoding = PositionalEncoding(512)
output = pos_encoding(input)
```

### PositionwiseFeedforwardLayer

A class that represents the position-wise feedforward layer.  

Parameters:  
- hid_dim (int): The hidden dimension of the layer.  
- pf_dim (int): The dimension of the position-wise feedforward layer.  
- dropout (float): The dropout rate.  

Methods:  
- __init__(self, hid_dim, pf_dim, dropout): Initializes the PositionwiseFeedforwardLayer object with the given parameters.  
- forward(self, x): Performs the position-wise feedforward operation on the input tensor x.  

Example:  
```python
feedforward_layer = PositionwiseFeedforwardLayer(512, 2048, 0.1)
output = feedforward_layer(input)
```

### MultiHeadAttentionLayer

A class that represents the multi-head attention layer.  

Parameters:  
- hid_dim (int): The hidden dimension of the layer.  
- n_heads (int): The number of attention heads.  
- dropout (float): The dropout rate.  
- device (str): The device to use for computation.  
- kwargs (optional): Additional keyword arguments.  

Methods:  
- __init__(self, hid_dim, n_heads, dropout, device='cuda', **kwargs): Initializes the MultiHeadAttentionLayer object with the given parameters.  
- forward(self, query, key, value): Performs the multi-head attention operation on the input tensors query, key, and value.  

Example:  
```python
attention_layer = MultiHeadAttentionLayer(512, 8, 0.1)
output = attention_layer(query, key, value)
```

### FastMultiHeadAttentionLayer

A class that represents the fast multi-head attention layer.  

Parameters:  
- hid_dim (int): The hidden dimension of the layer.  
- n_heads (int): The number of attention heads.  
- dropout (float): The dropout rate.  
- device (str): The device to use for computation.  
- kwargs (optional): Additional keyword arguments.  

Methods:  
- __init__(self, hid_dim, n_heads, dropout, device='cuda', **kwargs): Initializes the FastMultiHeadAttentionLayer object with the given parameters.  
- forward(self, query, key, value): Performs the fast multi-head attention operation on the input tensors query, key, and value.  

Example:  
```python
attention_layer = FastMultiHeadAttentionLayer(512, 8, 0.1)
output = attention_layer(query, key, value)
```

### OutMultiHeadAttentionLayer

A class that represents the output multi-head attention layer.  

Parameters:  
- q_len (int): The length of the query tensor.  
- class_num (int): The number of output classes.  
- hid_dim (int): The hidden dimension of the layer.  
- n_heads (int): The number of attention heads.  
- dropout (float): The dropout rate.  
- device (str): The device to use for computation.  
- kwargs (optional): Additional keyword arguments.  

Methods:  
- __init__(self, q_len, class_num, hid_dim, n_heads, dropout, device='cuda', **kwargs): Initializes the OutMultiHeadAttentionLayer object with the given parameters.  
- forward(self, query, key, value): Performs the output multi-head attention operation on the input tensors query, key, and value.  

Example:  
```python
attention_layer = OutMultiHeadAttentionLayer(512, 10, 512, 8, 0.1)
output = attention_layer(query, key, value)
```

### EncoderLayer

A class that represents an encoder layer.  

Parameters:  
- q_len (int): The length of the query tensor.  
- class_num (int): The number of output classes.  
- hid_dim (int): The hidden dimension of the layer.  
- n_heads (int): The number of attention heads.  
- pf_dim (int): The dimension of the position-wise feedforward layer.  
- dropout (float): The dropout rate.  
- device (str): The device to use for computation.  
- kwargs (optional): Additional keyword arguments.  

Methods:  
- __init__(self, q_len, class_num, hid_dim, n_heads, pf_dim, dropout, device='cuda', **kwargs): Initializes the EncoderLayer object with the given parameters.  
- forward(self, src): Performs the encoder layer operation on the input tensor src.  

Example:  
```python
encoder_layer = EncoderLayer(512, 10, 512, 8, 2048, 0.1)
output = encoder_layer(src)
```

### OutEncoderLayer

A class that represents the output encoder layer.  

Parameters:  
- q_len (int): The length of the query tensor.  
- class_num (int): The number of output classes.  
- hid_dim (int): The hidden dimension of the layer.  
- n_heads (int): The number of attention heads.  
- pf_dim (int): The dimension of the position-wise feedforward layer.  
- dropout (float): The dropout rate.  
- device (str): The device to use for computation.  
- kwargs (optional): Additional keyword arguments.  

Methods:  
- __init__(self, q_len, class_num, hid_dim, n_heads, pf_dim, dropout, device='cuda', **kwargs): Initializes the OutEncoderLayer object with the given parameters.  
- forward(self, src): Performs the output encoder layer operation on the input tensor src.  

Example:  
```python
encoder_layer = OutEncoderLayer(512, 10, 512, 8, 2048, 0.1)
output = encoder_layer(src)
```

### Trans

A class that represents the Transformer model.  

Parameters:  
- q_len (int): The length of the query tensor.  
- class_num (int): The number of output classes.  
- hid_dim (int): The hidden dimension of the model.  
- n_layers (int): The number of encoder layers.  
- n_heads (int): The number of attention heads.  
- pf_dim (int): The dimension of the position-wise feedforward layer.  
- dropout (float): The dropout rate.  
- device (str): The device to use for computation.  
- out_encoder_layer (optional): The output encoder layer class to use. Defaults to OutEncoderLayer.  
- kwargs (optional): Additional keyword arguments.  

Methods:  
- __init__(self, q_len, class_num, hid_dim, n_layers, n_heads, pf_dim, dropout, device, out_encoder_layer=OutEncoderLayer, **kwargs): Initializes the Trans object with the given parameters.  
- forward(self, src): Performs the Transformer model operation on the input tensor src.  

Example:  
```python
trans_model = Trans(512, 10, 512, 6, 8, 2048, 0.1)
output = trans_model(src)
```

### TransPE

A class that represents the Transformer model with positional encoding.  

Parameters:  
- q_len (int): The length of the query tensor.  
- class_num (int): The number of output classes.  
- hid_dim (int): The hidden dimension of the model.  
- n_layers (int): The number of encoder layers.  
- n_heads (int): The number of attention heads.  
- pf_dim (int): The dimension of the position-wise feedforward layer.  
- dropout (float): The dropout rate.  
- device (str): The device to use for computation.  
- kwargs (optional): Additional keyword arguments.  

Methods:  
- __init__(self, q_len, class_num, hid_dim, n_layers, n_heads, pf_dim, dropout, device, **kwargs): Initializes the TransPE object with the given parameters.  
- forward(self, src): Performs the Transformer model with positional encoding operation on the input tensor src.  

Example:  
```python
trans_model = TransPE(512, 10, 512, 6, 8, 2048, 0.1)
output = trans_model(src)
```

### OutEncoderLayerAvg

A class that represents the output encoder layer with average pooling.  

Parameters:  
- q_len (int): The length of the query tensor.  
- class_num (int): The number of output classes.  
- hid_dim (int): The hidden dimension of the layer.  
- n_heads (int): The number of attention heads.  
- pf_dim (int): The dimension of the position-wise feedforward layer.  
- dropout (float): The dropout rate.  
- device (str): The device to use for computation.  
- kwargs (optional): Additional keyword arguments.  

Methods:  
- __init__(self, q_len, class_num, hid_dim, n_heads, pf_dim, dropout, device='cuda', **kwargs): Initializes the OutEncoderLayerAvg object with the given parameters.  
- forward(self, src): Performs the output encoder layer with average pooling operation on the input tensor src.  

Example:  
```python
encoder_layer = OutEncoderLayerAvg(512, 10, 512, 8, 2048, 0.1)
output = encoder_layer(src)
```

### TransAvg

A class that represents the Transformer model with average pooling.  

Parameters:  
- q_len (int): The length of the query tensor.  
- class_num (int): The number of output classes.  
- hid_dim (int): The hidden dimension of the model.  
- n_layers (int): The number of encoder layers.  
- n_heads (int): The number of attention heads.  
- pf_dim (int): The dimension of the position-wise feedforward layer.  
- dropout (float): The dropout rate.  
- device (str): The device to use for computation.  
- kwargs (optional): Additional keyword arguments.  

Methods:  
- __init__(self, q_len, class_num, hid_dim, n_layers, n_heads, pf_dim, dropout, device, **kwargs): Initializes the TransAvg object with the given parameters.  
- forward(self, src): Performs the Transformer model with average pooling operation on the input tensor src.  

Example:  
```python
trans_model = TransAvg(512, 10, 512, 6, 8, 2048, 0.1)
output = trans_model(src)
```

### SeparableConv2d

A class that represents a separable convolutional layer.  

Parameters:  
- inc (int): The number of input channels.  
- outc (int): The number of output channels.  
- kernel_size (int): The size of the convolutional kernel.  
- stride (int): The stride of the convolution.  
- padding (int): The padding of the convolution.  
- depth (int, optional): The depth of the separable convolution. Defaults to 1.  

Methods:  
- __init__(self, inc, outc, kernel_size, stride, padding, depth=1): Initializes the SeparableConv2d object with the given parameters.  
- forward(self, x): Performs the separable convolution operation on the input tensor x.  

Example:  
```python
separable_conv = SeparableConv2d(3, 64, 3, 1, 1)
output = separable_conv(input)
```

### ResBlock

A class that represents a residual block.  

Parameters:  
- cfg (CnnCfg): The configuration of the residual block.  

Methods:  
- __init__(self, cfg): Initializes the ResBlock object with the given configuration.  
- forward(self, x): Performs the residual block operation on the input tensor x.  

Example:  
```python
res_block = ResBlock(CnnCfg(64, 64))
output = res_block(input)
```

### ResBlockR

A class that represents a residual block with exclusive gating.  

Parameters:  
- cfg (CnnCfg): The configuration of the residual block.  

Methods:  
- __init__(self, cfg): Initializes the ResBlockR object with the given configuration.  
- forward(self, x): Performs the residual block with exclusive gating operation on the input tensor x.  

Example:  
```python
res_block = ResBlockR(CnnCfg(64, 64))
output = res_block(input)
```

### SABlock

A class that represents a self-attention block.  

Parameters:  
- cfg (CnnCfg): The configuration of the self-attention block.  

Methods:  
- __init__(self, cfg): Initializes the SABlock object with the given configuration.  
- forward(self, x): Performs the self-attention block operation on the input tensor x.  

Example:  
```python
sa_block = SABlock(CnnCfg(64, 64))
output = sa_block(input)
```

### SABlockR

A class that represents a self-attention block with exclusive gating.  

Parameters:  
- cfg (CnnCfg): The configuration of the self-attention block.  

Methods:  
- __init__(self, cfg): Initializes the SABlockR object with the given configuration.  
- forward(self, x): Performs the self-attention block with exclusive gating operation on the input tensor x.  

Example:  
```python
sa_block = SABlockR(CnnCfg(64, 64))
output = sa_block(input)
```

### ScannBlock1d

A class that represents a 1D self-attention block.  

Parameters:  
- cfg (CnnCfg): The configuration of the 1D self-attention block.  

Methods:  
- __init__(self, cfg, **kwargs): Initializes the ScannBlock1d object with the given configuration.  
- forward(self, x): Performs the 1D self-attention block operation on the input tensor x.  

Example:  
```python
scann_block = ScannBlock1d(CnnCfg(64, 64))
output = scann_block(input)
```

### SABlock1D

A class that represents a 1D self-attention block.  

Parameters:  
- cfg (CnnCfg): The configuration of the 1D self-attention block.  

Methods:  
- __init__(self, cfg): Initializes the SABlock1D object with the given configuration.  
- forward(self, x): Performs the 1D self-attention block operation on the input tensor x.  

Example:  
```python
sa_block = SABlock1D(CnnCfg(64, 64))
output = sa_block(input)
```

### SABlock1DR

A class that represents a 1D self-attention block with exclusive gating.  

Parameters:  
- cfg (CnnCfg): The configuration of the 1D self-attention block.  

Methods:  
- __init__(self, cfg): Initializes the SABlock1DR object with the given configuration.  
- forward(self, x): Performs the 1D self-attention block with exclusive gating operation on the input tensor x.  

Example:  
```python
sa_block = SABlock1DR(CnnCfg(64, 64))
output = sa_block(input)
```

## Functions

### GenCnn1d

A function that generates a list of 1D convolutional layers.  

Parameters:  
- inc (int): The number of input channels.  
- outc (int): The number of output channels.  
- minCnnKSize (int): The minimum size of the convolutional kernel.  

Returns:  
- nn.ModuleList: A list of 1D convolutional layers.  

Example:  
```python
cnn_layers = GenCnn1d(64, 64, 3)
```