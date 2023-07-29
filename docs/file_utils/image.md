# mbapy.file_utils.image

This module provides functions for extracting features from frames using a neural network model.  

## Functions

### _load_nn_model(model_dir:str) -> torch.nn.Module

Load the neural network model from the specified directory.  

Parameters:  
- model_dir (str): The directory where the model is stored.  

Returns:  
- model (torch.nn.Module): The loaded neural network model.  

Example:  
```python
model = _load_nn_model('model_dir')
```

### _get_transform(resize, crop, normalize = None, device = 'cpu') -> torchvision.transforms.Compose

Returns a torchvision transform composed of a series of image transformations.  

Parameters:  
- resize (int, optional): The size of the shorter side of the image after resizing. Defaults to 256.  
- crop (int, optional): The size of the final cropped image. Defaults to 224.  
- normalize (dict, optional): A dictionary containing the mean and standard deviation values for image normalization. If not provided, default values of [0.485, 0.456, 0.406] for mean and [0.229, 0.224, 0.225] for standard deviation will be used. Defaults to None.  
- device (str, optional): The device on which the tensor will be stored. Defaults to 'cpu'.  

Returns:  
- torchvision.transforms.Compose: A torchvision transform composed of the specified image transformations.  

Example:  
```python
transform = _get_transform(resize=256, crop=224, normalize={'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]})
```

### calculate_frame_features(frame:Union[cv2.Mat, List[cv2.Mat]], model:torch.nn.Module = None, model_dir:str=None, transform = None, resize=None, crop = None) -> Union[Tensor, List[Tensor]]

Calculates frame features using a given model and transform.  

Parameters:  
- frame (Union[cv2.Mat, List[cv2.Mat]]): The frame or frames to calculate features on.  
- model (torch.nn.Module, optional): The model to use for feature calculation. Defaults to None.  
- model_dir (str, optional): The directory containing the model weights. Defaults to None.  
- transform (object, optional): The transform to apply to the frame(s). If None, the image is transformed to 224 pixels.  
- resize (object, optional): The resize configuration for the transform. If None, the image is resized to 256 pixels.  
- crop (object, optional): The crop configuration for the transform. If None, the image is cropped to 224 pixels.  

Returns:  
- Union[Tensor, List[Tensor]]: The calculated frame features.  

Example:  
```python
frame = cv2.imread('frame.jpg')
features = calculate_frame_features(frame, model, transform)
```
