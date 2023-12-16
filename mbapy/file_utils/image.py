'''
Date: 2023-07-18 23:01:44
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-08-03 22:44:54
FilePath: \BA_PY\mbapy\file_utils\image.py
Description: 
'''
import os
from typing import List, Union

import cv2
import numpy as np
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

if __name__ == '__main__':
    from mbapy.base import (check_parameters_path, get_default_call_for_None,
                            get_default_for_None, get_storage_path,
                            parameter_checker, put_err, put_log)
else:
    from ..base import (check_parameters_path, get_default_call_for_None,
                        get_default_for_None, get_storage_path,
                        parameter_checker, put_err, put_log)

@parameter_checker(check_parameters_path, raise_err=False)
def imread(path: str):
    img = cv2.imread(path)
    return img if img else cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1) # 中文路径
    
def imwrite(path: str, img:Union[np.ndarray, cv2.UMat, cv2.VideoCapture], quality:int=100):
    # check image and transfer to Image.Image
    if isinstance(img, np.ndarray):
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
    elif isinstance(img, cv2.UMat):
        img = cv2.UMat.get(img)
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
    elif isinstance(img, cv2.VideoCapture):
        success, frame = img.read()
        if success:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
        else:
            put_err(f"Failed to read video frame, return False", False)
            return False
    # save image
    if isinstance(img, Image.Image):
        try:
            img.save(path, format='JPEG', quality=quality)
            return True
        except Exception as e:
            put_err(f"Error saving image: {e}, return False", False)
            return False
    else:
        return put_err(f"Unknown image type: {type(img)}, return False", False)

def _load_nn_model(model_dir: str = None, model_name: str = 'resnet50'):
    """
    Load the neural network model from the specified directory or download to the directory.
    Notes: This func will remove the last layer of the model.

    Parameters:
        model_dir (str): The directory where the model is stored, defaults is 'path/to/mbapy/storage/nn_models'.
        model_name (str): The name of the model to load.

    Returns:
        model (torch.nn.Module): The loaded neural network model.
    """
    if model_dir is None:
        model_dir = get_storage_path('nn_models')
        put_log(f'Using default model directory: {model_dir}')
    torch.hub.set_dir(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    available_models = torchvision.models.__dict__
    if model_name not in available_models:
        raise ValueError(f"Model '{model_name}' is not available in torchvision.")

    model = available_models[model_name](pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model

def _get_transform(resize, crop,
                   normalize = None, device = 'cpu'):
    """
        Returns a torchvision transform composed of a series of image transformations.

        Args:
            resize (int, optional): The size of the shorter side of the image after resizing. Defaults to 256.
            crop (int, optional): The size of the final cropped image. Defaults to 224.
            normalize (dict, optional): A dictionary containing the mean and standard deviation values for image normalization. If not provided, default values of [0.485, 0.456, 0.406] for mean and [0.229, 0.224, 0.225] for standard deviation will be used. Defaults to None.
            device (str, optional): The device on which the tensor will be stored. Defaults to 'cpu'.

        Returns:
            torchvision.transforms.Compose: A torchvision transform composed of the specified image transformations.
    """
    normalize = get_default_for_None(normalize, {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]})
    transform = transforms.Compose([
        transforms.ToTensor(),
        lambda x: x.to(device = device),
        transforms.Resize(get_default_for_None(resize, 256), antialias=True),
        transforms.CenterCrop(get_default_for_None(crop, 224)),
        transforms.Normalize(**normalize)
    ])
    return transform

def calculate_frame_features(frame:Union[cv2.Mat, List[cv2.Mat]], model:torch.nn.Module = None, model_dir:str=None,
                             transform = None, resize=None, crop = None):
    """
    Calculates frame features using a given model and transform.

    Args:
        frame (Union[cv2.Mat, List[cv2.Mat]]): The frame or frames to calculate features on.
        model (torch.nn.Module, optional): The model to use for feature calculation. Defaults to None.
        model_dir (str, optional): The directory containing the model weights. Defaults to None.
            **Attention: There must be one is not None for model and model_path**. 
        transform (object, optional): The transform to apply to the frame(s). If None, the image is transformed to 224 pixels.
        resize (object, optional): The resize configuration for the transform. If None, the image is resized to 256 pixels.
        crop (object, optional): The crop configuration for the transform. If None, the image is cropped to 224 pixels.

    Returns:
        Union[Tensor, List[Tensor]]: The calculated frame features.
    """
    # load model and transform
    model = get_default_call_for_None(model, _load_nn_model, model_dir)
    transform = get_default_call_for_None(transform, _get_transform, resize, crop)
    # calculate features
    with torch.no_grad():
        if isinstance(frame, list):
            features = []
            for x in frame:
                x = transform(x).unsqueeze(0)
                features.append(model(x).view(-1))
        else:
            x = transform(frame).unsqueeze(0)
            features = model(x).view(-1)
    return features

__all__ = [
    'imread',
    'imwrite',
    '_load_nn_model',
    '_get_transform',
    'calculate_frame_features',    
]

if __name__ == '__main__':
    # 加载帧
    frame1 = cv2.imread("./data_tmp/extract_frames/frame_0-0-0.jpg")
    frame2 = cv2.imread("./data_tmp/extract_frames/frame_0-1-50.jpg")

    # 计算帧的特征向量
    model = _load_nn_model("./data_tmp/nn_model/torchvision")
    features1, features2 = calculate_frame_features([frame1, frame2], model, resize=(224, 224))

    # 比较特征向量
    similarity = torch.cosine_similarity(features1, features2, dim=0)

    print("Similarity:", similarity.item())
