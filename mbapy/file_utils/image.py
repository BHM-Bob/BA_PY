'''
Date: 2023-07-18 23:01:44
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-07-19 00:09:18
FilePath: \BA_PY\mbapy\file_utils\image.py
Description: 
'''
import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models as models

if __name__ == '__main__':
    from mbapy.base import get_default_call_for_None
else:
    from ..base import get_default_call_for_None

def _load_nn_model(model_dir:str):
    """
    Load the neural network model from the specified directory.

    Parameters:
        model_dir (str): The directory where the model is stored.

    Returns:
        model (torch.nn.Module): The loaded neural network model.
    """
    torch.hub.set_dir(model_dir)
    model = models.resnet50(pretrained=True, progress=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model

def _get_transform(resize, crop):
    """
    Returns a composed transformation function that resizes and crops an image.

    Parameters:
        resize (int or None): The desired size of the image after resizing. If None, the image is resized to 256 pixels.
        crop (int or None): The desired size of the image after cropping. If None, the image is cropped to 224 pixels.

    Returns:
        torchvision.transforms.Compose: A composed transformation function that resizes, crops, converts to a tensor, and normalizes the image.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(resize) if resize is not None else transforms.Resize(256),
        transforms.CenterCrop(crop) if crop is not None else transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

def calculate_frame_features(frame, model:torch.nn.Module = None, model_dir:str=None,
                             transform = None, resize=None, crop = None):
    """
    Calculate the frame features using a given model and transform.

    Args:
        frame (torch.Tensor): The input frame as a torch tensor.
        model (torch.nn.Module, optional): The neural network model to use for feature calculation. 
            Defaults to None.
        model_dir (str, optional): The directory to load the model from. Defaults to None.
        transform (callable, optional): The transformation to apply to the frame before feature calculation.
            Defaults to None.
        resize (tuple, optional): The size to resize the frame to. Defaults to None.
        crop (tuple, optional): The crop coordinates to apply to the frame. Defaults to None.

    Returns:
        torch.Tensor: The calculated features as a torch tensor.
    """
    # load model and transform
    model = get_default_call_for_None(model, _load_nn_model, model_dir)
    transform = get_default_call_for_None(transform, _get_transform, resize, crop)
    # calculate features
    frame = transform(frame).unsqueeze(0)
    with torch.no_grad():
        features = model(frame)
    features = torch.flatten(features)
    return features

if __name__ == '__main__':
    # 加载帧
    frame1 = cv2.imread("./data_tmp/extract_frames/frame_0-0-0.jpg")
    frame2 = cv2.imread("./data_tmp/extract_frames/frame_0-1-50.jpg")

    # 计算帧的特征向量
    model = _load_nn_model("./data_tmp/nn_model/torchvision")
    features1 = calculate_frame_features(frame1, model, resize=(224, 224))
    features2 = calculate_frame_features(frame2, model, resize=(224, 224))

    # 比较特征向量
    similarity = torch.cosine_similarity(features1, features2, dim=0)

    print("Similarity:", similarity.item())
