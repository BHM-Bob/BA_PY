
import os
from pathlib import Path
from typing import Dict, List, Union

import cv2
from tqdm import tqdm

if __name__ == '__main__':
    # dev mode
    from mbapy.base import *
    from mbapy.file_utils.image import *
else:
    # release mode
    from ..base import *
    from .image import *

def get_cv2_video_attr(video, attr_name:str, ret_int:bool = True):
    """
    Get the value of a specific attribute from a cv2 video object.

    Parameters:
        - video: cv2 video object.
        - attr_name (str): The name of the attribute to retrieve. for CAP_PROP_FRAME_WIDTH, just pass 'FRAME_WIDTH'.
        - ret_int (bool, optional): Indicates whether to return the attribute value as an integer. Defaults to True.

    Returns:
        - The value of the specified attribute. If ret_int is True, the value is returned as an integer.
          Otherwise, the value is returned as is.

    Example:
        >>> video = cv2.VideoCapture(0)
        >>> frame_width = get_cv2_video_attr(video, 'FRAME_WIDTH')
        >>> print(frame_width)
        >>> # Output: 640
    """
    if ret_int:
        return int(video.get(getattr(cv2, 'CAP_PROP_'+attr_name)))
    else:
        return video.get(getattr(cv2, 'CAP_PROP_'+attr_name))
    
@parameter_checker(check_parameters_path, raise_err=False)
def extract_frames_by_index(video_path:str, frame_indices:List[int]):
    """
    Extracts frames from a video file at specified frame indices.

    Parameters:
        video_path (str): The path to the video file.
        frame_indices (List[int]): A list of frame indices to extract.

    Returns:
        List[np.ndarray]: A list of frames as NumPy arrays.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return put_err(f'{video_path:s} can not be opened with cv2', None)
    sum_frame = get_cv2_video_attr(cap, 'FRAME_COUNT')

    frames = []
    for frame_index in tqdm(frame_indices):
        if 0 <= frame_index < sum_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
    cap.release()
    return frames
    
@parameter_checker(check_parameters_path, raise_err=False)
def extract_frame_to_img(video_path:str, img_type = 'jpg', return_frames = False,
                         write_file = True, dir:str = None, sum_frame = -1,
                         read_frame_interval = 0, img_size = [-1, -1], **kwargs):
    """
    Extract frames from a video and save them as images.

    Parameters:
    - video_path (str): The path to the video file.
    - img_type (str): The type of image file to save (default: 'jpg').
    - return_frames (bool): Whether to return the frames as a list (default: False).
    - write_file (bool): Whether to save the frames as image files (default: True).
    - dir (str): The directory to save the image files (default: None).
    - sum_frame (int): The number of frames to extract (-1 means extract all frames, default: -1).
    - read_frame_interval (int): The interval between frames to be read (default: 0).
    - img_size (List[int]): The size of the output images (default: [-1, -1]).
    - **kwargs: Additional keyword arguments.

    Returns:
    - frames (List[array]): The extracted frames as a list, if `return_frames` is True.
    
    Files:
    - writes image files in dir, each image file name include frame time stamp in format HH-MM-SS.
    """
    import cv2

    # Create the directory if it doesn't exist
    if write_file:
        video_path = Path(video_path)
        if not dir:
            dir = video_path.parent / video_path.stem
        if not os.path.exists(dir):
            os.makedirs(dir)
    # Open the video file
    video = cv2.VideoCapture(str(video_path))
    sum_frame = get_cv2_video_attr(video, 'FRAME_COUNT') if sum_frame == -1 else sum_frame
    frame_size = [get_cv2_video_attr(video, 'FRAME_WIDTH'), get_cv2_video_attr(video, 'FRAME_HEIGHT')]
    fps = get_cv2_video_attr(video, 'FPS')
    img_size[0] = frame_size[0] if img_size[0] <= 0 else img_size[0]
    img_size[1] = frame_size[1] if img_size[1] <= 0 else img_size[1]
    is_img_size_changed = img_size[0] != frame_size[0] and img_size[1] != frame_size
    # Read frames from the video
    frame_idx, fps_idx, frames = 0, 0, []
    bar = tqdm(range(sum_frame), desc=f'extract frames in {video_path.stem}')
    while True:
        # must read before skip
        success, frame = video.read()
        if read_frame_interval == 0 or frame_idx % (read_frame_interval+1) == 0:
            if not success:
                break
            if is_img_size_changed:
                cv2.resize(frame, img_size)
            if return_frames:
                frames.append(frame)
            # write frames to img file if needed
            if write_file:
                time = format_secs(frame_idx // fps, '{0:d}-{1:d}-{2:d}')
                img_name = f"{video_path.stem}_{time}-{fps_idx}.{img_type}"
                img_path = os.path.join(dir, img_name) if dir else img_name
                cv2.imwrite(img_path, frame)
            # update progress bar
            bar.update(read_frame_interval+1)
        frame_idx += 1
        fps_idx +=1
        if fps_idx == fps:
            fps_idx = 0
    # Release the video file
    video.release()
    # return frames lst
    return frames

@parameter_checker(check_parameters_path, raise_err=False)
def extract_unique_frames(video_path, threshold, read_frame_interval = 0,
                          scale_factor=1.0, compare_gray = True,
                          backend = 'skimage', model_dir = None,
                          torch_device = 'cuda'):
    """
    Extracts unique frames from a video based on structural similarity index (SSIM).

    Parameters:
        - video_path (str): The path to the video file.
        - threshold (float): The threshold value for SSIM. Frames with SSIM values above this threshold will be considered unique.
        - read_frame_interval (int, optional): The interval at which frames should be read from the video. Defaults to 0 (every frame).
        - scale_factor (float, optional): The factor by which the frame should be scaled. Defaults to 1.0 (no scaling).
        - compare_gray (bool, optional): Whether to compare frames in grayscale. Defaults to True.
        - backend (str, optional): The backend library to use for SSIM calculation. Defaults to 'skimage'.
            can be 'numpy', 'pytorch' or 'skimage'.
        - model_dir (str, optional): The directory containing the neural network models. Defaults to './data/nn_models/'.
        - torch_device (str, optional): The device to use for pytorch backend. Defaults to 'cuda'.

    Returns:
        Tuple[List[int], List[ndarray]]: A tuple containing two lists - the indices of the unique frames and the unique frames themselves.

    Notes:
        - This function requires the OpenCV and skimage libraries.
        - If the backend is set to 'torch-res50', this function requires a cuda compatible pytorch package.
            And will get features from the resnet50 model.
    """
    if backend == 'skimage':
        from skimage.metrics import structural_similarity
    elif backend == 'numpy':
        import numpy as np
    elif backend in ['pytorch', 'torch-res50']:
        import torch
    else:
        return put_err(f'backend {backend:s} is not supported', None)
    
    def _calculate_ssim(img1, img2):
        if backend == 'skimage':
            return structural_similarity(img1, img2, full=True)[1].mean()
        elif backend == 'numpy':
            return np.diff(img1, img2).mean()
        elif backend == 'pytorch':
            return torch.nn.functional.l1_loss(img1, img2).item()
        
    # open video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        return put_err(f'{video_path:s} can not be opened with cv2', None)
    fps = get_cv2_video_attr(video, 'FPS')
    width = int(get_cv2_video_attr(video, 'FRAME_WIDTH')*scale_factor)
    heigth = int(get_cv2_video_attr(video, 'FRAME_HEIGHT')*scale_factor)
    # compare the frames by SSIM
    unique_frames, unique_frames_idx = [], []
    progress_bar = tqdm(range(get_cv2_video_attr(video, 'FRAME_COUNT')), desc='unique frames: 1')
    if backend == 'torch-res50':
        from mbapy.file_utils.image import (_get_transform, _load_nn_model,
                                            calculate_frame_features)
        model = _load_nn_model(model_dir).to(torch_device)
        trans = _get_transform((width, heigth), (width, heigth), device=torch_device)
        _calculate_ssim = torch.nn.functional.cosine_similarity
    for frame_idx in range(get_cv2_video_attr(video, 'FRAME_COUNT')):
        # update frame_idx and progress bar
        progress_bar.update(1)
        # read next frame before skip
        ret, frame = video.read()
        if not ret:
            break
        # skip if frame is in interval
        if frame_idx % (read_frame_interval+1) != 0:
            continue
        # apply gray and scale factor
        if compare_gray and backend != 'torch-res50':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if scale_factor != 1.0:
            frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
        # tansfer frame to features
        if backend == 'torch-res50':
            frame = calculate_frame_features(frame, model, transform = trans).unsqueeze(0)
        # go through the history frame list, compare the SSIM for current frame with each frame
        is_unique = True
        for hist_frame in unique_frames:
            ssim = _calculate_ssim(frame, hist_frame)
            if ssim > threshold:
                is_unique = False
                break
        # append current frame if is unique
        if is_unique:
            unique_frames.append(frame)
            unique_frames_idx.append(frame_idx)
            time = '-'.join(map(lambda x: str(x), format_secs(frame_idx / fps)))
            progress_bar.set_description(f'{time:s} unique frame: {len(unique_frames)}')
    # release the video, return the unique frames
    video.release()
    return unique_frames_idx, unique_frames

__all__ = [
    'get_cv2_video_attr',
    'extract_frames_by_index',
    'extract_frame_to_img',
    'extract_unique_frames',
]

if __name__ == '__main__':
    # dev code
    extract_unique_frames('./data_tmp/video.mp4', 0.8, read_frame_interval=10, scale_factor=0.7)
    extract_frame_to_img('./data_tmp/video.mp4', dir = './data_tmp/video_full')
    # extract unique frames
    import time
    from glob import glob
    video_paths = glob(r"./data_tmp/*.mp4")
    interval, backend = 10, 'torch-res50'
    for video_path in video_paths:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        wdir = f'./data_tmp/unique_frames/{video_name} {get_fmt_time(timestamp = time.time())}'
        os.makedirs(wdir, exist_ok=True)
        idx, frames = extract_unique_frames(video_path, threshold=0.999,
                                    read_frame_interval=interval, scale_factor=0.7, backend=backend)
        for frame_idx, frame in enumerate(extract_frames_by_index(video_path, idx)):
            imwrite(os.path.join(wdir, f"frame_{frame_idx}.jpg"), frame)