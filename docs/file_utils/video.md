# mbapy.file_utils.video

This module provides functions for working with video files, including extracting frames from a video, saving frames as images, and extracting unique frames based on structural similarity index (SSIM).  

## Functions

### get_cv2_video_attr(video, attr_name:str, ret_int:bool = True) -> Any

Get the value of a specific attribute from a cv2 video object.  

Parameters:  
- video: cv2 video object.  
- attr_name (str): The name of the attribute to retrieve. for CAP_PROP_FRAME_WIDTH, just pass 'FRAME_WIDTH'.  
- ret_int (bool, optional): Indicates whether to return the attribute value as an integer. Defaults to True.  

Returns:  
- Any: The value of the specified attribute. If ret_int is True, the value is returned as an integer. Otherwise, the value is returned as is.  

Example:  
```python
video = cv2.VideoCapture(0)
frame_width = get_cv2_video_attr(video, 'FRAME_WIDTH')
print(frame_width)
# Output: 640
```

### extract_frames_by_index(video_path:str, frame_indices:List[int]) -> List[np.ndarray]

Extracts frames from a video file at specified frame indices.  

Parameters:  
- video_path (str): The path to the video file.  
- frame_indices (List[int]): A list of frame indices to extract.  

Returns:  
- List[np.ndarray]: A list of frames as NumPy arrays.  

### extract_frame_to_img(video_path:str, img_type = 'jpg', return_frames = False, write_file = True, dir:str = None, sum_frame = -1, read_frame_interval = 0, img_size = [-1, -1], **kwargs) -> List[array]

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

### extract_unique_frames(video_path, threshold, read_frame_interval = 0, scale_factor=1.0, compare_gray = True, backend = 'skimage', model_dir = './data/nn_models/') -> Tuple[List[int], List[ndarray]]

Extracts unique frames from a video based on structural similarity index (SSIM).  

Parameters:  
- video_path (str): The path to the video file.  
- threshold (float): The threshold value for SSIM. Frames with SSIM values above this threshold will be considered unique.  
- read_frame_interval (int, optional): The interval at which frames should be read from the video. Defaults to 0 (every frame).  
- scale_factor (float, optional): The factor by which the frame should be scaled. Defaults to 1.0 (no scaling).  
- compare_gray (bool, optional): Whether to compare frames in grayscale. Defaults to True.  
- backend (str, optional): The backend library to use for SSIM calculation. Defaults to 'skimage'.  
- model_dir (str, optional): The directory containing the neural network models. Defaults to './data/nn_models/'.  

Returns:  
- Tuple[List[int], List[ndarray]]: A tuple containing two lists - the indices of the unique frames and the unique frames themselves.  

Notes:  
- This function requires the OpenCV and skimage libraries.  
- If the backend is set to 'torch-res50', this function requires a cuda compatible pytorch package and will get features from the resnet50 model.