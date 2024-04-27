
*`Kimi` generated*.

## Command: edit
### Introduction
The `edit` command is a simple video editor that allows users to adjust the audio volume and video speed of common video files. It also provides the option to remove the audio track entirely.
### Parameters
- `--audio`: Adjust the audio volume. A floating-point number representing the volume level, with the default set to `1.0`.
- `--speed`: Adjust the video speed. A floating-point number representing the speed factor.
- `-i`, `--input`: The file path or directory path of the input video file(s).
- `-o`, `--output`: The file path or directory path where the edited video will be saved.

### Behavior
The command processes each input video file by adjusting the audio and speed as specified. If the input is a directory, it will process all video files within that directory. The edited videos are saved with a suffix indicating the changes made.

### Notes
- This command utilizes the `moviepy` library for video editing.
- The audio volume and speed adjustments are applied to all input videos specified.

### Example
```
mbapy-cli video edit --input "video.mp4" --output "edited_video.mp4" --audio 0.8 --speed 1.2
```

---

## Command: extract
### Introduction
The `extract` command is used to extract frames or audio from video files. It supports extracting frames based on index, all frames, or unique frames based on image similarity.
### Parameters
- `content`: The type of content to extract, either `audio` or `frames`.
- `-i`, `--input`: The file path or directory path of the input video file(s).
- `-r`, `--recursive`: A flag to indicate whether the search for video files should be recursive.
- `-m`, `--mode`: The mode for frame extraction, which can be `index`, `all`, or `unique`.
- `-idx`, `--frame-index`: A string defining the frame index range and step in the format `start:end:step`.
- `-interval`, `--frame-interval`: The interval between frames to extract.
- `-size`, `--frame-size`: The size of the extracted frames in the format `width,height`.
- `-th`, `--threshold`: The threshold for image similarity when extracting unique frames.
- `-scale`, `--scale`: The scale factor for the image size.
- `-gray`, `--gray`: A flag to indicate whether to compare images in grayscale.
- `-backend`, `--backend`: The backend to use for image comparison.
- `-tdevice`, `--torch-device`: The torch device to use for image comparison.
- `-mdir`, `--model-dir`: The model directory path for image comparison.
- `--audio-nbytes`: The number of bytes for audio, with options for 16-bit or 32-bit sound.
- `--audio-bitrate`: The bitrate for the extracted audio.

### Behavior
The command processes each input video file by extracting the specified content. For frame extraction, it can extract frames based on the provided index, all frames with an optional interval, or unique frames based on visual similarity. For audio extraction, it saves the audio track with the specified bitrate and sample size.

### Notes
- Frame extraction uses the `PIL` and `tqdm` libraries for image processing and progress display.
- Audio extraction uses the `moviepy` library to write the audio file with the specified parameters.

### Example
```
mbapy-cli video extract frames --input "video.mp4" --mode all --recursive --frame-size "640,480" --threshold 0.9
```

### Additional Information
Please replace `"video.mp4"` with the actual path to your video file when using the command. If you are running this script as a standalone Python script, ensure that the required libraries (`moviepy`, `PIL`, `tqdm`, etc.) are installed and properly configured in your Python environment.