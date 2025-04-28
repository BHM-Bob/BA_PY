'''
Date: 2024-02-05 12:03:34
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2025-02-15 11:51:37
Description: 
'''
import argparse
import os
from pathlib import Path
from typing import Dict, List

from moviepy.editor import *
from PIL import Image
from tqdm import tqdm

os.environ['MBAPY_AUTO_IMPORT_TORCH'] = 'False'
os.environ['MBAPY_FAST_LOAD'] = 'True'

if __name__ == '__main__':
    from mbapy.base import put_err
    from mbapy.file import get_paths_with_extension
    from mbapy.file_utils.video import (extract_frame_to_img,
                                        extract_frames_by_index,
                                        extract_unique_frames)
    from mbapy.scripts._script_utils_ import clean_path, show_args
else:
    from ..base import put_err
    from ..file import get_paths_with_extension
    from ..file_utils.video import (extract_frame_to_img,
                                    extract_frames_by_index,
                                    extract_unique_frames)
    from ._script_utils_ import clean_path, show_args
    
    
def edit_video(args):
    from moviepy.editor import VideoFileClip

    # process args
    args.input = clean_path(args.input)
    args.out = clean_path(args.out)
    if os.path.isdir(args.input):
        args.input = get_paths_with_extension(args.input,
                                              ['mp4', 'avi', 'mov', 'mkv',
                                               'mpg', 'mpeg', 'wmv', 'flv',
                                               'rmvb'])
    else:
        args.input = [args.input]
    # show args
    show_args(args, ['audio','speed', 'input', 'output'])
    # process each input
    for idx, path in enumerate(args.input):
        print(f'processing {idx+1}/{len(args.input)}: {path}')
        if args.audio == 0:
            video = VideoFileClip(str(path)).speedx(factor = args.speed).without_audio()
            suffix = f'audio={args.audio} speed={args.speed}'
            video.write_videofile(str(args.out / f'{path.stem}_{suffix}.mp4'),
                                  codec='libx264', audio_codec='aac')

def extract_video(args):
    # process args
    args.input = clean_path(args.input)
    if os.path.isdir(args.input):
        args.input = get_paths_with_extension(args.input,
                                              ['mp4', 'avi', 'mov', 'mkv',
                                               'mpg', 'mpeg', 'wmv', 'flv',
                                               'rmvb'], recursive=args.recursive)
    else:
        args.input = [args.input]
    # show args
    show_args(args, ['content', 'input', 'recursive', 'mode', 'frame_index',
                     'frame_interval', 'frame_size', 'threshold','scale',
                     'gray', 'backend','model_dir', 'audio_nbytes',
                     'audio_bitrate'])
    # process each input
    for idx, path in enumerate(args.input):
        print(f'processing {idx+1}/{len(args.input)}: {path}')
        file_name = os.path.basename(path)
        file_type = file_name.split('.')[-1]
        file_dir = os.path.dirname(path)
        if args.content == 'audio':
            suffix = f'audio-{args.audio_nbytes*8}bit-{args.audio_bitrate}'
            dist_name = f'{file_name.replace("."+file_type, "")}_{suffix}.wav'
            dist_path = os.path.join(file_dir, dist_name)
            audio = VideoFileClip(path).audio
            if audio is None:
                continue
            audio.write_audiofile(dist_path, nbytes=args.audio_nbytes,
                                  bitrate=args.audio_bitrate,
                                  codec=f'pcm_s{8*args.audio_nbytes}le',
                                  verbose=True)
        elif args.content == 'frames':
            path = str(path)
            # extract frames
            if args.mode == 'index':
                start, end, step = map(int, args.frame_index.split(':'))
                frames = extract_frames_by_index(path, list(range(start, end, step)))
            elif args.mode == 'all':
                save_dir = os.path.join(file_dir, f'{file_name.replace("."+file_type, "")}_frames')
                frames = extract_frame_to_img(path, dir=save_dir,
                                              read_frame_interval=args.frame_interval,
                                              img_size=args.frame_size)
                continue
            elif args.mode == 'unique':
                frames = extract_unique_frames(
                    path, args.threshold, args.frame_interval, args.scale,
                    args.gray, args.backend, args.model_dir)
            else:
                raise ValueError(f'unknown mode: {args.mode}')
            # save frames
            for i, frame in enumerate(tqdm(frames, desc='saving frames')):
                frame_name = f'{file_name.replace("."+file_type, "")}_{i}.jpg'
                frame_path = os.path.join(file_dir, frame_name)
                img = Image.fromarray(frame)
                img.save(frame_path)
    

_str2func = {
    'edit':edit_video,
    
    'extract':extract_video,
}


def main(sys_args: List[str] = None):
    args_paser = argparse.ArgumentParser()
    subparsers = args_paser.add_subparsers(title='subcommands', dest='sub_command')
    
    edit_args = subparsers.add_parser('edit', description='simple editor for a commen video file.')
    edit_args.add_argument('--audio', type = float, default=1.0,
                           help='adjust the audio volume, float, range from 0. Default is %(default)s')
    edit_args.add_argument('--speed', type = float, default=1.0,
                           help='adjust the video speed, float. Default is %(default)s.')
    edit_args.add_argument('-i', '--input', type=str, default='.',
                           help='input file path or dir path, default is %(default)s.')
    edit_args.add_argument('-o', '--output', type=str, default='.',
                           help='output file path or dir path, default is %(default)s.')
    
    extract_args = subparsers.add_parser('extract', description='extract frames or audio from a video file.')
    extract_args.add_argument('content', choices=['audio', 'frames'], type=str, default='audio',
                              help='content to extract, default is %(default)s.')
    extract_args.add_argument('-i', '--input', type=str, default='.',
                              help='input file path or dir path, default is %(default)s.')
    extract_args.add_argument('-r', '--recursive', action='store_true', default=False,
                              help='FLAG, recursive search. Default is %(default)s.')
    extract_args.add_argument('-m', '--mode', type=str, choices=['index', 'all', 'unique'], default='unique',
                              help='extract mode, default is %(default)s.')
    extract_args.add_argument('-idx', '--frame-index', type=str, default='',
                              help='Frmae index to extract, format: start:end:step", Default is %(default)s.')
    extract_args.add_argument('-interval', '--frame-interval', type=int, default=0,
                              help='Frmae interval to extract, Default is %(default)s.')
    extract_args.add_argument('-size', '--frame-size', type=int, nargs='+', default=[-1, -1],
                              help='image size to save, format: width,height, Default is %(default)s.')
    extract_args.add_argument('-th', '--threshold', type=float, default=0.9,
                              help='threshold for image similarity, Default is %(default)s.')
    extract_args.add_argument('-scale', '--scale', type=float, default=1.0,
                              help='scale factor for image size, Default is %(default)s.')
    extract_args.add_argument('-gray', '--gray', action='store_true', default=False,
                              help='FLAG, compare gray image, Default is %(default)s')
    extract_args.add_argument('-backend', '--backend', type=str, default='skimage',
                              help='backend for compare image, Default is %(default)s.')
    extract_args.add_argument('-tdevice', '--torch-device', type=str, default='cuda',
                              help='torch device for compare image, Default is %(default)s.')
    extract_args.add_argument('-mdir', '--model-dir', type=str, default='',
                              help='model dir path, Default is %(default)s.')
    extract_args.add_argument('--audio-nbytes', type=int, default=4, choices=[2, 4],
                              help='nbytes for audio, set to 2 for 16-bit sound, 4 for 32-bit sound, Default is %(default)s.')
    extract_args.add_argument('--audio-bitrate', type=str, default='3000k',
                              help='bitrate for audio, Default is %(default)s.')
    
    
    args = args_paser.parse_args(sys_args)
    
    if args.sub_command in _str2func:
        print(f'excuting command: {args.sub_command}')
        _str2func[args.sub_command](args)
    else:
        put_err(f'no such sub commmand: {args.sub_command}')


if __name__ == "__main__":
    # dev code
    # comment the following line when release
    # main(['extract', 'frames', '-i', './data_tmp/video.mp4'])
    
    # RELEASE CODE
    main()