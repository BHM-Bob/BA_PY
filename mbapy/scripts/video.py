'''
Date: 2024-02-05 12:03:34
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2024-02-08 13:31:15
Description: 
'''
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

os.environ['MBAPY_AUTO_IMPORT_TORCH'] = 'False'
os.environ['MBAPY_FAST_LOAD'] = 'True'
from mbapy import base as mb, file as mf

if __name__ == '__main__':
    from mbapy.scripts._script_utils_ import clean_path, show_args
else:
    from ._script_utils_ import clean_path, show_args
    
    
def edit_video(args):
    from moviepy.editor import VideoFileClip
    # process args
    args.input = clean_path(args.input)
    args.out = clean_path(args.out)
    if os.path.isdir(args.input):
        args.input = mf.get_paths_with_extension(args.input,
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
    args.output = clean_path(args.output)
    if os.path.isdir(args.input):
        args.input = mf.get_paths_with_extension(args.input,
                                                 ['mp4', 'avi', 'mov', 'mkv',
                                                  'mpg', 'mpeg', 'wmv', 'flv',
                                                  'rmvb'])
    else:
        args.input = [args.input]
    # show args
    show_args(args, ['content', 'input', 'output','recursive'])
    # process each input
    for idx, path in enumerate(args.input):
        print(f'processing {idx+1}/{len(args.input)}: {path}')
        if args.content == 'audio':
            raise NotImplementedError('audio extraction is not implemented yet.')
        elif args.content == 'frames':
            # extract frames
            if args.mode == 'index':
                start, end, step = map(int, args.frame_index.split(':'))
                frames =mf.extract_frames_by_index(path, list(range(start, end, step)))
            elif args.mode == 'all':
                img_size = tuple(map(int, args.frame_size.split(',')))
                frames = mf.extract_frame_to_img(path, '', True, False, None,
                                                 read_frame_interval=args.frame_interval,
                                                 img_size=img_size)
            elif args.mode == 'unique':
                frames = mf.extract_unique_frames(
                    path, args.threshold, args.frame_interval, args.scale,
                    args.gray, args.backend, args.model_dir)
            else:
                raise ValueError(f'unknown mode: {args.mode}')
            # save frames
    

_str2func = {
    'edit':edit_video,
    
    'extract':extract_video,
}


# if __name__ == '__main__':
#     # dev code
#     from mbapy.game import BaseInfo

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
    extract_args.add_argument('-o', '--output', type=str, default='.',
                              help='output dir path, default is %(default)s.')
    extract_args.add_argument('-r', '--recursive', action='store_true', default=False,
                              help='FLAG, recursive search. Default is %(default)s.')
    extract_args.add_argument('-m', '--mode', type=str, choices=['index', 'all', 'unique'], default='unique',
                              help='extract mode, default is %(default)s.')
    extract_args.add_argument('-idx', '--frame-index', type=str, default='',
                              help='Frmae index to extract, format: start:end:step", Default is %(default)s.')
    extract_args.add_argument('-interval', '--frame-interval', type=int, default=0,
                              help='Frmae interval to extract, Default is %(default)s.')
    extract_args.add_argument('-size', '--frame-size', type=str, default='-1,-1',
                              help='image size to save, format: width,height, Default is %(default)s.')
    extract_args.add_argument('-th', '--threshold', type=float, default=0.9,
                              help='threshold for image similarity, Default is %(default)s.')
    extract_args.add_argument('-scale', '--scale', type=float, default=1.0,
                              help='scale factor for image size, Default is %(default)s.')
    extract_args.add_argument('-gray', '--gray', action='store_true', default=False,
                              help='FLAG, compare gray image, Default is %(default)s')
    extract_args.add_argument('-backend', '--backend', type=str, default='skimage',
                              help='backend for compare image, Default is %(default)s.')
    extract_args.add_argument('-mdir', '--model-dir', type=str, default='',
                              help='model dir path, Default is %(default)s.')
    
    
    args = args_paser.parse_args(sys_args)
    
    if args.sub_command in _str2func:
        print(f'excuting command: {args.sub_command}')
        _str2func[args.sub_command](args)
    else:
        mb.put_err(f'no such sub commmand: {args.sub_command}')

if __name__ == "__main__":
    main()