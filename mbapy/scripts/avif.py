'''
Date: 2023-08-16 16:07:51
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2025-08-06 16:53:59
Description: convert jpeg to avif
'''
import argparse
import os
import platform
import shutil
import time
from multiprocessing import Manager
from pathlib import Path
from typing import Dict, List

from PIL import Image
from pillow_heif import register_heif_opener
from tqdm import tqdm

from mbapy.base import put_err, split_list
from mbapy.file import format_file_size, get_paths_with_extension, opts_file
from mbapy.scripts._script_utils_ import clean_path, show_args
from mbapy.web_utils.task import TaskPool

# disable waring: 
# Limit to around a quarter gigabyte for a 24-bit (3 bpp) image
Image.MAX_IMAGE_PIXELS = None

try:
    # in higher version (>0.22.0), avif is decrepated
    from pillow_heif import register_avif_opener
    register_avif_opener()
except:
    pass
register_heif_opener()


def transfer_img(args, paths: List[str]):
    before_size, after_size = 0, 0
    for path in (paths if args.multi_process > 1 else tqdm(paths)):
        sub_path = Path(path).relative_to(args.input)
        output_path = (Path(args.output) / sub_path).with_suffix(f'.{args.to}')
        before_size += os.path.getsize(path)
        try:
            if not output_path.parent.exists():
                os.makedirs(output_path.parent)
            with Image.open(path) as im:
                im.save(output_path, optimize=True, quality=args.quality)
        except:
            shutil.copy(path, output_path)
        if args.remove_origin:
            if platform.system().lower() == 'windows':
                os.system(f'attrib -r "{path}"')
            elif platform.system().lower() == 'linux':
                os.system(f'chmod 666 "{path}"')
            os.remove(path)
        after_size += os.path.getsize(output_path)
    
    return before_size, after_size
        

def main(sys_args: List[str] = None):
    args_paser = argparse.ArgumentParser()
    args_paser.add_argument('-t', '--to', type = str, default='avif',
                            choices=['avif', 'heic', 'jpg'],
                            help='format of output. Default is %(default)s')
    args_paser.add_argument('-q', '--quality', type = int, default=85,
                            help='quality of output. Default is %(default)s')
    args_paser.add_argument('-i', '--input', type=str, default='.',
                            help='input file path or dir path, default is %(default)s.')
    args_paser.add_argument('-r', '--recursive', action='store_true', default=False,
                            help='FLAG, recursive search, default is %(default)s.')
    args_paser.add_argument('-o', '--output', type=str, default='.',
                            help='output file path or dir path, default is %(default)s.')
    args_paser.add_argument('-rm', '--remove-origin', action='store_true', default=False,
                            help='FLAG, remove original files, default is %(default)s.')
    args_paser.add_argument('-ifmt', '--input-format', type=str, nargs='+',
                            default=['JPG', 'JPEG', 'PNG', 'BMP', 'jpg', 'jpeg', 'png', 'bmp'],
                            help='output file path or dir path, default is %(default)s.')
    args_paser.add_argument('-m', '--multi-process', type=int, default=4,
                            help='number of processes for parallel processing, default is %(default)s.')
    args_paser.add_argument('-b', '--batch', type=int, default=10,
                            help='number of batch size for a processes, default is %(default)s.')
    args = args_paser.parse_args(sys_args)
    
    args.input = clean_path(args.input)
    args.output = clean_path(args.output)
    show_args(args, ['to', 'input', 'output', 'quality', 'remove_origin',
                     'input_format', 'multi_process', 'batch', 'recursive'])
    # check ifmt and to has conflict
    if args.to in args.input_format:
        put_err(f'target format {args.to} is in input format, exit.')
        return
    paths = get_paths_with_extension(args.input, args.input_format,
                                     args.recursive)
    print(f'sum of files: {len(paths)}')
    if not paths:
        put_err('no files found, exit.')
        return
    
    before_size, after_size = 0, 0
    if args.multi_process == 1:
        before_size, after_size = transfer_img(args, paths)
    elif args.multi_process > 1:
        pool, batches_name = TaskPool('process', args.multi_process).start(), []
        batches = split_list(paths, args.batch)
        # add tasks to pool
        for batch in tqdm(batches, desc='processing batches'):
            batches_name.append(pool.add_task(None, transfer_img, args, batch))
            pool.wait_till(lambda : pool.count_waiting_tasks() == 0, 0.01, update_result_queue=False)
        results_dict = pool.wait_till_tasks_done(batches_name)
        for b, a in results_dict.values():
            before_size += b
            after_size += a
        pool.close(1)
        
    print(f'before: {format_file_size(before_size)}')
    print(f'after: {format_file_size(after_size)}')
    print(f'decrease: {format_file_size(before_size - after_size)}')
    print(f'rate: {round(after_size / before_size * 100, 2)}%')


if __name__ == "__main__":
    main()