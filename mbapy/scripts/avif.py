'''
Date: 2023-08-16 16:07:51
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2024-02-05 15:11:09
Description: convert jpeg to avif
'''
import argparse
import os
import time
import shutil
from typing import Dict, List
from pathlib import Path
from multiprocessing import Manager, Pool

from tqdm import tqdm
from mbapy.base import put_err, split_list
from mbapy.file import get_paths_with_extension, opts_file, format_file_size
from PIL import Image
from pillow_heif import register_avif_opener, register_heif_opener

if __name__ == '__main__':
    from mbapy.scripts._script_utils_ import clean_path, show_args
else:
    from ._script_utils_ import clean_path, show_args

register_avif_opener()
register_heif_opener()

def transfer_img(args, paths: List[str], file_size: Dict[str, int]):
    before_size, after_size = 0, 0
    for path in (paths if args.multi_process > 1 else tqdm(paths)):
        path = str(clean_path(path))
        sub_path = path[len(str(args.input)):]
        if sub_path.startswith(os.path.sep):
            sub_path = sub_path[len(os.path.sep):]
        output_path = os.path.join(args.output, sub_path)
        before_size += os.path.getsize(path)
        if path.endswith(args.to) and path != output_path:
            shutil.copy(path, output_path)
        else:
            try:
                output_path = output_path[:1-len(Path(path).suffix)] + args.to
                output_dir = Path(output_path).parent
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                with Image.open(path) as im:
                    im.save(output_path, optimize=True, quality=args.quality)
            except:
                shutil.copy(path, output_path)
            if args.remove_origin:
                os.remove(path)
        after_size += os.path.getsize(output_path)
    # update file size and return
    file_size['before'] += before_size
    file_size['after'] += after_size
    return file_size
        

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
    args_paser.add_argument('-ifmt', '--input-format', type=str,
                            default='jpg,jpeg,png,JPG,JPEG,PNG',
                            help='output file path or dir path, default is %(default)s.')
    args_paser.add_argument('-m', '--multi-process', type=int, default=4,
                            help='number of processes for parallel processing, default is %(default)s.')
    args_paser.add_argument('-b', '--batch', type=int, default=10,
                            help='number of batch size for a processes, default is %(default)s.')
    args = args_paser.parse_args(sys_args)
    
    args.input = clean_path(args.input)
    args.output = clean_path(args.output)
    args.input_format = args.input_format.split(',')
    show_args(args, ['to', 'input', 'output', 'quality', 'remove_origin',
                     'input_format', 'multi_process', 'recursive'])
    
    paths = get_paths_with_extension(args.input, args.input_format,
                                     args.recursive)
    print(f'sum of files: {len(paths)}')
    
    if args.multi_process == 1:
        file_size = transfer_img(args, paths, dict(before=0, after=0))
    elif args.multi_process > 1:
        with Manager() as manager:
            file_size = manager.dict(before=0, after=0)
            pool = Pool(args.multi_process)
            for path_batch in tqdm(split_list(paths, args.batch)):
                pool.apply_async(func=transfer_img,
                                 args=(args, path_batch, file_size))
                while len(pool._cache) == args.multi_process:
                    time.sleep(1)
            file_size = dict(**file_size)
            
        pool.close()
        pool.join()
        
    print(f'before: {format_file_size(file_size["before"])}')
    print(f'after: {format_file_size(file_size["after"])}')
    print(f'decrease: {format_file_size(file_size["before"] - file_size["after"])}')
    print(f'rate: {round(file_size["after"] / file_size["before"] * 100, 2)}%')

# dev code
# main(['-i', r'E:\My_Progs\z_Progs_Data_HC', '-o', r'E:\My_Progs\z_Progs_Data_HC\avif',
#       '-r', '-t', 'avif', '-q', '80', '-ifmt', 'jpg', '-m', '1'])


if __name__ == "__main__":
    main()