'''
Date: 2024-02-06 15:41:40
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2025-09-13 22:36:42
Description: 
'''

import argparse
import os
import shutil
from typing import Dict, List

from tqdm import tqdm
from mbapy.base import put_err
from mbapy.file import get_paths_with_extension, get_valid_path_on_exists

if __name__ == '__main__':
    from mbapy.scripts._script_utils_ import clean_path, show_args
else:
    from ._script_utils_ import clean_path, show_args

def main(sys_args: List[str] = None):
    # make and parse args
    args_paser = argparse.ArgumentParser(description = 'move files with specific suffix or sub-string in name')
    args_paser.add_argument('-i', '--input', type=str,
                            help='input/src files path or dir path.')
    args_paser.add_argument('-o', '--output', type=str,
                            help='output/dist files path or dir path.')
    args_paser.add_argument('-t', '--type', type = str, nargs='+', default=[],
                            help='format of files to remove, splited by ",". Default is %(default)s')
    args_paser.add_argument('-n', '--name', type = str, default='',
                            help='sub-string of name of files to remove. Default is %(default)s')
    args_paser.add_argument('-e', '--exclude', type = str, nargs='+', default=None,
                            help='sub-string of name of files to exclude. Default is %(default)s')
    args_paser.add_argument('-r', '--recursive', action='store_true', default=False,
                            help='FLAG, recursive search. Default is %(default)s.')
    args_paser.add_argument('--on-exist', type=str, choices=['skip', 'overwrite', 'random'], default='overwrite',
                            help='What to do if target file exists: skip, overwrite, or random (add random uuid4 suffix). Default is %(default)s.')
    args = args_paser.parse_args(sys_args)
    
    # process args
    args.input = clean_path(args.input)
    args.output = clean_path(args.output)
    show_args(args, ['input', 'output', 'type', 'name', 'recursive', 'on_exist'])
    
    # short cut if only a single file to move
    if os.path.isfile(args.input):
        output_file = args.output if os.path.isfile(args.output) else os.path.join(args.output, os.path.basename(args.input))
        if os.path.exists(output_file):
            if args.on_exist == 'skip':
                put_err(f'{output_file} exists, skip')
                return
            elif args.on_exist == 'random':
                output_file = get_valid_path_on_exists(output_file)
                if output_file is None:
                    put_err(f'Failed to generate unique filename for {args.output}, skip')
                    return
        return shutil.copy(args.input, output_file)
    
    # get input paths
    paths = get_paths_with_extension(args.input, args.type,
                                     args.recursive, args.name, neg_name_substr=args.exclude)
    
    # copy
    copied_count, skip_count, random_count, overwrite_count, err_count = 0, 0, 0, 0, 0
    for path in tqdm(paths):
        try:
            sub_path = os.path.relpath(path, args.input)
            output_path = os.path.join(args.output, sub_path)
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if os.path.exists(output_path):
                if args.on_exist == 'skip':
                    put_err(f'{output_path} exists, skip')
                    skip_count += 1
                    continue
                elif args.on_exist == 'random':
                    output_path_new = get_valid_path_on_exists(output_path)
                    if output_path_new is None:
                        put_err(f'Failed to generate unique filename for {output_path}, skip')
                        skip_count += 1
                        continue
                    output_path = output_path_new
                    random_count += 1
                else: # overwrite
                    overwrite_count += 1
            shutil.copy(path, output_path)
            copied_count += 1
        except Exception as e:
            put_err(f'can not copy {path}, skip ({e})')
            err_count += 1
            
    # summary
    print(f'files to copy: {len(paths)}\n', 
          f'copied: {copied_count}\n', 
          f'skip: {skip_count}\n', 
          f'random: {random_count}\n', 
          f'overwrite: {overwrite_count}\n',
          f'err: {err_count}')
    return paths
    
    

# dev code
# main(['-i', r'E:\\1', '-o', 'E:\\2', '-t', 'JPG'])


if __name__ == "__main__":
    main()