'''
Date: 2024-06-15 12:08:22
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2024-07-11 07:50:41
Description: 
'''

import argparse
import os
import platform
import shutil
from typing import Dict, List

from tqdm import tqdm

from mbapy.base import put_err
from mbapy.file import get_paths_with_extension

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
    args_paser.add_argument('-t', '--type', type = str, default='',
                            help='format of files to remove, splited by ",". Default is %(default)s')
    args_paser.add_argument('-n', '--name', type = str, default='',
                            help='sub-string of name of files to remove. Default is %(default)s')
    args_paser.add_argument('-r', '--recursive', action='store_true', default=False,
                            help='FLAG, recursive search. Default is %(default)s.')
    args = args_paser.parse_args(sys_args)
    
    # process args
    args.type = args.type.split(',')
    args.input = clean_path(args.input)
    args.output = clean_path(args.output)
    show_args(args, ['input', 'output', 'type', 'name', 'recursive'])
    
    # short cut if only a single file to move
    if os.path.isfile(args.input) and not os.path.isdir(args.output):
        return shutil.move(args.input, args.output)
    
    # get input paths
    paths = get_paths_with_extension(args.input, args.type,
                                     args.recursive, args.name)
    print(f'files to move: {len(paths)}')
    
    # move
    for path in tqdm(paths):
        try:
            sub_path = path[len(str(args.input)):]
            if sub_path.startswith(os.path.sep):
                sub_path = sub_path[len(os.path.sep):]
            output_path = os.path.join(args.output, sub_path)
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if platform.system().lower() == 'windows':
                os.system(f'attrib -r "{path}"')
            elif platform.system().lower() == 'linux':
                os.system(f'chmod 666 "{path}"')
            shutil.move(path, output_path)
        except Exception as e:
            put_err(f'can not move {path} with error {e}, skip')
    return paths
    
    

# dev code
# main(['-i', r'E:\\1', '-o', 'E:\\2', '-t', 'JPG'])


if __name__ == "__main__":
    main()