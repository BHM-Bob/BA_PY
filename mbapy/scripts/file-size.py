'''
Date: 2024-04-15 21:01:33
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2024-04-15 22:33:41
Description: 
'''

import argparse
import os
from typing import Dict, List

from tqdm import tqdm
from mbapy.base import put_err
from mbapy.file import get_paths_with_extension, format_file_size

if __name__ == '__main__':
    from mbapy.scripts._script_utils_ import clean_path, show_args
else:
    from ._script_utils_ import clean_path, show_args
    

def main(sys_args: List[str] = None):
    """
    Returns:
        - paths: List[str], paths
        - type_size: Dict[str, int], file type and size
    """
    # make and parse args
    args_paser = argparse.ArgumentParser(description = 'count files with specific suffix or sub-string in name')
    args_paser.add_argument('-i', '--input', type=str, default='.',
                            help='input/src files path or dir path. Default is %(default)s.')
    args_paser.add_argument('-t', '--type', type = str, default=[],
                            help='format of files to count, splited by ",". Default is %(default)s')
    args_paser.add_argument('-n', '--name', type = str, default='',
                            help='sub-string of name of files to remove. Default is %(default)s')
    args_paser.add_argument('-r', '--recursive', action='store_true', default=False,
                            help='FLAG, recursive search. Default is %(default)s.')
    args_paser.add_argument('--enable-case', action='store_true', default=False,
                            help='FLAG, ignore case when counting. Default is %(default)s.')
    args = args_paser.parse_args(sys_args)
    
    # process args
    args.type = args.type.split(',') if args.type else args.type
    args.input = clean_path(args.input)
    show_args(args, ['input', 'type', 'name', 'recursive', 'enable_case'])
    
    # get input paths
    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = get_paths_with_extension(args.input, args.type,
                                        args.recursive, args.name)
    
    # copy
    type_size = {}  # type: Dict[str, List[int, int]]
    for path in tqdm(paths, leave=False):
        try:
            file_type = os.path.splitext(path)[-1][1:]
            if not args.enable_case:
                file_type = file_type.lower()
            if file_type not in type_size:
                type_size[file_type] = [1, os.path.getsize(path)]
            else:
                type_size[file_type][0] += 1
                type_size[file_type][1] += os.path.getsize(path)
        except:
            put_err(f'can not get size of {path}, skip')
    
    # output result
    total_file, total_size = len(paths), sum([i[-1] for i in type_size.values()])
    for file_type, info in type_size.items():
        count, size = info
        print(f'{file_type:10s}: {format_file_size(size):10s} ({size:10d} bytes) ({size/total_size:2.2%} of total size) ({count:10d} files) ({count/total_file:2.2%} of total files)')
    print(f'\ntotal files: {total_file}\ntotal size: {format_file_size(total_size)} ({size} bytes)')
        
    return paths, type_size
    

# # dev code

if __name__ == "__main__":
    main()