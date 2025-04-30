'''
Date: 2024-04-15 21:01:33
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2025-02-15 11:47:09
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
    args_paser.add_argument('-t', '--type', type = str, nargs='+', default=[],
                            help='format of files. Default is %(default)s')
    args_paser.add_argument('-n', '--name', type = str, default='',
                            help='sub-string of name of files to remove. Default is %(default)s')
    args_paser.add_argument('-r', '--recursive', action='store_true', default=False,
                            help='FLAG, recursive search. Default is %(default)s.')
    args_paser.add_argument('--enable-case', action='store_true', default=False,
                            help='FLAG, ignore case when counting. Default is %(default)s.')
    args_paser.add_argument('--sort-by-name', action='store_true', default=False,
                            help='FLAG, sort result by file name, if not set, sort by size. Default is %(default)s.')
    args = args_paser.parse_args(sys_args)
    
    # process args
    args.input = clean_path(args.input)
    show_args(args, ['input', 'type', 'name', 'recursive', 'enable_case', 'sort_by_name'])
    
    # get input paths
    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = get_paths_with_extension(args.input, args.type,
                                        args.recursive, args.name)
    
    # copy
    type_size, type_info = {}, []
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
    
    # sort result
    total_file, total_size = len(paths), sum([i[-1] for i in type_size.values()])
    for file_type, info in type_size.items():
        count, size = info
        type_info.append((file_type, size, count))
    if args.sort_by_name:
        type_info.sort(key=lambda x: x[0])
    else:
        type_info.sort(key=lambda x: x[1], reverse=True)
        
    # print result
    for file_type, size, count in type_info:
        print(f'{file_type:10s}: {format_file_size(size):15s} ({size:10d} bytes) ({size/total_size:6.2%} of total size) ({count:10d} files) ({count/total_file:6.2%} of total files)')
    print(f'\ntotal types: {len(type_size)}\ntotal files: {total_file}\ntotal size: {format_file_size(total_size)} ({total_size} bytes)')
        
    return paths, type_size
    

# # dev code

if __name__ == "__main__":
    main()