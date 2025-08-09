'''
Date: 2024-07-16 13:43:25
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2025-08-09 09:18:53
Description: 
'''

import argparse
import os
from typing import Dict, List

if __name__ == '__main__':
    from mbapy.file import format_file_size, get_paths_with_extension
    from mbapy.scripts._script_utils_ import _print, clean_path, show_args
else:
    from ..file import format_file_size, get_paths_with_extension
    from ._script_utils_ import _print, clean_path, show_args
    

def main(sys_args: List[str] = None):
    args_paser = argparse.ArgumentParser(description = 'delete files with specific suffix or sub-string in name')
    args_paser.add_argument('-t', '--type', type = str, nargs='+', default=[],
                            help='format of files to remove, splited by ",". Default is %(default)s')
    args_paser.add_argument('-n', '--name', type = str, default='',
                            help='sub-string of name of files to remove. Default is %(default)s')
    args_paser.add_argument('-i', '--input', type=str, default='.',
                            help='files path or dir path.')
    args_paser.add_argument('-o', '--output', type=str, default=None,
                            help='output list txt file path or dir path. Default is %(default)s.')
    args_paser.add_argument('-r', '--recursive', action='store_true', default=False,
                            help='FLAG, recursive search. Default is %(default)s.')
    args_paser.add_argument('-s', '--sum', action='store_true', default=False,
                            help='FLAG, sum size of files. Default is %(default)s.')
    args = args_paser.parse_args(sys_args)
    
    # process IO path
    args.input = clean_path(args.input)
    if args.output is not None:
        if os.path.isdir(args.output):
            args.output = os.path.join(args.output, '__mbapy_scripts_find_file_result.txt')
        args.output = clean_path(args.output)
    f_handle = open(args.output, 'w') if args.output is not None else None
    # show args
    show_args(args, ['type', 'name', 'input', 'output', 'recursive'])
    
    paths = get_paths_with_extension(args.input, args.type,
                                     args.recursive, args.name, c_version=True)
    _print(f'files finded: {len(paths)} in dir: {args.input}', f_handle)
    # show info
    if args.sum:
        total_size = 0
        for path in paths:
            total_size += os.path.getsize(path)
        _print(f'total files size: {format_file_size(total_size)}', f_handle)
    else:
        for path in paths:
            info_str = ', '.join([path[len(str(args.input)):], format_file_size(os.path.getsize(path))])
            _print(info_str, f_handle)
    _print(f'files finded: {len(paths)} in dir: {args.input}', f_handle) # print again in bottom
    return paths


if __name__ == "__main__":
    main()