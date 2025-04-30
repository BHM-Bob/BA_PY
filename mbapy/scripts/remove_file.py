'''
Date: 2024-02-05 15:12:32
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2025-02-15 11:49:49
Description: 
'''

import argparse
import os
import platform
from typing import Dict, List

from tqdm import tqdm

from mbapy.base import put_err
from mbapy.file import get_paths_with_extension

if __name__ == '__main__':
    from mbapy.scripts._script_utils_ import clean_path, show_args
else:
    from ._script_utils_ import clean_path, show_args
    

def main(sys_args: List[str] = None):
    args_paser = argparse.ArgumentParser(description = 'delete files with specific suffix or sub-string in name')
    args_paser.add_argument('-t', '--type', type = str, nargs='+', default=[],
                            help='format of files to remove, splited by ",". Default is %(default)s')
    args_paser.add_argument('-n', '--name', type = str, default='',
                            help='sub-string of name of files to remove. Default is %(default)s')
    args_paser.add_argument('-i', '--input', type=str,
                            help='files path or dir path.')
    args_paser.add_argument('-r', '--recursive', action='store_true', default=False,
                            help='FLAG, recursive search. Default is %(default)s.')
    args = args_paser.parse_args(sys_args)
    
    args.input = clean_path(args.input)
    show_args(args, ['type', 'input', 'name', 'recursive'])
    
    paths = get_paths_with_extension(args.input, args.type,
                                     args.recursive, args.name)
    print(f'files to remove: {len(paths)}')
    # manual confirm
    inputs = input('Are you sure to delete these files? (y/n) ')
    if inputs.lower() != 'y':
        print('cancel delete files')
        return []
    # delete files
    for path in tqdm(paths):
        try:
            if platform.system().lower() == 'windows':
                os.system(f'attrib -r "{path}"')
            elif platform.system().lower() == 'linux':
                os.system(f'chmod 666 "{path}"')
            os.remove(path)
        except Exception as e:
            put_err(f'Error: {e}. Can not delete {path}, skip')
    return paths
    
    

# dev code
# main(['-i', r'E:\\', '-t', 'JPG'])


if __name__ == "__main__":
    main()