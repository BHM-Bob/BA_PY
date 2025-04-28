'''
Date: 2024-02-05 15:12:32
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2025-02-15 11:46:08
Description: 
'''

import argparse
import os
import shutil
from typing import Dict, List

from tqdm import tqdm

if __name__ == '__main__':
    from mbapy.base import put_err
    from mbapy.file import get_paths_with_extension
    from mbapy.scripts._script_utils_ import clean_path, show_args
else:
    from ..base import put_err
    from ..file import get_paths_with_extension
    from ._script_utils_ import clean_path, show_args
    

def main(sys_args: List[str] = None):
    # set and parse args
    args_paser = argparse.ArgumentParser(description = 'move files with specific suffix or sub-string in name to root dir')
    args_paser.add_argument('-t', '--type', type = str, nargs='+', default=[],
                            help='format of files to remove, splited by ",". Default is %(default)s')
    args_paser.add_argument('-n', '--name', type = str, default='',
                            help='sub-string of name of files to remove. Default is %(default)s')
    args_paser.add_argument('-i', '--input', type=str, default='.',
                            help='files path or dir path. Default is %(default)s.')
    args_paser.add_argument('-r', '--recursive', action='store_true', default=False,
                            help='FLAG, recursive search. Default is %(default)s.')
    args_paser.add_argument('-j', '--join-str', type=str, default=' ',
                            help='join string of file name. Default is %(default)s.')
    args_paser.add_argument('-rmd', '--remove-dir', action='store_true', default=False,
                            help='FLAG, remove empty dir after move files. Default is %(default)s.')
    args = args_paser.parse_args(sys_args)
    # process args
    args.input = clean_path(args.input)
    show_args(args, ['type', 'input', 'name', 'recursive', 'join_str', 'remove_dir'])
    
    paths = get_paths_with_extension(args.input, args.type,
                                     args.recursive, args.name)
    print(f'files to move: {len(paths)} (include root dir to skip)')
    # move files
    root = str(args.input)
    empty_dirs = []
    for path in tqdm(paths):
        try:
            path = str(clean_path(path))
            dist = path.replace(root, '').replace(os.path.sep, args.join_str)
            dist = os.path.join(root, dist)
            if root != dist:
                shutil.move(path, dist)
                if os.path.dirname(path) not in empty_dirs:
                    empty_dirs.append(os.path.dirname(path))
        except Exception as e:
            put_err(f'Error: {e}. Can not move {path}, skip')
    # remove empty dir
    if args.remove_dir:
        for empty_dir in empty_dirs:
            try:
                os.rmdir(empty_dir)
            except Exception as e:
                put_err(f'Error: {e}. Can not remove empty dir {empty_dir}, skip')
    # return paths
    return paths
    
    

# dev code
# main(['-i', r'E:\\', '-t', 'JPG'])


if __name__ == "__main__":
    main()