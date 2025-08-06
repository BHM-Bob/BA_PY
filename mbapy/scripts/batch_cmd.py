'''
Date: 2024-02-05 15:12:32
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2025-06-22 20:46:55
Description: 
'''

import argparse
import os
from typing import Dict, List

from mbapy.base import put_log
from mbapy.file import get_paths_with_extension
from tqdm import tqdm

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
    args_paser.add_argument('-c', '--cmd', type=str, default=None,
                            help='command to excute, such as "cp %%s ~/a.txt", can use "%%s" to pass path. Default is %(default)s.')
    args_paser.add_argument('-d', '--use-dir', action='store_true', default=False,
                            help='Whether pass path with it\'s dir. Default is %(default)s.')
    args_paser.add_argument('-cd', '--change-dir', action='store_true', default=False,
                            help='Whether cd to path\'s dir before excuting cmd. Default is %(default)s.')
    args = args_paser.parse_args(sys_args)
    
    args.input = clean_path(args.input)
    show_args(args, ['type', 'input', 'name', 'recursive', 'cmd', 'use_dir', 'change_dir'])
    
    paths = get_paths_with_extension(args.input, args.type,
                                     args.recursive, args.name)
    print(f'{len(paths)} file(s) matched.')
    # manual input cmd if is None
    if args.cmd is None:
        args.cmd = input('input commad:')
    # delete files
    for path in tqdm(paths, total=len(paths), desc='Excute command'):
        _dir = os.path.dirname(path)
        if '%s' in args.cmd:
            cmd = args.cmd.replace('%s', _dir if args.use_dir else path)
        else:
            cmd = f'{args.cmd}'
        if args.change_dir:
            cmd = f'cd {_dir} && {cmd}'
        put_log(f'Excuting: {cmd}')
        os.system(cmd)
    return paths


if __name__ == "__main__":
    main()