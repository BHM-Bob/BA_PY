'''
Date: 2024-01-12 16:06:35
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2024-06-30 17:21:31
FilePath: \BA_PY\mbapy\scripts\_script_utils_.py
Description: 
'''
import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Union

if __name__ == '__main__':
    from mbapy.base import check_parameters_path, parameter_checker, put_err
    from mbapy.file import opts_file
else:
    from ..base import check_parameters_path, parameter_checker, put_err
    from ..file import opts_file


def clean_path(path: str):
    return Path(path.replace('"', '').replace("'", '')).resolve()

def _print(content: str, f, verbose = True):
    if f is not None:
        f.write(content+'\n')
    if verbose:
        print(content)

def show_args(args, args_name: List[str], printf = print):
    printf('')
    for arg_name in args_name:
        printf(f'get arg: {arg_name}: {getattr(args, arg_name)}')
    printf('')

class Command:
    def __init__(self, args: argparse.Namespace, printf = print) -> None:
        self.args = args
        self.printf = printf
        
    def process_args(self):
        pass
    
    def main_process(self):
        pass
        
    def excute(self):
        self.process_args()
        show_args(self.args, list(self.args.__dict__.keys()), self.printf)
        return self.main_process()
    
def excute_command(args_paser: argparse.ArgumentParser, sys_args: List[str],
                   _str2func: Dict[str, callable]):
    args = args_paser.parse_args(sys_args)
    
    if args.sub_command in _str2func:
        try:
            if issubclass(_str2func[args.sub_command], Command):
                _str2func[args.sub_command](args).excute()
            else:
                _str2func[args.sub_command](args)
        except:
            if callable(_str2func[args.sub_command]):
                _str2func[args.sub_command](args)
    else:
        put_err(f'no such sub commmand: {args.sub_command}')
        
        