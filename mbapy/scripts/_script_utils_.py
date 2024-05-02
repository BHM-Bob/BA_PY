'''
Date: 2024-01-12 16:06:35
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2024-05-02 21:29:19
FilePath: \BA_PY\mbapy\scripts\_script_utils_.py
Description: 
'''
import argparse
from typing import List, Dict
from pathlib import Path


if __name__ == '__main__':
    from mbapy.base import put_err
else:
    from ..base import put_err


def clean_path(path: str):
    return Path(path.replace('"', '').replace("'", '')).resolve()

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
        print(f'excuting command: {args.sub_command}')
        if issubclass(_str2func[args.sub_command], Command):
            _str2func[args.sub_command](args).excute()
        else:
            _str2func[args.sub_command](args)
    else:
        put_err(f'no such sub commmand: {args.sub_command}')
        
        