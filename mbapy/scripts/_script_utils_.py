'''
Date: 2024-01-12 16:06:35
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2025-01-27 23:01:24
FilePath: \BA_PY\mbapy\scripts\_script_utils_.py
Description: 
'''
import argparse
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Union

if __name__ == '__main__':
    from mbapy.base import check_parameters_path, parameter_checker, put_err
    from mbapy.file import opts_file
    from mbapy.game import BaseInfo
else:
    from ..base import check_parameters_path, parameter_checker, put_err
    from ..file import opts_file
    from ..game import BaseInfo


def clean_path(path: str, return_str: bool = False):
    path = Path(path.replace('"', '').replace("'", '')).resolve()
    if not return_str:
        return path
    return str(path)

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

class Command(BaseInfo):
    def __init__(self, args: argparse.Namespace, printf = print) -> None:
        self.args = args
        self.printf = printf
        self._pickle_except_list = []
        
    def process_args(self):
        pass
    
    def main_process(self):
        pass
        
    def excute(self):
        self.process_args()
        show_args(self.args, list(self.args.__dict__.keys()), self.printf)
        return self.main_process()
    
    def save_session(self, module_name: str, module_path: str = 'mbapy.scripts', path: str = os.curdir):
        raise NotImplementedError('save_session is not implemented')
        if not Path(path).parent.exists():
            os.makedirs(Path(path).parent, exist_ok=True)
        session = {'__module_path__': module_path, '__module_name__': module_name, '__cmd_name__': self.__class__.__name__}
        for k,v in self.args.__dict__.items():
            if k not in self._pickle_except_list:
                session[k] = v
        opts_file(path, 'wb', way = 'pkl', data = session)
        
    @parameter_checker(path = check_parameters_path)
    def load_session(self, path: str):
        raise NotImplementedError('load_session is not implemented')
        return opts_file(path, 'rb', way = 'pkl')
        
    def exec_from_session(self, session: Union[str, Dict[str, Any]]):
        raise NotImplementedError('exec_from_session is not implemented')
        if isinstance(session, str) and check_parameters_path(session):
            session = self.load_session(session)
        
    
def excute_command(args_paser: argparse.ArgumentParser, sys_args: List[str],
                   _str2func: Dict[str, callable]):
    args = args_paser.parse_args(sys_args)
    
    if args.sub_command in _str2func:
        cmd = _str2func[args.sub_command]
        try:
            # if is inherite from Command, excute it
            if isinstance(cmd, type) and issubclass(cmd, Command):
                cmd(args).excute()
            # if is a function, excute it
            elif callable(cmd):
                cmd(args)
            # else, just try
            else:
                cmd(args)
        except Exception as e:
            traceback.print_exception(type(e), e, e.__traceback__)
    else:
        put_err(f'no such sub commmand: {args.sub_command}')

