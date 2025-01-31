'''
Date: 2025-01-25 17:21:51
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2025-01-31 10:20:03
Description: 
'''
import argparse
import os
import shutil
from typing import Callable, Dict, List, Optional, Tuple, Union

os.environ['MBAPY_AUTO_IMPORT_TORCH'] = 'False'
os.environ['MBAPY_FAST_LOAD'] = 'True'

from mbapy.base import import_file_as_package, put_err, put_log
from mbapy.file import opts_file
from mbapy.scripts._script_utils_ import Command, clean_path, excute_command


class script(Command):
    def __init__(self, args: argparse.Namespace, printf=print) -> None:
        super().__init__(args, printf)

    @staticmethod
    def make_args(args: argparse.ArgumentParser):
        args.add_argument('path', type = str,
                          help="python script file path to install.")
        args.add_argument('-F', '--force', default=False, action='store_true',
                          help="force to install the script if it has already installed.")
        return args
    
    def process_args(self):
        self.args.path = clean_path(self.args.path, True)
        # check file exist and is a python script file
        if not os.path.isfile(self.args.path):
            put_err(f"The file {self.args.path} is not exist.", _exit=True)
        if not self.args.path.endswith('.py'):
            put_err(f"The file {self.args.path} is not a python script file.", _exit=True)
        # check if has already installed
        self.file_name = os.path.basename(self.args.path)
        self.script_name = os.path.splitext(self.file_name)[0]
        self.dist_path = os.path.expanduser(f'~/.mbapy/scripts/{self.file_name}')
        os.makedirs(os.path.dirname(self.dist_path), exist_ok=True)
        if os.path.isfile(self.dist_path):
            if not self.args.force:
                put_err(f"The script {self.file_name} has already installed in ~/.mbapy/scripts, please remove it first.", _exit=True)
            put_log(f"The script {self.file_name} has already installed in {self.dist_path}, install with force flag to overwrite it.")

    def main_process(self):
        # import_file_as_package
        put_log(f'Loading the script {self.file_name} as a package...')
        pkg = import_file_as_package(self.args.path)
        if pkg is None:
            put_err(f"Failed to load the script {self.args.path}.", _exit=True)
        if not hasattr(pkg, 'main') and not callable(pkg.main):
            put_err(f"The script {self.file_name} does not have a main function.", _exit=True)
        # get and print kpg info from pkg.__INFO__
        if not hasattr(pkg, '__INFO__'):
            pkg.__INFO__ = {}
        pkg.__INFO__.setdefault('name', self.script_name)
        pkg.__INFO__.setdefault('script name', self.script_name)
        pkg.__INFO__.setdefault('exec_names', [pkg.__INFO__['name']])
        info_keys = ['version', 'brief', 'detailed'] # 'version' is the additional info part for user-script
        [pkg.__INFO__.setdefault(k, 'unknown') for k in info_keys]
        [print(f'{k:s}: {v}') for k, v in pkg.__INFO__.items()]
        # check if has exec_names conflicts with mbapy.scripts
        from mbapy.scripts._main_ import exec2script
        for n in exec2script:
            if n in pkg.__INFO__['exec_names']:
                put_log(f'Installing script {self.file_name} with exec_names {pkg.__INFO__["exec_names"]} conflicts with mbapy.scripts.{exec2script[n]}, will overwrite it when call.')
        # remind version change if this is force install bu user
        if os.path.isfile(self.dist_path):
            pkg_old = import_file_as_package(self.dist_path, f'{pkg.__INFO__["name"]}_old')
            if pkg_old is None:
                put_err(f'Failed to import the old version of the script {self.file_name} from {self.dist_path}.', _exit=True)
            if not hasattr(pkg_old, '__INFO__'):
                pkg_old.__INFO__ = {}
            ver_old = pkg_old.__INFO__.get('version', 'unknown')
            put_log(f'Installing version {pkg.__INFO__["version"]}, the old version is {ver_old}.')
        # update mbapy-cli-scripts-list.json in ~/.mbapy/scripts
        scripts_list_path = os.path.expanduser(f'~/.mbapy/scripts/mbapy-cli-scripts-list.json')
        if not os.path.isfile(scripts_list_path):
            scripts_lst = {}
        else:
            scripts_lst = opts_file(scripts_list_path, way='json')
        scripts_lst[self.file_name] = pkg.__INFO__
        opts_file(scripts_list_path, 'w', way='json', data=scripts_lst)
        put_log(f"The script {self.file_name}'s info has been added to mbapy-cli-scripts-list.json in {scripts_list_path}.")
        # copy script file to ~/.mbapy/scripts
        shutil.copy(self.args.path, os.path.expanduser(f'~/.mbapy/scripts/{os.path.basename(self.args.path)}'))
        put_log(f"The script {self.file_name} has been installed into {self.dist_path} successfully.")

_str2func = {
    'script': script,
}


def main(sys_args: List[str] = None):
    args_paser = argparse.ArgumentParser()
    subparsers = args_paser.add_subparsers(title='subcommands', dest='sub_command')
    script_args = script.make_args(subparsers.add_parser('script', description='install a script file into mbapy-cli'))

    excute_command(args_paser, sys_args, _str2func)

if __name__ == "__main__":
    # dev code, MUST COMMENT OUT BEFORE RELEASE
    # main('script mbapy/scripts/scihub.py -F'.split())
    
    main()