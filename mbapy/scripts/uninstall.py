'''
Date: 2025-01-31 10:19:44
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2025-01-31 10:40:20
Description: 
'''
import argparse
import os
from typing import List

from mbapy.base import put_err, put_log
from mbapy.file import opts_file
from mbapy.scripts._script_utils_ import Command, clean_path, excute_command


class script(Command):
    def __init__(self, args: argparse.Namespace, printf=print) -> None:
        super().__init__(args, printf)

    @staticmethod
    def make_args(args: argparse.ArgumentParser):
        args.add_argument('script_name', type=str,
                          help="Name of the installed script to uninstall.")
        return args

    def process_args(self):
        # Process script name to ensure it has .py extension
        self.script_name = self.args.script_name.strip()
        if not self.script_name.endswith('.py'):
            self.script_name += '.py'
        # Determine the installed script path
        self.dist_path = os.path.expanduser(f'~/.mbapy/scripts/{self.script_name}')
        # Check if the script exists unless force is specified
        if not os.path.isfile(self.dist_path):
            put_err(f"The script {self.script_name} is not installed.", _exit=True)

    def main_process(self):
        # Remove the script file if it exists
        if os.path.isfile(self.dist_path):
            try:
                os.remove(self.dist_path)
                put_log(f"Removed script file: {self.dist_path}")
            except Exception as e:
                put_err(f"Failed to remove {self.dist_path}: {str(e)}", _exit=True)
        else:
            put_log(f"Script file {self.script_name} not found, skipping deletion.")
        # Update the scripts list JSON
        scripts_list_path = os.path.expanduser('~/.mbapy/scripts/mbapy-cli-scripts-list.json')
        if os.path.isfile(scripts_list_path):
            scripts_lst = opts_file(scripts_list_path, way='json')
            if self.script_name in scripts_lst:
                del scripts_lst[self.script_name]
                opts_file(scripts_list_path, 'w', way='json', data=scripts_lst)
                put_log(f"Removed {self.script_name} from scripts list.")
            else:
                put_log(f"{self.script_name} not found in scripts list, skipping removal.")
        else:
            put_log("Scripts list file not found, skipping update.")
        put_log(f"Successfully uninstalled {self.script_name}.")

_str2func = {
    'script': script,
}

def main(sys_args: List[str] = None):
    args_parser = argparse.ArgumentParser()
    subparsers = args_parser.add_subparsers(title='subcommands', dest='sub_command')
    script.make_args(subparsers.add_parser('script', description='Uninstall a script file into mbapy-cli'))
    
    excute_command(args_parser, sys_args, _str2func)

if __name__ == "__main__":
    main()