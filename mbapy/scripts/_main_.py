'''
Date: 2024-01-08 21:31:52
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2025-01-29 09:32:31
FilePath: \BA_PY\mbapy\scripts\_main_.py
Description: 
'''
import importlib
import os
import sys
from typing import Dict

os.environ['MBAPY_FAST_LOAD'] = 'True'
os.environ['MBAPY_AUTO_IMPORT_TORCH'] = 'False'

from mbapy.base import (get_fmt_time, get_storage_path, import_file_as_package,
                        put_log)
from mbapy.file import opts_file

load_exec2script = lambda x: {n: x_i['script name'] for x_i in x.values() for n in x_i['exec_names']}
# load scripts list
scripts_info = opts_file(get_storage_path('mbapy-cli-scripts-list.json'), way = 'json')
exec2script: Dict[str, str] = load_exec2script(scripts_info)
# load extention scripts list
EXT_SCRIPT_PATH = os.path.expanduser(f'~/.mbapy/scripts/mbapy-cli-scripts-list.json')
if os.path.isfile(EXT_SCRIPT_PATH):
    ext_scripts_info = opts_file(EXT_SCRIPT_PATH, way = 'json')
else:
    ext_scripts_info = {}
ext_exec2script: Dict[str, str] = load_exec2script(ext_scripts_info)
# merge scripts list
scripts_info.update(ext_scripts_info)
exec2script.update(ext_exec2script)


def print_version_info():
    import mbapy
    print('mbapy python package command-line tools')
    print('mbapy version: ', mbapy.__version__, ', build: ', mbapy.__build__)
    print('mbapy author: ', mbapy.__author__, ', email: ', mbapy.__author_email__)
    print('mbapy url: ', mbapy.__url__, ', license: ', mbapy.__license__)
    
def print_help_info():
    help_info = """
            usage-1: mbapy-cli [-h] [-l | -i]
            options:
            -h, --help  show this help message and exit
            -l, --list  print scripts list
            -i, --info  print scripts info
            usage-2: mbapy-cli [sub-scripts-name] [args] [-h]
            options:
            sub-scripts-name  name of scripts in mbapy.scripts
            args  args for sub-scripts
            -h, --help  show this help message and exit
            """
    print(help_info)
    
def print_scripts_list():
    for idx, script in enumerate(scripts_info):
        print(f'scripts {idx:3d}: {scripts_info[script]["name"]}')
        print(f'script file name: {scripts_info[script]["script name"]}')
        print(f'exec names: {", ".join(scripts_info[script]["exec_names"])}')
        print(scripts_info[script]['brief'])
        print('-'*100)

def print_scripts_info():
    for idx, script in enumerate(scripts_info):
        print(f'scripts {idx:3d}: {scripts_info[script]["name"]}')
        print(f'script file name: {scripts_info[script]["script name"]}')
        print(f'exec names: {", ".join(scripts_info[script]["exec_names"])}')
        print(scripts_info[script]['brief'])
        print(scripts_info[script]['detailed'])
        print('-'*100)
        
def exec_scripts():
    import mbapy

    # remind overwrite num
    overwite_num = len(set(ext_exec2script.keys()) & set(exec2script.keys()))
    if overwite_num > 0:
        put_log(f'Overwrite {overwite_num} script(s) from {EXT_SCRIPT_PATH}.')
    # check --pause-after-exec argumet
    pause_after_exec = '--pause-after-exec' in sys.argv
    if pause_after_exec:
        sys.argv.remove('--pause-after-exec')
    # check and exec scripts NOTE: DO NOT use exec
    if sys.argv[1] in ext_exec2script:
        ext_script_path = os.path.expanduser(f'~/.mbapy/scripts/{ext_exec2script[sys.argv[1]]}.py')
        put_log(f'Loading external script: {sys.argv[1]} from {ext_script_path}')
        script = import_file_as_package(ext_script_path)
    else:
        script_name = exec2script[sys.argv[1]]
        script = importlib.import_module(f'.{script_name}', 'mbapy.scripts')
    script.main(sys.argv[2:])
    # pause if --pause-after-exec
    if pause_after_exec:
        os.system('pause') # avoid cmd window close immediately
    
def main():  
    def _handle_unkown():
        print(f'mbapy-cli: unkown scripts: {sys.argv[1]} and args: {", ".join(sys.argv[2:])}, SKIP.')
        print('bellow are all scripts list:\n\n')
        print_scripts_list()
        
    if len(sys.argv) == 1:
        print_version_info()
    elif len(sys.argv) == 2:
        if sys.argv[1] in ['-l', '--list']:
            print_scripts_list()
        elif sys.argv[1] in ['-i', '--info']:
            print_scripts_info()
        elif sys.argv[1] in ['-h', '--help']:
            print_help_info()
        elif sys.argv[1] in exec2script:
            # exec scripts with only ZERO arg
            exec_scripts()
        elif os.path.exists(sys.argv[1]) and sys.argv[1].endswith('.mpss'):
            # load MbaPy Script Session file and exec
            from mbapy.scripts._script_utils_ import Command
            print(f'loading session from file: {sys.argv[1]}')
            Command(None).exec_from_session(sys.argv[1])
            os.system('pause') # avoid cmd window close immediately
        else:
            _handle_unkown()
    else:
        if sys.argv[1] in exec2script:
            # exec scripts
            exec_scripts()
        else:
            _handle_unkown()
    # exit
    print(f'\nmbapy-cli: exit at {get_fmt_time("%Y-%m-%d %H:%M:%S.%f")}')
            

if __name__ == '__main__':
    # dev code
    # sys.argv = 'mbapy-cli scihub -h'.split()
    
    main()