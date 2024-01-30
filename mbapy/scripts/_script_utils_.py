'''
Date: 2024-01-12 16:06:35
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2024-01-25 12:32:38
FilePath: \BA_PY\mbapy\scripts\_script_utils_.py
Description: 
'''
from typing import List, Dict
from pathlib import Path

def clean_path(path: str):
    return Path(path.replace('"', '').replace("'", '')).resolve()

def show_args(args, args_name: List[str]):
    print('')
    for arg_name in args_name:
        print(f'get arg: {arg_name}: {getattr(args, arg_name)}')
    print('')