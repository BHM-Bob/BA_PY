'''
Date: 2024-01-11 11:23:49
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2024-01-12 16:11:47
FilePath: \BA_PY\mbapy\scripts\reviz.py
Description: 
'''
import argparse
import os
from pathlib import Path
from typing import Dict, List

os.environ['MBAPY_AUTO_IMPORT_TORCH'] = 'False'
os.environ['MBAPY_FAST_LOAD'] = 'True'
from mbapy.dl_torch.utils import launch_visdom, re_viz_from_json_record

if __name__ == '__main__':
    from mbapy.scripts._script_utils_ import clean_path
else:
    from ._script_utils_ import clean_path
    

def main(sys_args: List[str] = None):
    args_paser = argparse.ArgumentParser()

    args_paser.add_argument('-i', '--input', type = str, help='input file')
    
    args = args_paser.parse_args(sys_args)
    
    args.input = clean_path(args.input)
    print(f'get arg: input: {args.input}')
    
    if not os.path.exists(args.input):
        print(f'input file not exists: {args.input}, skip')
        return None
    if os.path.isdir(args.input):
        for path in os.listdir(args.input):
            if 'record' in path and path.endswith('.json'):
                print(f'find file with record and is json: {path}')
                args.input = path
                break
            elif path.endswith('.json'):
                print(f'find json file: {path}')
                args.input = path
                break
    
    viz_env = Path(args.input.parent.name)
    print(f'starting vizdom service(Env={viz_env})')
    launch_visdom(viz_env)
    print(f'reviz with file {args.input}')    
    
    re_viz_from_json_record(args.input)

if __name__ == "__main__":
    main()