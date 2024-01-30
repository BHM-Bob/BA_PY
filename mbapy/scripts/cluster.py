import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

os.environ['MBAPY_AUTO_IMPORT_TORCH'] = 'False'
os.environ['MBAPY_FAST_LOAD'] = 'True'
from mbapy import base, file, stats

if __name__ == '__main__':
    from mbapy.scripts._script_utils_ import clean_path
else:
    from ._script_utils_ import clean_path
    
    

_str2func = {
}


# if __name__ == '__main__':
#     # dev code
#     from mbapy.game import BaseInfo
#     calcu_mw_of_mutations(BaseInfo(seq = 'Fmoc-Cys(Acm)-Val-Asn(Trt)', out = '.',
#                                    max_repeat = 1, weight = '', mass = False))
#     calcu_mw_of_mutations(BaseInfo(seq = 'Fmoc-Cys(Acm)-Val-Asn(Trt)', out = '.',
#                                    max_repeat = 1, weight = '', mass = True))

def main(sys_args: List[str] = None):
    args_paser = argparse.ArgumentParser()
    subparsers = args_paser.add_subparsers(title='subcommands', dest='sub_command')
    
    sub_val_args = subparsers.add_parser('subval', aliases = ['sb'], description='calcu SPPS substitution value for a release test of resin.')
    sub_val_args.add_argument('-a', '-A', '--absorbance', '--Absorbance', type = str,
                              help='Absorbance (OD value), input as 0.503,0.533')
    sub_val_args.add_argument('-m', '-w', '--weight', type = str,
                              help='resin wight (mg), input as 0.165,0.155')
    sub_val_args.add_argument('-c', '--coff', default = 16.4, type = float,
                              help='coff, default is 16.4')
    
    args = args_paser.parse_args(sys_args)
    
    if args.sub_command in _str2func:
        print(f'excuting command: {args.sub_command}')
        _str2func[args.sub_command](args)
    else:
        base.put_err(f'no such sub commmand: {args.sub_command}')

if __name__ == "__main__":
    main()