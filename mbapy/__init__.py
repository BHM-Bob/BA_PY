'''
Author: BHM-Bob 2262029386@qq.com
Date: 2022-11-01 22:16:49
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-10-19 22:01:01
Description: some helpful python scripts in plot, stats and deeplearning
'''
import os

# os.environ.__getitem__(self, key), no default value
if 'MBAPY_AUTO_IMPORT_TORCH' not in os.environ:
    os.environ['MBAPY_AUTO_IMPORT_TORCH'] = 'True'
if 'MBAPY_FAST_LOAD' not in os.environ:
    os.environ['MBAPY_FAST_LOAD'] = 'False'
    
if os.environ['MBAPY_FAST_LOAD'] == 'False':
    from . import base, file, paper, plot, stats, web
    
from .__version__ import (__author__, __author_email__, __build__,
                          __description__, __license__, __title__, __url__,
                          __version__)

try:
    if 'MBAPY_AUTO_IMPORT_TORCH' in os.environ and\
        os.environ['MBAPY_AUTO_IMPORT_TORCH'] == 'True':
            import torch

            from mbapy import dl_torch as dl_torch
except:
    pass
    # print('no torch module available')


# def main():
#     pass
"""
var naming:

constant : CONSTANT_VAR_NAME
variable : var_name
func : func_name
global var: globalVarName
class : ClassName
"""