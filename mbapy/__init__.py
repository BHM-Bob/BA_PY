'''
Author: BHM-Bob 2262029386@qq.com
Date: 2022-11-01 22:16:49
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2024-05-20 21:22:29
Description: some helpful python scripts in plot, stats and deeplearning
'''
import os
    
if os.environ.get('MBAPY_FAST_LOAD', 'False') == 'False':
    from . import base, file, paper, plot, stats, web, sci_instrument
    # load frequently used functions in sub-packages
    from .base import put_log, put_err
    from .file import get_paths_with_extension, opts_file
    from .game import BaseInfo
    from .plot import get_palette, save_show
    from .web import TaskPool
    
    
from .__version__ import (__author__, __author_email__, __build__,
                          __description__, __license__, __title__, __url__,
                          __version__)

try:
    if os.environ.get('MBAPY_AUTO_IMPORT_TORCH', 'False') == 'True':
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