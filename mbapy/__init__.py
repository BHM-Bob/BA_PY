'''
Author: BHM-Bob 2262029386@qq.com
Date: 2022-11-01 22:16:49
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-06-30 12:21:54
Description: some helpful python scripts in plot, stats and deeplearning
'''
from . import base, file, plot, web, stats
from .__version__ import (
    __author__,
    __author_email__,
    __build__,
    __description__,
    __license__,
    __title__,
    __version__,
    __url__,
)

try:
    import torch
    from mbapy import dl_torch as dl_torch
except:
    pass
    # print('no torch available')

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