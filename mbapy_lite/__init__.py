'''
Author: BHM-Bob 2262029386@qq.com
Date: 2024-10-17 10:11:44
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2024-10-17 16:26:45
Description: some helpful python scripts in plot, stats and deeplearning
'''
import os

# os.environ.__getitem__(self, key), no default value
if 'MBAPY_FAST_LOAD' not in os.environ:
    os.environ['MBAPY_FAST_LOAD'] = 'False'
    
if os.environ['MBAPY_FAST_LOAD'] == 'False':
    from . import base, file, plot, stats, web
    
from .__version__ import (__author__, __author_email__, __build__,
                          __description__, __license__, __title__, __url__,
                          __version__)


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