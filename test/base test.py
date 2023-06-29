'''
Author: BHM-Bob 2262029386@qq.com
Date: 2023-05-05 21:33:36
LastEditors: BHM-Bob
LastEditTime: 2023-05-05 22:05:16
Description: 
'''

import sys

sys.path.append(r'../../../')

import base

class Test4BaseAutoParse:
    @base.autoparse
    def __init__(self, a, b = 1, c = 2, *args, **kwargs) -> None:
        pass
    
a = Test4BaseAutoParse(1, 2, 3, d = 4, c = 3, e = 5)
pass