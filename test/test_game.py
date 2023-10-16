'''
Date: 2023-10-12 21:35:05
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-10-16 22:53:58
Description: just need run on a success
'''

import numpy as np
import pygame as pg
from objprint import op

from mbapy.game import BaseInfo, Rect, Size, Sur


class CannotParse:
    def __init__(self) -> None:
        self.i = 0
        self.s = 'cannot parse'
        
class Parseable(BaseInfo):
    def __init__(self) -> None:
        self.i = 1
        self.s = 'parseable'
        self.cannot = CannotParse()
        self.list = [1, '2', {3: 4}, CannotParse(), {5: CannotParse()}]
        self.dict = {'k1': [1, '2', CannotParse(), {3: 4}], 'k2': CannotParse(), 'k3': [1, '2', {3: 4}]}
        
class Parseable2(BaseInfo):
    def __init__(self) -> None:
        self.arr = np.arange(1000).reshape(1, 20, 50).astype(np.float32)
        self.sur = Sur('sur', pg.Surface((200, 300)), Rect(0, 0, 200, 300))
        self.rect = pg.Rect(0, 0, 200, 300)
        self.test = [[Parseable()],
                     [Parseable(), CannotParse()],
                     Parseable()]
        self.test2 = [[Parseable()],
                     [Parseable(), CannotParse()],
                     CannotParse()]
        
def test_BaseInfo():
    p = Parseable2()
    d = p.to_dict(True, True, True)
    p.to_json('./data_tmp/test_game.json')
    op(d)
    op(p.from_dict(d).to_dict())
    return d
    
if __name__ == '__main__':
    test_BaseInfo()