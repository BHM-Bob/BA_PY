'''
Date: 2023-10-12 21:35:05
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-10-17 13:56:55
Description: just need run on a success
'''

import numpy as np
import pygame as pg
from objprint import op

from mbapy.game import BaseInfo, Rect, Size, Sur


class CannotParse:
    def __init__(self, i = 0) -> None:
        self.i = i
        self.s = 'cannot parse'
        
class Parseable(BaseInfo):
    def __init__(self, i = 0) -> None:
        self.i = i
        self.s = 'parseable'
        self.cannot = CannotParse()
        self.list = [1, '2', {(3, ): 4}, CannotParse(i), {5: CannotParse(i)}]
        self.dict = {'k1': [1, '2', CannotParse(i), {3: 4}], 'k2': CannotParse(i), 'k3': [1, '2', {3: 4}]}
        
class Parseable2(BaseInfo):
    def __init__(self, i = 0) -> None:
        self.arr = np.arange(1000).reshape(1, 20, 50).astype(np.float32)
        self.sur = Sur('sur', pg.Surface((200, 300)), Rect(0, 0, 200, 300))
        self.rect = pg.Rect(0, 0, 200, 300)
        self.test = [[Parseable(i)],
                     [Parseable(i), CannotParse(i)],
                     Parseable(i)]
        self.test2 = [[Parseable(i)],
                     [Parseable(i), CannotParse(i)],
                     CannotParse(i)]
        
def test_BaseInfo():
    p = Parseable2(1)
    d = p.to_dict(True, True, True)
    p.to_json('./data_tmp/test_game.json')
    op(d)
    p2 = Parseable2(2)
    op(p2.from_json('./data_tmp/test_game.json').to_dict())
    return d
    
if __name__ == '__main__':
    test_BaseInfo()