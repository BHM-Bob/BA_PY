'''
Date: 2023-10-12 21:35:05
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-10-12 21:39:38
Description: just need run on a success
'''

from objprint import op

from mbapy.game import BaseInfo


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
        self.dict = {'k1': [1, '2', CannotParse(), {3: 4}], 'k2': CannotParse()}
        
def test_BaseInfo():
    p = Parseable()
    d = p.to_dict(True, True)
    op(d)
    return d
    
if __name__ == '__main__':
    test_BaseInfo()