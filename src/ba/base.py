'''
Author: BHM-Bob 2262029386@qq.com
Date: 2022-10-19 22:46:30
LastEditors: BHM-Bob
LastEditTime: 2022-10-19 22:54:06
Description: 
'''
import time
from functools import wraps

import numpy as np


def TimeCosts( runTimes = 1 ):
    """
    @TimeCosts(9)
    def f(r):
        for i in range(1, 999999):
            r += (i if r % i % i == 0 else 1)
        return i
    print(f(8))
    """
    def Wrapper_0( func ):
        @wraps(func)
        def Wrapper_1(*args, **kwargs):
            t0, ret = time.time(), []
            for times in range(runTimes):
                t1 = time.time()
                ret.append(func(*args, **kwargs))
                print(f'{times:2d} | {func.__name__:s} used {time.time()-t1:10.3f}s')
            print(f'{func.__name__:s} used {time.time()-t0:10.3f}s in total, {(time.time()-t0)/runTimes:10.3f}s by mean')
            return ret
        return Wrapper_1
    return Wrapper_0

def RandChooce4Times(choicesRange = [0,10], times = 100):
    randSeq = np.random.randint(low = choicesRange[0], high = choicesRange[1]+1, size = [times])
    count = [ np.sum(np.equal(randSeq,i)) for i in range(choicesRange[0],choicesRange[1]+1) ]
    return np.argmax(np.array(count))