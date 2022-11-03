
import time
from functools import wraps

import numpy as np


def TimeCosts(runTimes:int = 1):
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

def rand_choose_times(choices_range:list[int] = [0,10], times:int = 100):
    randSeq = np.random.randint(low = choices_range[0], high = choices_range[1]+1, size = [times])
    count = [ np.sum(np.equal(randSeq,i)) for i in range(choices_range[0],choices_range[1]+1) ]
    return np.argmax(np.array(count))