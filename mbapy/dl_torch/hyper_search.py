'''
Author: BHM-Bob 2262029386@qq.com
Date: 2023-05-01 20:16:20
LastEditors: BHM-Bob
LastEditTime: 2023-05-01 20:17:32
Description:
'''
import math
import pickle
import time

import numpy as np
from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, hp, tpe

space = {
    "a": hp.uniform("a", -10, 10),
    "b": hp.uniform("b", -10, 10),
    "c": hp.uniform("c", -10, 10),
}
def func(a, b, c):
    return math.sin(a)*(b**2)/(a-c)
def objective(params):
    loss = func(**params)
    return {
        'loss': loss,
        'status': STATUS_FAIL if math.isnan(loss) else STATUS_OK,
        # -- store other results like this
        'eval_time': time.time(),
        'other_stuff': {'type': None, 'value': [0, 1, 2]},
        # -- attachments are handled differently
        'attachments':
            {'time_module': pickle.dumps(time.time)}
        }
trials = Trials()
best = fmin(objective,
    space=space,
    algo=tpe.suggest,
    max_evals=1000,
    trials=trials,
    rstate= np.random.default_rng(0))
print(best)