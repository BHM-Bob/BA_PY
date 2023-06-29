'''
Date: 2023-05-22 15:01:36
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-05-22 15:21:48
FilePath: /BA_PY/mbapy/test/stats/test_test.py
Description: 
'''
import mbapy.stats as ms
import pandas as pd
import numpy as np
import scipy

print(ms.get_interval(data = np.random.rand(16)))
print(ms.get_interval(data = [0, 8, 7, 3.9, 9, 4, 9]))
print(ms.get_interval(data = pd.Series(data = np.random.rand(16), name = 'a')))