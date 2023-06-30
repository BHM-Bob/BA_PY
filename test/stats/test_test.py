'''
Date: 2023-05-22 15:01:36
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-06-30 18:50:28
FilePath: \BA_PY\test\stats\test_test.py
Description: 
'''
import pandas as pd
import numpy as np
import scipy
import statsmodels.api as sm

import unittest

import mbapy.stats as ms

print(ms.test.get_interval(data = np.random.rand(16)))
print(ms.test.get_interval(data = [0, 8, 7, 3.9, 9, 4, 9]))
print(ms.test.get_interval(data = pd.Series(data = np.random.rand(16), name = 'a')))

class TestGetInterval(unittest.TestCase):

    def test_mean_se(self):
        # Testing when mean and se are provided
        mean = 10
        se = 2
        confidence = 0.95
        interval, kwargs = ms.test.get_interval(mean=mean, se=se, confidence=confidence)
        self.assertEqual(len(interval), 2)
        self.assertLess(interval[0], interval[1])
        self.assertEqual(kwargs['loc'], mean)
        self.assertEqual(kwargs['scale'], se)

    def test_data_confidence(self):
        # Testing when data and confidence are provided
        data = [1, 2, 3, 4, 5]
        confidence = 0.99
        interval, kwargs = ms.test.get_interval(data=data, confidence=confidence)
        self.assertEqual(len(interval), 2)
        self.assertLess(interval[0], interval[1])
        self.assertEqual(kwargs['loc'], np.mean(data).item())
        self.assertEqual(kwargs['scale'], scipy.stats.sem(data))

    def test_se_none_data_none(self):
        # Testing when se is not provided and data is None
        mean = 5
        confidence = 0.9
        with self.assertRaises(AssertionError) as context:
            ms.test.get_interval(mean=mean, confidence=confidence)
        self.assertEqual(str(context.exception), 'se is None and data is None')

    def test_se_none_data(self):
        # Testing when se is None and data is provided
        se = None
        data = [1, 2, 3, 4, 5]
        confidence = 0.85
        interval, kwargs = ms.test.get_interval(data=data, confidence=confidence)
        self.assertEqual(len(interval), 2)
        self.assertLess(interval[0], interval[1])
        self.assertEqual(kwargs['loc'], np.mean(data).item())
        self.assertEqual(kwargs['scale'], scipy.stats.sem(data))
        

class TestTurkeyToTable(unittest.TestCase):

    def test_turkey_result(self):
        df = pd.read_excel('./data/plot.xlsx', sheet_name='ym')
        model = ms.test.multicomp_turkeyHSD({'solution':[], 'time':['after']}, 'whole', df)
        result = ms.test.turkey_to_table(model)
        print(result)

if __name__ == '__main__':
    unittest.main()
