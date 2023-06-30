'''
Date: 2023-06-29 22:27:31
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-06-30 12:16:01
FilePath: \BA_PY\test\stats\test_df.py
Description: 
'''
import pandas as pd
import numpy as np
import unittest

from mbapy.stats import df as msd

class TestGetValue(unittest.TestCase):

    def test_single_value(self):
        df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        column = 'A'
        mask = np.array([True, False, False, False, False])
        expected_output = [1]
        self.assertEqual(msd.get_value(df, column, mask), expected_output)

    def test_multiple_values(self):
        df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        column = 'A'
        mask = np.array([False, True, True, False, False])
        expected_output = [2, 3]
        self.assertEqual(msd.get_value(df, column, mask), expected_output)

    def test_empty_mask(self):
        df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        column = 'A'
        mask = np.array([False, False, False, False, False])
        expected_output = []
        self.assertEqual(msd.get_value(df, column, mask), expected_output)

if __name__ == '__main__':
    unittest.main()