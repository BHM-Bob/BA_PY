'''
Date: 2023-10-01 23:02:00
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-10-01 23:12:18
Description: 
'''
import unittest

import numpy as np

from mbapy.stats import max_pool2d


class TestMaxPool2D(unittest.TestCase):
    def test_max_pool2d(self):
        # Test case 1: Pooling window size is (2, 2) and stride is (1, 1)
        x1 = np.array([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12],
                       [13, 14, 15, 16]])
        expected_output1 = np.array([[6, 7, 8],
                                     [10, 11, 12],
                                     [14, 15, 16]]).tolist()
        self.assertEqual(max_pool2d(x1, (2, 2), (1, 1)).tolist(), expected_output1)

        # Test case 2: Pooling window size is (3, 3) and stride is (2, 2)
        x2 = np.array([[1, 2, 3, 4, 5, 6],
                       [7, 8, 9, 10, 11, 12],
                       [13, 14, 15, 16, 17, 18],
                       [19, 20, 21, 22, 23, 24],
                       [25, 26, 27, 28, 29, 30],
                       [31, 32, 33, 34, 35, 36]])
        expected_output2 = np.array([[15, 17],
                                     [27, 29]]).tolist()
        self.assertEqual(max_pool2d(x2, (3, 3), (2, 2)).tolist(), expected_output2)

        # Test case 3: Pooling window size is (2, 2) and stride is None
        x3 = np.array([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12],
                       [13, 14, 15, 16]])
        expected_output3 = np.array([[6, 8],
                                     [14, 16]]).tolist()
        self.assertEqual(max_pool2d(x3, (2, 2)).tolist(), expected_output3)

if __name__ == '__main__':
    unittest.main()