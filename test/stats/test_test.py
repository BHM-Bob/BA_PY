'''
Date: 2023-05-22 15:01:36
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2025-04-30 16:21:22
FilePath: \BA_PY\test\stats\test_test.py
Description: 
'''
import unittest

import numpy as np
import pandas as pd
import scipy

from mbapy.stats.test import get_interval, multicomp_turkeyHSD, turkey_to_table


class TestGetInterval(unittest.TestCase):

    def test_mean_se(self):
        # Testing when mean and se are provided
        mean = 10
        se = 2
        confidence = 0.95
        interval, kwargs = get_interval(mean=mean, se=se, confidence=confidence)
        self.assertEqual(len(interval), 2)
        self.assertLess(interval[0], interval[1])
        self.assertEqual(kwargs['loc'], mean)
        self.assertEqual(kwargs['scale'], se)

    def test_data_confidence(self):
        # Testing when data and confidence are provided
        data = [1, 2, 3, 4, 5]
        confidence = 0.99
        interval, kwargs = get_interval(data=data, confidence=confidence)
        self.assertEqual(len(interval), 2)
        self.assertLess(interval[0], interval[1])
        self.assertEqual(kwargs['loc'], np.mean(data).item())
        self.assertEqual(kwargs['scale'], scipy.stats.sem(data))

    def test_se_none_data_none(self):
        # Testing when se is not provided and data is None
        mean = 5
        confidence = 0.9
        with self.assertRaises(AssertionError) as context:
            get_interval(mean=mean, confidence=confidence)
        self.assertEqual(str(context.exception), 'se is None and data is None')

    def test_se_none_data(self):
        # Testing when se is None and data is provided
        se = None
        data = [1, 2, 3, 4, 5]
        confidence = 0.85
        interval, kwargs = get_interval(data=data, confidence=confidence)
        self.assertEqual(len(interval), 2)
        self.assertLess(interval[0], interval[1])
        self.assertEqual(kwargs['loc'], np.mean(data).item())
        self.assertEqual(kwargs['scale'], scipy.stats.sem(data))
        
    def test_get_interval_with_data(self):
        """测试使用数据样本的情况（应使用t分布）"""
        data = np.array([4.5, 5.0, 5.5, 4.8, 5.2])
        (low, high), params = get_interval(data=data, confidence=0.95)
        
        # 验证参数计算
        self.assertAlmostEqual(params['loc'], np.mean(data), delta=1e-5)
        self.assertAlmostEqual(params['scale'], scipy.stats.sem(data), delta=1e-5)
        self.assertEqual(params['df'], len(data)-1)
        
        # 验证置信区间范围
        t_crit = scipy.stats.t.ppf(0.975, df=4)
        expected = np.mean(data) + np.array([-1, 1]) * t_crit * scipy.stats.sem(data)
        self.assertAlmostEqual(low, expected[0], delta=1e-5)
        self.assertAlmostEqual(high, expected[1], delta=1e-5)

    def test_get_interval_with_se(self):
        """测试直接提供均值和标准误（应使用正态分布）"""
        (low, high), params = get_interval(mean=5.0, se=1.0, confidence=0.95)
        
        # 验证参数
        self.assertEqual(params['loc'], 5.0)
        self.assertEqual(params['scale'], 1.0)
        
        # 验证置信区间
        z_crit = scipy.stats.norm.ppf(0.975)
        expected = 5.0 + np.array([-1, 1]) * z_crit * 1.0
        self.assertAlmostEqual(low, expected[0], delta=1e-5)
        self.assertAlmostEqual(high, expected[1], delta=1e-5)

    def test_invalid_input(self):
        """测试无效输入"""
        with self.assertRaises(AssertionError):
            get_interval()  # 不提供任何参数


class TestTurkeyToTable(unittest.TestCase):

    def test_turkey_result(self):
        df = pd.read_excel('./data/plot.xlsx', sheet_name='ym')
        model = multicomp_turkeyHSD({'solution':[], 'time':['after']}, 'whole', df)
        result = turkey_to_table(model)
        print(result)
        

if __name__ == '__main__':
    unittest.main()
