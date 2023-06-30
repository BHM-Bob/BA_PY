'''
Date: 2023-06-30 12:25:23
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-06-30 12:25:44
FilePath: \BA_PY\test\test_base.py
Description: 
'''
import unittest
from functools import wraps

from mbapy.base import autoparse

class TestAutoParse(unittest.TestCase):
    @autoparse
    def __init__(self, x):
        self.x = x

    def test_default_parameter(self):
        obj = self.__class__(x=10)
        self.assertEqual(obj.x, 10)

    def test_default_parameter_with_args(self):
        obj = self.__class__(10)
        self.assertEqual(obj.x, 10)

    def test_multiple_parameters(self):
        obj = self.__class__(x=10, y=20)
        self.assertEqual(obj.x, 10)
        self.assertEqual(obj.y, 20)

    def test_default_parameter_with_kwargs(self):
        obj = self.__class__(y=20)
        self.assertEqual(obj.x, 0)
        self.assertEqual(obj.y, 20)

    def test_default_parameter_with_missing_kwargs(self):
        obj = self.__class__()
        self.assertEqual(obj.x, 0)

if __name__ == '__main__':
    unittest.main()