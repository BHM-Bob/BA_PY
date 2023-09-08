'''
Date: 2023-06-30 12:25:23
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-07-12 21:19:31
FilePath: \BA_PY\test\test_base.py
Description: 
'''
import unittest
import inspect
from functools import wraps

from mbapy.base import *

class TestPutErr(unittest.TestCase):
    def test_no_error(self):
        self.assertEqual(put_err("error message"), None)

    def test_with_error(self):
        self.assertEqual(put_err("error message", 123), 123)

    def test_no_error_with_inspect(self):
        self.assertEqual(put_err("error message", ret=456), 456)
        self.assertEqual(put_err("error message", ret="abc"), "abc")

def test_TimeCosts():
    @TimeCosts(2)
    def func(times, *args, **kwargs):
        return times

    result = func(3)
    assert result == [0, 1, 2]

    result = func(s=3)
    assert result == [0, 1, 2]

    result = func(3, 'hello')
    assert result == [0, 1, 2]

    result = func(3, 'hello', 123)
    assert result == [0, 1, 2]

    result = func(3, 'hello', 123, key='value')
    assert result == [0, 1, 2]

    result = func()
    assert result == []

class SplitListTestCase(unittest.TestCase):
    def test_split_list_size_2(self):
        result = split_list([1, 2, 3, 4, 5, 6], n=2)
        self.assertEqual(result, [[1, 2], [3, 4], [5, 6]])

    def test_split_list_size_3(self):
        result = split_list([1, 2, 3, 4, 5, 6], n=3)
        self.assertEqual(result, [[1, 2, 3], [4, 5, 6]])

    def test_split_list_drop_last(self):
        result = split_list([1, 2, 3, 4, 5, 6], n=2, drop_last=True)
        self.assertEqual(result, [[1, 2], [3, 4], [5, 6]])

    def test_split_empty_list(self):
        result = split_list([], n=2)
        self.assertEqual(result, [])

    def test_split_list_size_1(self):
        result = split_list([1, 2, 3, 4, 5, 6], n=1)
        self.assertEqual(result, [[1], [2], [3], [4], [5], [6]])


if __name__ == '__main__':
    unittest.main()