'''
Date: 2023-07-12 21:22:09
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2024-01-09 22:50:04
FilePath: \BA_PY\test\test_web.py
Description: 
'''
import unittest

from mbapy.web import *


class TestGetBetween(unittest.TestCase):
    def test_get_between(self):
        # Testing basic functionality
        self.assertEqual(get_between("abcdefg", "b", "f"), "cde")
        self.assertEqual(get_between("abcdefg", "b", "f", ret_head=True), "bcde")
        self.assertEqual(get_between("abcdefg", "b", "f", ret_tail=True), "cdef")
        self.assertEqual(get_between("abcdefg", "b", "f", ret_head=True, ret_tail=True), "bcdef")
        
        # Testing with last occurrence of head
        self.assertEqual(get_between("abcdefgabc", "b", "f", headRFind=True), "")
        
        # Testing with first occurrence of tail
        self.assertEqual(get_between("abcdefg", "b", "f", tailRFind=False), "cde")
        
        # Testing with ret_head and ret_tail set to False
        self.assertEqual(get_between("abcdefg", "b", "f", ret_head=False, ret_tail=False), "cde")
        
        # Testing with find_tail_from_head set to True
        self.assertEqual(get_between("abxxxcdecfg", "b", "c", find_tail_from_head=True), "xxx")
        
        # Testing error handling when head is not found
        self.assertEqual(get_between("abcdefg", "x", "f"), "abcdefg")
        
        # Testing error handling when tail is not found
        self.assertEqual(get_between("abcdefg", "b", "x"), "abcdefg")
        
        # Testing error handling when head and tail are the same index
        self.assertEqual(get_between("abcdefg", "b", "b"), "")

if __name__ == '__main__':
    unittest.main()