'''
Date: 2024-11-10 21:55:48
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2024-11-10 22:02:02
Description: 
'''
import unittest
import os
import json
import tempfile
import pandas as pd

from mbapy.file import opts_file

class TestOptsFile(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.file_path = os.path.join(self.test_dir.name, 'test.txt')
        self.pkl_file_path = os.path.join(self.test_dir.name, 'test.pkl')
        self.json_file_path = os.path.join(self.test_dir.name, 'test.json')
        self.csv_file_path = os.path.join(self.test_dir.name, 'test.csv')
    
    def tearDown(self):
        self.test_dir.cleanup()

    def test_str_write_read(self):
        data = "Hello, World!"
        opts_file(self.file_path, mode='w', way='str', data=data)
        read_data = opts_file(self.file_path, mode='r', way='str')
        self.assertEqual(read_data, data)

    def test_json_read_write(self):
        data = {'key': 'value'}
        opts_file(self.json_file_path, mode='w', way='json', data=data)
        read_data = opts_file(self.json_file_path, mode='r', way='json')
        self.assertEqual(read_data, data)

    def test_auto_file_type_detection(self):
        opts_file(self.json_file_path, mode='w', way='json', data={'key': 'value'})
        self.assertEqual(opts_file(self.json_file_path, mode='r', way='__auto__')['key'], 'value')
        
    def test_None_pkl_read_write(self):
        data = None
        opts_file(self.pkl_file_path, mode='wb', way='pkl', data=data)
        read_data = opts_file(self.pkl_file_path, mode='rb', way='pkl')
        self.assertEqual(read_data, data)

if __name__ == '__main__':
    unittest.main()