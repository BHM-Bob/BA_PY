'''
Date: 2024-11-10 21:55:48
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2025-02-10 19:44:06
Description: 
'''
import os
import tempfile
import unittest

from mbapy.file import get_paths_with_extension, opts_file


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
        self.assertEqual(opts_file(self.json_file_path, mode='r', way='auto')['key'], 'value')
        
    def test_None_pkl_read_write(self):
        data = None
        opts_file(self.pkl_file_path, mode='wb', way='pkl', data=data)
        read_data = opts_file(self.pkl_file_path, mode='rb', way='pkl')
        self.assertEqual(read_data, data)

class TestGetPaths(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_folder = os.path.join(self.test_dir.name, 'test_folder')
        os.makedirs(self.test_folder)
        
        self.file1 = os.path.join(self.test_folder, 'file1.txt')
        self.file2 = os.path.join(self.test_folder, 'file2.jpg')
        self.subdir = os.path.join(self.test_folder, 'subdir')
        os.makedirs(self.subdir)
        self.file10 = os.path.join(self.subdir, 'file10.txt')
        self.file3 = os.path.join(self.subdir, 'file3.txt')
        
        with open(self.file1, 'w') as f:
            f.write('This is file1.txt')
        with open(self.file2, 'w') as f:
            f.write('This is file2.jpg')
        with open(self.file3, 'w') as f:
            f.write('This is file3.txt')
        with open(self.file10, 'w') as f:
            f.write('This is file10.txt')

    def tearDown(self):
        self.test_dir.cleanup()

    def test_get_paths_with_extension_WithExtension(self):
        result = get_paths_with_extension(self.test_folder, ['.txt'], recursive=False)
        self.assertEqual(result, [self.file1])

    def test_get_paths_with_extension_NoExtension(self):
        result = get_paths_with_extension(self.test_folder, [], recursive=False)
        self.assertEqual(result, [self.file1, self.file2])

    def test_get_paths_with_extension_NameSubstring(self):
        result = get_paths_with_extension(self.test_folder, ['.txt'], name_substr='file1')
        self.assertEqual(result, [self.file1, self.file10])

    def test_get_paths_with_extension_SearchNameInDir(self):
        result = get_paths_with_extension(self.test_folder, ['.txt'], search_name_in_dir=True, name_substr='sub')
        self.assertEqual(result, [self.subdir])

    def test_get_paths_with_extension_Sort(self):
        result = get_paths_with_extension(self.test_folder, ['.txt', '.jpg'], recursive=True, sort=True)
        self.assertEqual(result, [self.file1, self.file2, self.file10, self.file3])

    def test_get_paths_with_extension_NaturalSort(self):
        result = get_paths_with_extension(self.test_folder, ['.txt'], sort='natsort')
        self.assertEqual(result, [self.file1, self.file3, self.file10])

    def test_get_paths_with_extension_UnknownSortOption(self):
        result = get_paths_with_extension(self.test_folder, ['.txt'], sort='unknown')
        self.assertEqual(set(result), set([self.file1, self.file10, self.file3]))

if __name__ == '__main__':
    unittest.main()