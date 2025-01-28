'''
Date: 2023-06-30 12:25:23
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2025-01-27 19:44:45
FilePath: \BA_PY\test\test_base.py
Description: 
'''
import inspect
import os
import tempfile
import unittest
from functools import wraps

from mbapy.base import *


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

class ConfigTestCase(unittest.TestCase):
    def test_err_level(self):
        Configs.err_warning_level = 999
        sum_log = len(Configs.logs)
        put_err('test no err log')
        self.assertEqual(sum_log, len(Configs.logs))
        
        Configs.err_warning_level = 0
        sum_log = len(Configs.logs)
        put_err('test full err log')
        self.assertEqual(sum_log + 1, len(Configs.logs))


class ImportFileAsPackageTestCase(unittest.TestCase):
    def test_import_file_as_package_valid_module_name(self):
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmpfile:
            tmpfile.write(b"print('Hello, World!')")
            file_path = tmpfile.name

        # Import the file as a package with a valid module name
        module = import_file_as_package(file_path, module_name="my_module")

        # Assert that the module was imported successfully
        assert module is not None

        # Clean up
        os.remove(file_path)

    def test_import_file_as_package_invalid_module_name(self):
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmpfile:
            tmpfile.write(b"print('Hello, World!')")
            file_path = tmpfile.name

        # Import the file as a package with an invalid module name
        module = import_file_as_package(file_path, module_name="invalid!module@name")

        # Assert that the module was imported successfully with a generated name
        assert module is not None

        # Clean up
        os.remove(file_path)

    def test_import_file_as_package_no_force_reload(self):
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmpfile:
            tmpfile.write(b"print('Hello, World!')")
            file_path = tmpfile.name

        # Import the file as a package
        module1 = import_file_as_package(file_path)

        # Import the file again with force_reload=True
        module2 = import_file_as_package(file_path, force_reload=False)

        # Assert that the modules are different
        assert module1 is module2

        # Clean up
        os.remove(file_path)

    def test_import_file_as_package_force_reload(self):
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmpfile:
            tmpfile.write(b"print('Hello, World!')")
            file_path = tmpfile.name

        # Import the file as a package
        module1 = import_file_as_package(file_path)

        # Import the file again with force_reload=True
        module2 = import_file_as_package(file_path, force_reload=True)

        # Assert that the modules are different
        assert module1 is not module2

        # Clean up
        os.remove(file_path)

    def test_import_file_as_package_file_not_found(self):
        # Try to import a non-existent file
        module = import_file_as_package("non_existent_file.py")

        # Assert that the module is None
        assert module is None
        
    def test_import_dir_from_init_file(self):
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a temporary file in the directory
            with open(os.path.join(tmpdir, "__init__.py"), "w") as f:
                f.write("print('Hello, World!')")

            # Import the directory as a package
            module = import_file_as_package(tmpdir)

            # Assert that the module was imported successfully
            assert module is not None


if __name__ == '__main__':
    unittest.main()