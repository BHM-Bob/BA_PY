'''
Date: 2024-05-21 22:16:15
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2024-07-03 21:46:07
Description:
'''
import unittest

import pandas as pd

from mbapy.sci_instrument.hplc import SciexData, SciexTicData


class TestSciexData(unittest.TestCase):

    def setUp(self):
        self.data_file_path = 'data_tmp/scripts/hplc/SCIEX.txt'
        self.processed_data_path = 'data_tmp/scripts/hplc/SCIEX.xlsx'

    def test_load_raw_data_file(self):
        data = SciexTicData(self.data_file_path)
        raw_data = data.load_raw_data_file()

    def test_load_processed_data_file(self):
        data = SciexData(self.data_file_path)
        data.save_processed_data()
        data.load_processed_data_file(self.processed_data_path)

class TestSciexTicData(unittest.TestCase):

    def setUp(self):
        self.data_file_path = 'data_tmp/scripts/mass/TIC.txt'
        self.processed_data_path = 'data_tmp/scripts/mass/TIC.xlsx'

    def test_load_raw_data_file(self):
        data = SciexTicData(self.data_file_path)
        raw_data = data.load_raw_data_file()

    def test_load_processed_data_file(self):
        data = SciexTicData(self.data_file_path)
        data.save_processed_data()
        data.load_processed_data_file(self.processed_data_path)

if __name__ == '__main__':
    unittest.main()