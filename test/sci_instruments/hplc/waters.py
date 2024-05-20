'''
Date: 2024-05-20 21:20:21
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2024-05-20 21:26:45
Description: 
'''
import unittest
import pandas as pd
from mbapy.sci_instrument.hplc.waters import WatersData

class TestWatersData(unittest.TestCase):

    def setUp(self):
        self.data_file_path = 'data_tmp/scripts/hplc/ORI_DATA5184.arw'
        self.processed_data_path = 'data_tmp/scripts/hplc/ORI_DATA5184.xlsx'

    def test_load_raw_data_file(self):
        waters_data = WatersData(self.data_file_path)
        raw_data = waters_data.load_raw_data_file()

    def test_load_processed_data_file(self):
        waters_data = WatersData(self.data_file_path)
        waters_data.save_processed_data()
        waters_data.load_processed_data_file(self.processed_data_path)

if __name__ == '__main__':
    unittest.main()