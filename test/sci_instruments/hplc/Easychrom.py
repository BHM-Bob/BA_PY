
import unittest

import pandas as pd

from mbapy.sci_instrument.hplc import EasychromData


class TestSciexTicData(unittest.TestCase):

    def setUp(self):
        self.data_file_path = 'data_tmp/scripts/hplc/Easychrom.txt'
        self.processed_data_path = 'data_tmp/scripts/hplc/Easychrom.xlsx'

    def test_load_raw_data_file(self):
        data = EasychromData(self.data_file_path)
        data.save_processed_data()
        raw_data = data.load_raw_data_file()

if __name__ == '__main__':
    unittest.main()