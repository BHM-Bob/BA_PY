import unittest
import pandas as pd
from mbapy.sci_instrument.mass import SciexOriData, SciexPeakListData

class TestSciexOriData(unittest.TestCase):

    def setUp(self):
        self.data_file_path = 'data_tmp/scripts/mass/d.txt'
        self.processed_data_path = 'data_tmp/scripts/mass/d.xlsx'

    def test_load_raw_data_file(self):
        data = SciexOriData(self.data_file_path)
        raw_data = data.load_raw_data_file()

    def test_load_processed_data_file(self):
        data = SciexOriData(self.data_file_path)
        data.save_processed_data()
        data.load_processed_data_file(self.processed_data_path)
        

class TestSciexPeakListData(unittest.TestCase):

    def setUp(self):
        self.data_file_path = 'data_tmp/scripts/mass/pl.txt'
        self.processed_data_path = 'data_tmp/scripts/mass/pl.xlsx'

    def test_load_raw_data_file(self):
        data = SciexPeakListData(self.data_file_path)
        raw_data = data.load_raw_data_file()

    def test_load_processed_data_file(self):
        data = SciexPeakListData(self.data_file_path)
        data.save_processed_data()
        data.load_processed_data_file(self.processed_data_path)

if __name__ == '__main__':
    unittest.main()