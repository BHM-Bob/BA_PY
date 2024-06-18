import os
from pathlib import Path
from typing import Dict, List, Union

import scipy
import numpy as np
import pandas as pd

if __name__ == '__main__':
    from mbapy.base import put_err, parameter_checker, check_parameters_path
    from mbapy.file import decode_bits_to_str, write_sheets, opts_file
    from mbapy.sci_instrument.hplc._base import HplcData
else:
    from ...base import put_err, parameter_checker, check_parameters_path
    from ...file import decode_bits_to_str, write_sheets, opts_file
    from ._base import HplcData
    

class WatersData(HplcData):
    @parameter_checker(data_file_path=lambda path: path is None or (check_parameters_path(path) and path.endswith('.arw')))
    def __init__(self, data_file_path: str = None) -> None:
        super().__init__(data_file_path)
        self.raw_data = self.load_raw_data_file()
        self.processed_data = self.process_raw_data() if self.raw_data else None
        self.tag = self.make_tag() if self.processed_data else None
        self.processed_data_path: str = None
        self.X_HEADER = 'Time'
        self.Y_HEADER = 'Absorbance'
        self.TICKS_IN_MINUTE = 60 # how many ticks in one minute
    
    @parameter_checker(path=lambda path: path is None or check_parameters_path(path))
    def load_processed_data_file(self, path: str = None, data_bytes: bytes = None):
        if path is None and data_bytes is None:
            return put_err('No processed data file specified, return None')
        elif path is not None and data_bytes is not None:
            put_err('Both path and data_bytes are specified, only act with path')
            data_bytes = None
        self.info_df = pd.read_excel(path or data_bytes, sheet_name='Info')
        self.data_df = pd.read_excel(path or data_bytes, sheet_name='Data')
        return self.info_df, self.data_df        
    
    def make_tag(self, tag: str = None, tags: List[str] = ['"样品名称"', '"采集日期"', '"通道"'],
                 join_str: str = '_'):
        if tag is None:
            info_df = self.processed_data[0]
            tag = self.tag = join_str.join([info_df.loc[0, t].strip('"') for t in tags])
        return tag
    
    def process_raw_data(self, *args, **kwargs):
        try:
            lines = self.raw_data.splitlines()
            info_df = pd.DataFrame([lines[1].split('\t')], columns = lines[0].split('\t'))
            data_df = pd.DataFrame([line.split('\t') for line in lines[2:]],
                                columns = [self.X_HEADER, self.Y_HEADER]).astype({self.X_HEADER: float, self.Y_HEADER: float})
            self.data_df, self.info_df = data_df, info_df
            self.SUCCEED_LOADED = True
            return info_df, data_df
        except:
            return put_err('Failed to process raw data, return None')
    
    def get_abs_data(self, *args, **kwargs):
        return self.processed_data[1] if self.processed_data else self.process_raw_data()[1]
    
    def save_processed_data(self, path: str = None, *args, **kwargs):
        if not self.processed_data:
            self.process_raw_data()
        if path is None:
            path = self.processed_data_path = Path(self.data_file_path).with_suffix('.xlsx')
        write_sheets(path, {'Info': self.info_df, 'Data': self.data_df}, index = False)
        return path


__all__ = [
    'WatersData',
]


if __name__ == '__main__':
    WatersData()
    data = WatersData('data_tmp/scripts/hplc/ORI_DATA5184.arw')
    data.save_processed_data()
    print(data.get_tag())
    peaks_idx = data.search_peaks(0.1, 0.01)
    print(peaks_idx)
    area = data.calcu_peaks_area(peaks_idx)
    print(area)
    data.get_area()