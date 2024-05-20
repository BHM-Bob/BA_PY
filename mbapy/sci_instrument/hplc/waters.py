import os
from pathlib import Path
from typing import Dict, List, Union

import scipy
import numpy as np
import pandas as pd

if __name__ == '__main__':
    from mbapy.base import put_err, parameter_checker, check_parameters_path
    from mbapy.file import decode_bits_to_str, write_sheets, opts_file
    from mbapy.sci_instrument.hplc._base import HPLC_Data
else:
    from ...base import put_err, parameter_checker, check_parameters_path
    from ...file import decode_bits_to_str, write_sheets, opts_file
    from ._base import HPLC_Data
    

class WatersData(HPLC_Data):
    @parameter_checker(data_file_path=lambda path: path is None or (check_parameters_path(path) and (path.endswith('.arw') or path.endswith('.xlsx'))))
    def __init__(self, data_file_path: str = None) -> None:
        super().__init__(str(Path(data_file_path).resolve()))
        self.raw_data = self.load_raw_data_file()
        self.processed_data = self.process_raw_data() if self.raw_data else None
        self.tag = self.make_tag() if self.processed_data else None
        self.process_raw_data_path: str = None
        self.X_HEADER = 'Time'
        self.Y_HEADER = 'Absorbance'
        
    def load_raw_data_file(self, raw_data_bytes: bytes = None):
        if raw_data_bytes is None and self.data_file_path:
            raw_data_bytes = opts_file(self.data_file_path, 'rb')
        else:
            return None # no waring, return None
        return decode_bits_to_str(raw_data_bytes)
    
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
        lines = self.raw_data.splitlines()
        info_df = pd.DataFrame([lines[1].split('\t')], columns = lines[0].split('\t'))
        data_df = pd.DataFrame([line.split('\t') for line in lines[2:]],
                               columns = ['Time', 'Absorbance']).astype({'Time': float, 'Absorbance': float})
        self.data_df, self.info_df = data_df, info_df
        return info_df, data_df
    
    def get_abs_data(self, *args, **kwargs):
        return self.processed_data[1] if self.processed_data else self.process_raw_data()[1]
    
    def save_processed_data(self, path: str = None, *args, **kwargs):
        if not self.processed_data:
            self.process_raw_data()
        if path is None:
            path = self.process_raw_data_path = Path(self.data_file_path).with_suffix('.xlsx')
        write_sheets(path, {'Info': self.info_df, 'Data': self.data_df})
        return path
    
    def get_processed_data(self, *args, **kwargs):
        return self.processed_data or self.process_raw_data()
    
    def get_tag(self, *args, **kwargs):
        return self.tag or self.make_tag()


__all__ = [
    'WatersData',
]


if __name__ == '__main__':
    data = WatersData('data_tmp/scripts/hplc/ORI_DATA5184.arw')
    data.save_processed_data()
    print(data.get_tag())