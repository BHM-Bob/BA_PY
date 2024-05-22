'''
Date: 2024-05-20 16:53:21
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2024-05-22 09:40:49
Description: 
'''
import os
from pathlib import Path
from typing import Dict, List, Union

import scipy
import numpy as np
import pandas as pd

if __name__ == '__main__':
    from mbapy.base import put_err, parameter_checker, check_parameters_path
    from mbapy.file import decode_bits_to_str, get_paths_with_extension, get_valid_file_path, opts_file, write_sheets
else:
    from ...base import put_err, parameter_checker, check_parameters_path
    from ...file import decode_bits_to_str, get_paths_with_extension, get_valid_file_path, opts_file, write_sheets
    
    
class HplcData:
    def __init__(self, data_file_path: Union[None, str, List[str]] = None) -> None:
        self.data_file_path = data_file_path if not data_file_path else str(Path(data_file_path).resolve())
        self.processed_data = None
        self.tag = None
        self.processed_data_path = None
        self.raw_data = None
        self.data_df = None
        self.X_HEADER = 'Time'
        self.Y_HEADER = 'Absorbance'
        self.TICKS_IN_MINUTE = 60 # how many ticks in one minute
        self.SUCCEED_LOADED = False
        
    def check_processed_data_empty(self):
        if isinstance(self.processed_data, pd.DataFrame):
            return self.processed_data.empty
        else:
            return not bool(self.processed_data)
        
    def load_raw_data_file(self, raw_data_bytes: bytes = None):
        if raw_data_bytes is None and self.data_file_path:
            raw_data_bytes = opts_file(self.data_file_path, 'rb')
        elif raw_data_bytes is None and not self.data_file_path:
            return put_err('No raw data file specified, return None')
        return decode_bits_to_str(raw_data_bytes)
    
    def load_raw_data_from_bytes(self, raw_data_bytes: bytes):
        return self.load_raw_data_file(raw_data_bytes)
    
    @parameter_checker(path=lambda path: path is None or check_parameters_path(path))
    def load_processed_data_file(self, path: str = None, data_bytes: bytes = None):
        if path is None and data_bytes is None:
            return put_err('No processed data file specified, return None')
        elif path is not None and data_bytes is not None:
            put_err('Both path and data_bytes are specified, only act with path')
            data_bytes = None
        self.data_df = pd.read_excel(path or data_bytes, sheet_name='Data')
        return self.data_df
    
    def make_tag(self, tag: str = None, **kwargs):
        if tag is None and self.data_file_path is not None:
            tag = self.tag = Path(self.data_file_path).stem
        return tag
    
    def process_raw_data(self, *args, **kwargs):
        try:
            lines = self.raw_data.splitlines()
            header = lines[0].split('\t')
            if header[0] == self.X_HEADER and header[1] == self.Y_HEADER:
                data_df = pd.DataFrame([line.split('\t') for line in lines[1:]],
                                       columns = [self.X_HEADER, self.Y_HEADER]).astype({self.X_HEADER: float, self.Y_HEADER: float})
                self.SUCCEED_LOADED = True
                self.data_df = data_df
                return data_df
            else:
                return put_err(f'Invalid file header, expected "{self.X_HEADER}" and "{self.Y_HEADER}", but got "{header[0]}" and "{header[1]}", return None')
        except:
            put_err('Failed to process raw data, return None')
            return None
    
    def save_processed_data(self, path: str = None, *args, **kwargs):
        if self.check_processed_data_empty():
            self.process_raw_data()
        if path is None:
            path = self.processed_data_path = Path(self.data_file_path).with_suffix('.xlsx')
        write_sheets(path, {'Data': self.data_df})
        return path
    
    def get_abs_data(self, *args, **kwargs):
        return self.processed_data if not self.check_processed_data_empty() else self.process_raw_data()
    
    def get_tick_by_minute(self, minute: float):
        if self.TICKS_IN_MINUTE is not None:
            return minute * self.TICKS_IN_MINUTE
        else:
            data_df = self.get_processed_data()
            return np.argmin(np.abs(data_df[self.X_HEADER] - minute))
    
    def get_processed_data(self, *args, **kwargs):
        if not self.check_processed_data_empty():
            return self.processed_data
        else:
            self.process_raw_data()
    
    def get_tag(self, *args, **kwargs):
        return self.tag or self.make_tag()
    
    
__all__ = [
    
]
    
if __name__ == '__main__':
    pass