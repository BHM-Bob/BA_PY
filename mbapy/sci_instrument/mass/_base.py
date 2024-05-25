'''
Date: 2024-05-20 16:53:21
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2024-05-25 08:37:44
Description: mbapy.sci_instrument.mass._base
'''
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import scipy

if __name__ == '__main__':
    from mbapy.base import check_parameters_path, parameter_checker, put_err
    from mbapy.file import (decode_bits_to_str, get_paths_with_extension,
                            get_valid_file_path, opts_file, write_sheets)
    from mbapy.sci_instrument._base import SciInstrumentData
else:
    from ...base import check_parameters_path, parameter_checker, put_err
    from ...file import (decode_bits_to_str, get_paths_with_extension,
                         get_valid_file_path, opts_file, write_sheets)
    from .._base import SciInstrumentData
    
    
class MassData(SciInstrumentData):
    def __init__(self, data_file_path: Union[None, str, List[str]] = None) -> None:
        super().__init__(data_file_path)
        self.peak_df = None
        self.X_HEADER = 'Mass/charge (charge)'
        self.Y_HEADER = 'Height'
        self.MULTI_HEADERS = [self.X_HEADER, self.Y_HEADER]
        self.HEADERS_TYPE = {self.X_HEADER: float, self.Y_HEADER: float}
        
    
    @parameter_checker(path=lambda path: path is None or check_parameters_path(path))
    def load_processed_data_file(self, path: str = None, data_bytes: bytes = None):
        if path is None and data_bytes is None:
            return put_err('No processed data file specified, return None')
        elif path is not None and data_bytes is not None:
            put_err('Both path and data_bytes are specified, only act with path')
            data_bytes = None
        self.data_df = pd.read_excel(path or data_bytes, sheet_name='Data')
        if 'Peak' in pd.ExcelFile(path or data_bytes).sheet_names:
            self.peak_df = pd.read_excel(path or data_bytes, sheet_name='Peak')
        return self.data_df
    
    def process_raw_data(self, *args, **kwargs):
        try:
            lines = self.raw_data.splitlines()
            headers = lines[0].split('\t')
            if all([headers[i] == self.MULTI_HEADERS[i] for i in range(len(headers))]):
                data_df = pd.DataFrame([line.split('\t') for line in lines[1:]],
                                       columns = self.MULTI_HEADERS).astype(self.HEADERS_TYPE)
                self.SUCCEED_LOADED = True
                self.data_df = data_df
                return data_df
            else:
                return put_err(f'Invalid file header, expected {self.MULTI_HEADERS}, got {headers}, return None')
        except:
            put_err('Failed to process raw data, return None')
            return None
    
    def save_processed_data(self, path: str = None, *args, **kwargs):
        if self.check_processed_data_empty():
            self.process_raw_data()
        if path is None:
            path = self.processed_data_path = Path(self.data_file_path).with_suffix('.xlsx')
        if self.peak_df is not None:
            write_sheets(path, {'Data': self.data_df, 'Peak': self.peak_df})
        else:
            write_sheets(path, {'Data': self.data_df})
        return path
    
    @staticmethod
    def get_tick_by_minute(x):
        return put_err(f'Not supported for {__class__.__name__}, return None')
    
    def search_peaks(self, xlim: Tuple[float, float] = None, min_height: float = None,
                     min_height_percent: float = 1, min_width: float = 4, **kwargs):
        """
        Parameters:
            - xlim: (float, float), the range of x-axis to search peaks, default is None, which means all data
            - min_height: float, the minimum height of peaks, default is None, which means 0
            - min_height_percent: float, [0, 100], the minimum height of peaks as a percentage of the maximum height, default is None
            - min_width: int, the minimum width of peaks, default is 4
        
        Returns:
            - peak_df: pandas.DataFrame, the dataframe of peaks
        """
        tmp_df = self.data_df.copy()
        if xlim is None:
            xlim = (tmp_df[self.X_HEADER].min(), tmp_df[self.X_HEADER].max())
        if min_height is None:
            min_height = 0
        if min_height_percent is not None:
            min_height = max(min_height, min_height_percent / 100 * tmp_df[self.Y_HEADER].max())
        tmp_df = tmp_df[tmp_df[self.Y_HEADER] >= min_height]
        y = self.data_df[self.Y_HEADER][(self.data_df[self.X_HEADER] >= xlim[0]) &\
                                          (self.data_df[self.X_HEADER] <= xlim[1]) &\
                                            (self.data_df[self.Y_HEADER] >= min_height)]
        peaks_idx = scipy.signal.find_peaks_cwt(y, min_width)
        self.peak_df = tmp_df.iloc[peaks_idx].copy()
        return self.peak_df
    
    
__all__ = [
    'MassData',
]
    
if __name__ == '__main__':
    pass