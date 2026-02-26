'''
Date: 2024-05-20 16:52:52
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2025-08-28 21:04:19
Description: mbapy.sci_instrument.mass.SCIEX
'''
import os
from functools import partial
from pathlib import Path
import re
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import scipy

if __name__ == '__main__':
    from mbapy.base import check_parameters_path, parameter_checker, put_err
    from mbapy.sci_instrument._base import path_param_checker
    from mbapy.sci_instrument.mass._base import MassData
else:
    from ...base import check_parameters_path, parameter_checker, put_err
    from .._base import path_param_checker
    from ._base import MassData
    
    
class SciexPeakListData(MassData):
    DATA_FILE_SUFFIX: List[str] = ['.txt', '.xlsx']
    RECOMENDED_DATA_FILE_SUFFIX: str = '.xlsx'
    @parameter_checker(data_file_path=partial(path_param_checker, suffixs=DATA_FILE_SUFFIX))
    def __init__(self, data_file_path: Optional[str] = None) -> None:
        super().__init__(data_file_path)
        self.X_HEADER = 'Mass/charge (charge)'
        self.Y_HEADER = 'Height'
        self.CHARGE_HEADER = 'Charge'
        self.MONOISOTOPIC_HEADER = 'Monoisotopic'
        self.X_MZ_HEADER = 'Mass/charge (charge)'
        self.X_M_HEADER = 'Mass (charge)'
        # Mass/Charge	Area	Height	Width	Width at 50%	Resolution	Charge	Monoisotopic	Mass (charge)	Mass/charge (charge)
        self.MULTI_HEADERS = ['Mass/Charge', 'Area', 'Height', 'Width',
                              'Width at 50%', 'Resolution', 'Charge',
                              'Monoisotopic', 'Mass (charge)', 'Mass/charge (charge)']
        self.HEADERS_TYPE = {'Mass/Charge':float, 'Height':float, 'Charge':int,
                            'Monoisotopic':str, 'Mass (charge)':str, 'Mass/charge (charge)':str}
        self.match_df = pd.DataFrame(columns=['x', 'X_HEADER', 'y', 'Y_HEADER', 'c', 'CHARGE_HEADER',
                                              self.MONOISOTOPIC_HEADER, 'mode', 'substance'])
        if self.data_file_path.endswith('.txt'):
            self.raw_data = self.load_raw_data_file()
            self.processed_data = self.process_raw_data() if self.raw_data else None # return data_df or None
        elif self.data_file_path.endswith('.xlsx'):
            self.processed_data = self.load_processed_data_file(data_file_path) # return data_df
        self.tag = self.make_tag() if not self.check_processed_data_empty() else None
        
    def process_raw_data(self, *args, **kwargs):
        super().process_raw_data(*args, **kwargs)
        if self.SUCCEED_LOADED:
            self.data_df['Mass (charge)'] = self.data_df['Mass (charge)'].str.extract(r'(\d+\.\d+)', expand=False).astype(float)
            self.data_df['Mass/charge (charge)'] = self.data_df['Mass/charge (charge)'].str.extract(r'(\d+\.\d+)', expand=False).astype(float)
            self.data_df['Monoisotopic'] = (self.data_df['Monoisotopic'] == 'Yes').astype(bool)
            self.peak_df = self.data_df.copy()
            return self.data_df


class SciexOriData(MassData):
    DATA_FILE_SUFFIX: List[str] = ['.txt', 'csv', 'CSV', '.xlsx']
    RECOMENDED_DATA_FILE_SUFFIX: str = '.xlsx'
    @parameter_checker(data_file_path=partial(path_param_checker, suffixs=DATA_FILE_SUFFIX))
    def __init__(self, data_file_path: Optional[str] = None) -> None:
        super().__init__(data_file_path)
        self.X_HEADER = 'Mass/Charge'
        self.Y_HEADER = 'Intensity'
        self.X_MZ_HEADER = 'Mass/Charge'
        self.X_M_HEADER = None
        self.MULTI_HEADERS = [self.X_HEADER, self.Y_HEADER]
        self.HEADERS_TYPE = {self.X_HEADER:float, self.Y_HEADER:float}
        if self.data_file_path.endswith('.txt'):
            self.raw_data = self.load_raw_data_file()
            self.processed_data = self.process_raw_data() if self.raw_data else None # return data_df or None
        elif any([self.data_file_path.endswith(suffix) for suffix in self.DATA_FILE_SUFFIX]):
            self.processed_data = self.load_processed_data_file(data_file_path) # return data_df
        self.tag = self.make_tag() if not self.check_processed_data_empty() else None
        

class SciexMZMine(SciexOriData):
    DATA_FILE_SUFFIX: List[str] = ['.xlsx']
    RECOMENDED_DATA_FILE_SUFFIX: str = '.xlsx'
    @parameter_checker(data_file_path=partial(path_param_checker, suffixs=DATA_FILE_SUFFIX))
    def __init__(self, data_file_path: Optional[str] = None) -> None:
        super().__init__(data_file_path)
    
    def load_processed_data_file(self, path: str = None, data_bytes: bytes = None):
        df = pd.read_excel(self.data_file_path or path or data_bytes)
        if re.match(r'Scan #\d+', df.columns[0]) and df.columns[1] == 'Unnamed: 1':
            # original data format: head: col1='Scan #1397', col2='Unnamed: 1'
            self.data_df = self.peak_df = self.processed_data = pd.DataFrame(df.iloc[1:, :].values, columns=self.MULTI_HEADERS)
            self.SUCCEED_LOADED = True
        elif self.MULTI_HEADERS == list(df.columns):
            self.data_df = self.peak_df = self.processed_data = df
            self.SUCCEED_LOADED = True
        else:
            self.processed_data = None
            self.SUCCEED_LOADED = False
        return self.processed_data


__all__ = [
    'SciexPeakListData',
    'SciexOriData',
    'SciexMZMine'
]


if __name__ == '__main__':
    pl = SciexPeakListData('data_tmp/scripts/mass/pl.txt')
    print(pl.processed_data.head)
    
    ori = SciexOriData('data_tmp/scripts/mass/d.txt')
    print(ori.processed_data.head)
    print(ori.tag)
    print(ori.get_tick_by_minute(0.6))
    print(ori.search_peaks())
    ori.save_processed_data()
    ori.load_processed_data_file('data_tmp/scripts/mass/d.xlsx')
    print(ori.processed_data.head)