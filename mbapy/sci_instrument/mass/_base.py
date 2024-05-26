'''
Date: 2024-05-20 16:53:21
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2024-05-26 22:44:40
Description: mbapy.sci_instrument.mass._base
'''
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import scipy

# if __name__ == '__main__':
from mbapy.base import check_parameters_path, parameter_checker, put_err
from mbapy.file import (decode_bits_to_str, get_paths_with_extension,
                        get_valid_file_path, opts_file, write_sheets)
from mbapy.sci_instrument._base import SciInstrumentData
from mbapy.web import TaskPool

# else:
#     from ...base import check_parameters_path, parameter_checker, put_err
#     from ...file import (decode_bits_to_str, get_paths_with_extension,
#                          get_valid_file_path, opts_file, write_sheets)
#     from ...web import TaskPool
#     from .._base import SciInstrumentData
    
    
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
    
    def search_peaks(self, xlim: Tuple[float, float] = None, min_width: float = 4,
                     parallel: TaskPool = None, n_parallel: int = 4):
        """
        1. filter data by xlim
        2. find peaks using scipy.signal.find_peaks_cwt with min_width parameter
        
        Parameters:
            - xlim: (float, float), the range of x-axis to search peaks, default is None, which means all data
            - min_width: int, the minimum width of peaks, default is 4
            - parallel: TaskPool, the parallel task pool, default is None, which means not use parallel
            - n_parallel: int, the number of parallel processes, default is 4
        
        Returns:
            - peak_df: pandas.DataFrame, the dataframe of peaks. Has columns: [X_HEADER, Y_HEADER, 'index']
            
        Note:
            - use scipy.signal.find_peaks_cwt to find peaks.
            - the result is different between parallel and non-parallel because of the usage of scipy.signal.find_peaks_cwt
        """
        # 1. filter data by xlim
        if xlim is None:
            xlim = (self.data_df[self.X_HEADER].min(), self.data_df[self.X_HEADER].max())
        y = self.data_df[self.Y_HEADER][(self.data_df[self.X_HEADER] >= xlim[0]) &\
                                          (self.data_df[self.X_HEADER] <= xlim[1])]
        # 2. find peaks using scipy.signal.find_peaks_cwt with min_width parameter
        if parallel is not None:
            y_list = np.array_split(np.array(y), n_parallel)
            y_list = [np.concatenate([y_list[i], y_list[i+1][:min_width]]) for i in range(n_parallel-1)] + [y_list[-1]]
            names = [parallel.add_task(None, scipy.signal.find_peaks_cwt, y_i, min_width) for y_i in y_list]
            y_list_r = list(parallel.wait_till_tasks_done(names).values())
            size = 0
            for i, r_i in enumerate(y_list_r):
                r_i += size
                size += y_list[i].size
            peaks_idx = np.concatenate(y_list_r)
        else:
            peaks_idx = scipy.signal.find_peaks_cwt(y, min_width)
        if peaks_idx.size == 0:
            return put_err('No peaks found, return None')
        self.peak_df = self.data_df.iloc[peaks_idx].copy()
        # return self.peak_df.reset_index(drop=False)
        return self.peak_df.reset_index(drop=False)
    
    def filter_peaks(self, xlim: Tuple[float, float] = None, min_height: float = None,
                     min_height_percent: float = 1):
        # search peaks if peak_df is None
        if self.check_processed_data_empty(self.peak_df):
            self.search_peaks(xlim)
        # filter peaks by xlim and min_height
        if xlim is None:
            xlim = (self.peak_df[self.X_HEADER].min(), self.peak_df[self.X_HEADER].max())
        if min_height is None:
            min_height = 0
        if min_height_percent is not None:
            min_height = max(min_height, min_height_percent / 100 * self.peak_df[self.Y_HEADER].max())
        self.peak_df = self.peak_df[(self.peak_df[self.X_HEADER] >= xlim[0]) &\
                                        (self.peak_df[self.X_HEADER] <= xlim[1]) &\
                                            (self.peak_df[self.Y_HEADER] >= min_height)].copy()
        return self.peak_df.reset_index(drop=True)
    
    
__all__ = [
    'MassData',
]
    
if __name__ == '__main__':
    data = MassData('data_tmp/scripts/mass/d.txt')
    data.X_HEADER, data.Y_HEADER = 'Mass/Charge', 'Intensity'
    data.MULTI_HEADERS = [data.X_HEADER, data.Y_HEADER]
    data.HEADERS_TYPE = {data.X_HEADER:float, data.Y_HEADER:float}
    data.raw_data = data.load_raw_data_file()
    data.processed_data = data.process_raw_data()
    task_pool = TaskPool('process', 4).run()
    peak_df_4 = data.search_peaks(parallel=task_pool).copy()
    peak_df_1 = data.search_peaks().copy()
    print(peak_df_4.equals(peak_df_1))
    data.peak_df = data.filter_peaks(xlim=(100, 200), min_height=1000)
    data.save_processed_data()