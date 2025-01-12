'''
Date: 2024-05-20 16:53:21
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2025-01-12 20:39:32
Description: mbapy.sci_instrument.mass._base
'''
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import scipy

# use multi-process
from mbapy.base import check_parameters_path, parameter_checker, put_err
from mbapy.file import write_sheets
from mbapy.sci_instrument._base import SciInstrumentData
from mbapy.web import TaskPool


class MassData(SciInstrumentData):
    ESI_IRON_MODE = {
        '[M+H]+': dict(m = 1, iron = 'H', im = 1.00782503207, c = 1),
        '[M+2H]2+': dict(m=1, iron='H', im=2.01565006414, c=2),
        '[M+3H]3+': dict(m=1, iron='H', im=3.02347509621, c=3),
        '[M+Na]+': dict(m = 1, iron = 'Na', im = 22.9897692809, c = 1),
        '[M+K]+': dict(m = 1, iron = 'K', im = 38.96370668, c = 1),
        '[M+Li]+': dict(m = 1, iron = 'Li', im = 7.01600455, c = 1),
        '[M+NH4]+': dict(m = 1, iron = 'NH4', im = 18.03437413308, c = 1),
        '[M+ACN+H]+': dict(m = 1, iron = 'ACN', im = 42.03437413308, c = 1),
        '[M+CH3OH+H]+': dict(m = 1, iron = 'CH3OH', im = 33.03403977991, c = 1),
        '[2M+H]+': dict(m = 2, iron = 'H', im = 1.00782503207, c = 1),
        '[2M+Na]+': dict(m = 2, iron = 'Na', im = 22.9897692809, c = 1),
        '[2M+K]+': dict(m = 2, iron = 'K', im = 38.96370668, c = 1),
        '[M]+': dict(m = 1, iron = '', im = 0, c = 1),
        }
    def __init__(self, data_file_path: Union[None, str, List[str]] = None) -> None:
        super().__init__(data_file_path)
        self.peak_df = None
        self.match_df = pd.DataFrame(columns=['x', 'X_HEADER', 'y', 'Y_HEADER', 'c', 'CHARGE_HEADER',
                                              'Monoisotopic', 'mode', 'substance'])
        self.plot_params: Dict[str, Any] = {'min_tag_lim': 0}
        self.X_HEADER = 'Mass/charge (charge)'
        self.Y_HEADER = 'Height'
        self.CHARGE_HEADER = None
        self.MONOISOTOPIC_HEADER = None
        self.X_MZ_HEADER = None
        self.X_M_HEADER = None
        self.MULTI_HEADERS = [self.X_HEADER, self.Y_HEADER]
        self.HEADERS_TYPE = {self.X_HEADER: float, self.Y_HEADER: float}
        
    @parameter_checker(path=lambda path: path is None or check_parameters_path(path))
    def load_processed_data_file(self, path: str = None, data_bytes: bytes = None):
        try:
            if path is None and data_bytes is None:
                path = Path(self.data_file_path).with_suffix('.xlsx')
                print(f'assuming {path} is the processed data file')
            elif path is not None and data_bytes is not None:
                put_err('Both path and data_bytes are specified, only act with path')
                data_bytes = None
            self.data_df = pd.read_excel(path or data_bytes, sheet_name='Data')
            if set(self.data_df.columns) - set(['Unnamed: 0']) == set(self.MULTI_HEADERS):
                self.data_df = self.data_df.astype(self.HEADERS_TYPE)
                if 'Peak' in pd.ExcelFile(path or data_bytes).sheet_names:
                    self.peak_df = pd.read_excel(path or data_bytes, sheet_name='Peak')
                if 'Match' in pd.ExcelFile(path or data_bytes).sheet_names:
                    self.match_df = pd.read_excel(path or data_bytes, sheet_name='Match')
                self.SUCCEED_LOADED = True
            else:
                return put_err(f'Invalid file header, expected {self.MULTI_HEADERS}, got {self.data_df.columns}, return None')
            return self.data_df
        except:
            return put_err(f'Failed to load processed data file {path or data_bytes}, return None')
    
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
        dict4save = {'Data': self.data_df, 'Match': self.match_df}
        if self.peak_df is not None:
            dict4save.update({'Peak': self.peak_df})
        write_sheets(path, dict4save, index = False)
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
            y_list = [np.concatenate([y_list[i], y_list[i+1][:min_width]]) for i in range(n_parallel-1)] + [y_list[-1]] # add min_width at right edge
            names = [parallel.add_task(None, scipy.signal.find_peaks_cwt, y_i, min_width) for y_i in y_list]
            y_list_r = list(parallel.wait_till_tasks_done(names).values())
            size = 0
            for i, r_i in enumerate(y_list_r):
                r_i += size
                size += (y_list[i].size - min_width) # minus back min_width at right edge
            peaks_idx = np.unique(np.concatenate(y_list_r)) # remove duplicates
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
    
    def add_match_record(self, x: float, y: float, c: float, monoisotopic: bool, mode: str, substance: str,
                         x_header: str = None, y_header: str = None, charge_header: str = None):
        x_header, y_header = x_header or self.X_HEADER, y_header or self.Y_HEADER
        charge_header = self.CHARGE_HEADER or ''
        self.match_df.loc[len(self.match_df)+1] = [x, x_header, y, y_header, c, charge_header, monoisotopic, mode, substance]
        return self.match_df
    
    
__all__ = [
    'MassData',
]
    
if __name__ == '__main__':
    data = MassData('data_tmp/scripts/mass/d CK.txt')
    data.X_HEADER, data.Y_HEADER = 'Mass/Charge', 'Intensity'
    data.MULTI_HEADERS = [data.X_HEADER, data.Y_HEADER]
    data.HEADERS_TYPE = {data.X_HEADER:float, data.Y_HEADER:float}
    # data.load_processed_data_file()
    data.raw_data = data.load_raw_data_file()
    data.processed_data = data.process_raw_data()
    task_pool = TaskPool('process', 4).start()
    peak_df_4 = data.search_peaks(parallel=task_pool).copy()
    peak_df_1 = data.search_peaks().copy()
    print(peak_df_4.equals(peak_df_1))
    data.peak_df = data.filter_peaks(xlim=(100, 200), min_height=1000)
    data.save_processed_data()