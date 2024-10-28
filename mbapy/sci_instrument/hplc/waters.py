'''
Date: 2024-06-18 16:25:14
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2024-10-28 16:19:51
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
            tag = self.tag = join_str.join([info_df.loc[0, t].strip('"') for t in tags if t in info_df.columns])
        return tag
    
    def process_raw_data(self):
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
    
    def get_abs_data(self, origin_data: bool = False) -> pd.DataFrame:
        if self.refined_abs_data is not None and not origin_data:
            return self.refined_abs_data
        return self.processed_data[1] if self.processed_data else self.process_raw_data()[1]
    
    def save_processed_data(self, path: str = None):
        if not self.processed_data:
            self.process_raw_data()
        if path is None:
            path = self.processed_data_path = Path(self.data_file_path).with_suffix('.xlsx')
        write_sheets(path, {'Info': self.info_df, 'Data': self.data_df}, index = False)
        return path
    
    
class WatersPdaData(WatersData):
    @parameter_checker(data_file_path=lambda path: path is None or (check_parameters_path(path) and path.endswith('.arw')))
    def __init__(self, data_file_path: str = None) -> None:
        # do not use WatersData.__init__, because process_raw_data will fail
        super(WatersData, self).__init__(data_file_path)
        self.IS_PDA = True
        self.wave_length = None
        self.opt_wave_length = None
        self.raw_data = self.load_raw_data_file()
        self.processed_data = self.process_raw_data() if self.raw_data else None
        self.tag = self.make_tag() if self.processed_data else None
        self.processed_data_path: str = None
    
    def process_raw_data(self):
        try:
            lines = self.raw_data.splitlines()
            info_df = pd.DataFrame([lines[1].split('\t')], columns = lines[0].split('\t'))
            self.wave_length = list(map(lambda x : float(x.strip()), lines[2].split('\t')[1:]))
            data_df = pd.DataFrame([list(map(lambda x : float(x.strip()), line.split('\t'))) for line in lines[4:]],
                                   columns = [self.X_HEADER] + self.wave_length)
            self.data_df, self.info_df = data_df, info_df
            if len(self.data_df.columns) == 2:
                put_err(f'only one wave length found in {self.data_file_path}, {self.wave_length}, assume not PDA data')
                self.SUCCEED_LOADED = False
            else:
                self.SUCCEED_LOADED = True
            return info_df, data_df
        except:
            return put_err('Failed to process raw data, return None')
    
    def get_abs_data(self, wave_length: float = None, origin_data: bool = False) -> pd.DataFrame:
        # get wave_length
        if wave_length is None:
            if self.opt_wave_length is None:
                raise ValueError('No wave_length specified, please specify one')
            wave_length = self.opt_wave_length
        # get full data
        data = super().get_abs_data(origin_data=origin_data)
        if not origin_data:
            return data
        # get specific wave_length data
        if wave_length not in self.wave_length:
            diff = np.abs(np.array(self.wave_length)-wave_length)
            nearst_idx = diff.argmin(axis=0)
            if nearst_idx == 0 or nearst_idx == len(self.wave_length)-1:
                abs_data = data[self.wave_length[nearst_idx]]
            else:
                # linear interpolation
                delta_wave = self.wave_length[nearst_idx+1] - self.wave_length[nearst_idx-1]
                abs1, abs2 = data[self.wave_length[nearst_idx-1]], data[self.wave_length[nearst_idx+1]]
                abs_data = (wave_length - self.wave_length[nearst_idx-1]) * (abs2 - abs1) / delta_wave + abs1
        else:
            abs_data = data[wave_length]
        return pd.DataFrame({self.X_HEADER: data[self.X_HEADER], self.Y_HEADER: abs_data})
    
    def set_opt_wave_length(self, wave_length: float):
        if wave_length < min(self.wave_length) or wave_length > max(self.wave_length):
            raise ValueError(f'wave_length out of range {min(self.wave_length)}-{max(self.wave_length)}, got {wave_length}')
        self.opt_wave_length = wave_length


__all__ = [
    'WatersData',
    'WatersPdaData'
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
    
    # WatersPdaData
    data = WatersPdaData('data_tmp/scripts/hplc/WatersPDA.arw')
    data.set_opt_wave_length(228)
    data.refined_abs_data = data.get_abs_data(origin_data=True).copy()
    data.get_abs_data(origin_data=False)
    peaks_idx = data.search_peaks(0.1, 0.01)
    print(peaks_idx)
    area = data.calcu_peaks_area(peaks_idx)
    print(area)
    data.get_area()