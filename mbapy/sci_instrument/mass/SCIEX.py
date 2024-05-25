'''
Date: 2024-05-20 16:52:52
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2024-05-25 08:38:08
Description: mbapy.sci_instrument.mass.SCIEX
'''
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import scipy

if __name__ == '__main__':
    from mbapy.base import check_parameters_path, parameter_checker, put_err
    from mbapy.sci_instrument.mass._base import MassData
else:
    from ...base import check_parameters_path, parameter_checker, put_err
    from ._base import MassData
    
    
class SciexPeakListData(MassData):
    @parameter_checker(data_file_path=lambda path: path is None or (check_parameters_path(path) and path.endswith('.txt')))
    def __init__(self, data_file_path: Optional[str] = None) -> None:
        super().__init__(data_file_path)
        self.X_HEADER = 'Mass/charge (charge)'
        self.Y_HEADER = 'Height'
        # Mass/Charge	Area	Height	Width	Width at 50%	Resolution	Charge	Monoisotopic	Mass (charge)	Mass/charge (charge)
        self.MULTI_HEADERS = ['Mass/Charge', 'Area', 'Height', 'Width',
                              'Width at 50%', 'Resolution', 'Charge',
                              'Monoisotopic', 'Mass (charge)', 'Mass/charge (charge)']
        self.HEADERS_TYPE = {'Mass/Charge':float, 'Height':float, 'Charge':int,
                            'Monoisotopic':str, 'Mass (charge)':str, 'Mass/charge (charge)':str}
        self.raw_data = self.load_raw_data_file()
        self.processed_data = self.process_raw_data() if self.raw_data else None
        self.tag = self.make_tag() if not self.check_processed_data_empty() else None
        
    def process_raw_data(self, *args, **kwargs):
        super().process_raw_data(*args, **kwargs)
        if self.SUCCEED_LOADED:
            self.data_df['Mass (charge)'] = self.data_df['Mass (charge)'].str.extract(r'(\d+\.\d+)', expand=False).astype(float)
            self.data_df['Mass/charge (charge)'] = self.data_df['Mass/charge (charge)'].str.extract(r'(\d+\.\d+)', expand=False).astype(float)
            return self.data_df


class SciexOriData(MassData):
    @parameter_checker(data_file_path=lambda path: path is None or (check_parameters_path(path) and path.endswith('.txt')))
    def __init__(self, data_file_path: Optional[str] = None) -> None:
        super().__init__(data_file_path)
        self.X_HEADER = 'Mass/Charge'
        self.Y_HEADER = 'Intensity'
        self.MULTI_HEADERS = [self.X_HEADER, self.Y_HEADER]
        self.HEADERS_TYPE = {self.X_HEADER:float, self.Y_HEADER:float}
        self.raw_data = self.load_raw_data_file()
        self.processed_data = self.process_raw_data() if self.raw_data else None
        self.tag = self.make_tag() if not self.check_processed_data_empty() else None
    

__all__ = [
    'SciexPeakListData',
    'SciexOriData',
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