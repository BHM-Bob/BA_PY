'''
Date: 2024-07-03 15:33:36
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2024-07-03 15:53:31
Description: 
'''

from typing import Dict, List, Optional, Union

import pandas as pd

if __name__ == '__main__':
    from mbapy.base import check_parameters_path, parameter_checker, put_err
    from mbapy.sci_instrument.hplc.SCIEX import SciexData
else:
    from ...base import check_parameters_path, parameter_checker, put_err
    from .SCIEX import SciexData
    
    
class EasychromData(SciexData):
    @parameter_checker(data_file_path=lambda path: path is None or (check_parameters_path(path) and path.endswith('.txt')))
    def __init__(self, data_file_path: Optional[str] = None) -> None:
        super().__init__(data_file_path)
    
    def process_raw_data(self, y_scale: float = 0.001, **kwargs):
        try:
            lines = [line.split() for line in self.raw_data.splitlines()]
            data_df = pd.DataFrame([[float(t), float(a)] for t,a in lines],
                                   columns = [self.X_HEADER, self.Y_HEADER]).astype({self.X_HEADER: float, self.Y_HEADER: float})
            data_df[self.Y_HEADER] *= y_scale
            self.SUCCEED_LOADED = True
            self.data_df = data_df
            return data_df
        except:
            put_err('Failed to process raw data, return None')
            return None
        

__all__ = [
    'EasychromData'
]


if __name__ == '__main__':
    # dev code
    EasychromData('data_tmp/scripts/hplc/Easychrom.txt')