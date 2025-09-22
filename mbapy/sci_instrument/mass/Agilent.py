'''
Date: 2025-08-28 20:44:57
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2025-08-28 21:22:08
Description: 
'''
from functools import partial
from typing import Dict, List, Optional, Union

import pandas as pd

if __name__ == '__main__':
    from mbapy.base import check_parameters_path, parameter_checker, put_err
    from mbapy.sci_instrument._base import path_param_checker
    from mbapy.sci_instrument.mass.SCIEX import SciexOriData
else:
    from ...base import check_parameters_path, parameter_checker, put_err
    from .._base import path_param_checker
    from .SCIEX import SciexOriData

class Agilent(SciexOriData):
    DATA_FILE_SUFFIX: List[str] = ['.CSV']
    RECOMENDED_DATA_FILE_SUFFIX: str = '.CSV'
    @parameter_checker(data_file_path=partial(path_param_checker, suffixs=DATA_FILE_SUFFIX))
    def __init__(self, data_file_path: Optional[str] = None) -> None:
        super().__init__(data_file_path)
    
    def load_processed_data_file(self, path: str = None, data_bytes: bytes = None):
        # original data format: head: col1='#Point', col2='X(Thompsons)', col3='Y(Counts)'
        df = pd.read_csv(self.data_file_path or path or data_bytes, skiprows=1)
        if df.columns[1] == 'X(Thompsons)' and df.columns[2] == 'Y(Counts)':
            self.data_df = self.peak_df = self.processed_data = pd.DataFrame(df.iloc[:, 1:].values, columns=self.MULTI_HEADERS)
            self.SUCCEED_LOADED = True
        else:
            self.processed_data = None
            self.SUCCEED_LOADED = False
        return self.processed_data
