'''
Date: 2024-05-20 16:53:21
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2024-05-24 22:24:03
Description: mbapy.sci_instrument.hplc._base
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
    from mbapy.sci_instrument._base import SciInstrumentData
else:
    from ...base import put_err, parameter_checker, check_parameters_path
    from ...file import decode_bits_to_str, get_paths_with_extension, get_valid_file_path, opts_file, write_sheets
    from .._base import SciInstrumentData
    
    
class HplcData(SciInstrumentData):
    def __init__(self, data_file_path: Union[None, str, List[str]] = None) -> None:
        super().__init__(data_file_path)
        self.X_HEADER = 'Time'
        self.Y_HEADER = 'Absorbance'
        self.TICKS_IN_MINUTE = 60 # how many ticks in one minute
        
    def get_abs_data(self, *args, **kwargs):
        return self.processed_data if not self.check_processed_data_empty() else self.process_raw_data()
    
    
__all__ = [
    
]
    
if __name__ == '__main__':
    pass