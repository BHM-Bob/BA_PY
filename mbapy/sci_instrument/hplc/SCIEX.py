import os
from pathlib import Path
from typing import Dict, List, Union, Optional

import scipy
import numpy as np
import pandas as pd

if __name__ == '__main__':
    from mbapy.base import put_err, parameter_checker, check_parameters_path
    from mbapy.sci_instrument.hplc._base import HplcData
else:
    from ...base import put_err, parameter_checker, check_parameters_path
    from ._base import HplcData
    
    
class SciexData(HplcData):
    @parameter_checker(data_file_path=lambda path: path is None or (check_parameters_path(path) and path.endswith('.txt')))
    def __init__(self, data_file_path: Optional[str] = None) -> None:
        super().__init__(data_file_path)
        self.X_HEADER = 'Time'
        self.Y_HEADER = 'Absorbance'
        self.raw_data = self.load_raw_data_file()
        self.processed_data = self.process_raw_data() if self.raw_data else None
        self.tag = self.make_tag() if not self.check_processed_data_empty() else None
        self.processed_data_path: str = None
        self.TICKS_IN_MINUTE = None # how many ticks in one minute


class SciexTicData(HplcData):
    @parameter_checker(data_file_path=lambda path: path is None or (check_parameters_path(path) and path.endswith('.txt')))
    def __init__(self, data_file_path: Optional[str] = None) -> None:
        super().__init__(data_file_path)
        self.X_HEADER = 'Time'
        self.Y_HEADER = 'Intensity'
        self.raw_data = self.load_raw_data_file()
        self.processed_data = self.process_raw_data() if self.raw_data else None
        self.tag = self.make_tag() if not self.check_processed_data_empty() else None
        self.processed_data_path: str = None
        self.TICKS_IN_MINUTE = None # how many ticks in one minute
    

__all__ = [
    'SciexData',
    'SciexTicData',
]


if __name__ == '__main__':
    SciexTicData(None)
    tic = SciexTicData('data_tmp/scripts/mass/TIC.txt')
    print(tic.raw_data)
    print(tic.processed_data)
    print(tic.tag)
    print(tic.get_tick_by_minute(0.6))
    tic.save_processed_data()
    tic.load_processed_data_file('data_tmp/scripts/mass/TIC.xlsx')