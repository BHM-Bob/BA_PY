'''
Date: 2024-05-20 16:53:21
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2024-05-20 21:42:01
Description: 
'''
import os
from pathlib import Path
from typing import Dict, List, Union

import scipy
import numpy as np
import pandas as pd

if __name__ == '__main__':
    from mbapy.base import put_err
    from mbapy.file import decode_bits_to_str, get_paths_with_extension, get_valid_file_path
else:
    from ...base import put_err
    from ...file import decode_bits_to_str, get_paths_with_extension, get_valid_file_path
    
    
class HPLC_Data:
    def __init__(self, data_file_path: Union[str, List[str]]) -> None:
        self.data_file_path = data_file_path
        self.X_HEADER = 'Time'
        self.Y_HEADER = 'Absorbance'
        
    def load_raw_data_file(self, *args, **kwargs):
        raise NotImplementedError()
    
    def load_processed_data_file(self, *args, **kwargs):
        raise NotImplementedError()
    
    def process_raw_data(self, *args, **kwargs):
        raise NotImplementedError()
    
    def save_processed_data(self, *args, **kwargs):
        raise NotImplementedError()
    
    def get_abs_data(self, *args, **kwargs):
        raise NotImplementedError()
    
    def get_processed_data(self, *args, **kwargs):
        raise NotImplementedError()
    
    def get_tag(self, *args, **kwargs):
        raise NotImplementedError()
    
    
__all__ = [
    
]
    
if __name__ == '__main__':
    pass