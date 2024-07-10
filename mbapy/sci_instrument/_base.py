
import os
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import scipy

if __name__ == '__main__':
    from mbapy.base import (check_parameters_path, get_default_for_None,
                            parameter_checker, put_err)
    from mbapy.file import (decode_bits_to_str, get_paths_with_extension,
                            get_valid_file_path, opts_file, write_sheets)
else:
    from ..base import (check_parameters_path, get_default_for_None,
                        parameter_checker, put_err)
    from ..file import (decode_bits_to_str, get_paths_with_extension,
                        get_valid_file_path, opts_file, write_sheets)
    
    
def path_param_checker(path: str, suffixs: List[str] = None):
    """return True means path is valid (None, or a valid path with valid suffixs)"""
    if path is None:
        return True
    else:
        suffixs = get_default_for_None(suffixs, [])
        return check_parameters_path(path) and any(path.endswith(suffix) for suffix in suffixs)
    
class SciInstrumentData:
    DATA_FILE_SUFFIX: List[str] = []
    RECOMENDED_DATA_FILE_SUFFIX: str = ''
    def __init__(self, data_file_path: Union[None, str, List[str]] = None) -> None:
        self.data_file_path = data_file_path if not data_file_path else str(Path(data_file_path).resolve()) # None or a absolute path
        self.processed_data = None
        self.tag = None
        self.processed_data_path = None
        self.raw_data = None
        self.data_df = None
        self.X_HEADER = None
        self.Y_HEADER = None
        self.TICKS_IN_MINUTE = None # how many ticks in one minute
        self.SUCCEED_LOADED = False
        
    def check_processed_data_empty(self, processed_data = None):
        """check if processed_data or self.processed_data is empty"""
        processed_data = get_default_for_None(processed_data, self.processed_data)
        if isinstance(processed_data, pd.DataFrame):
            return processed_data.empty
        else:
            return not bool(processed_data)
        
    def load_raw_data_file(self, raw_data_bytes: bytes = None):
        """
        Load the raw data file and return the decoded string
        
        Parameters:
            - raw_data_bytes (bytes): The raw data bytes. Defaults to None.

        Returns:
            - The decoded string from the raw data bytes.
        """
        if raw_data_bytes is None and self.data_file_path:
            raw_data_bytes = opts_file(self.data_file_path, 'rb')
        elif raw_data_bytes is None and not self.data_file_path:
            return put_err('No raw data file specified, return None')
        return decode_bits_to_str(raw_data_bytes)
    
    def load_raw_data_from_bytes(self, raw_data_bytes: bytes):
        return self.load_raw_data_file(raw_data_bytes)
    
    @parameter_checker(path=lambda path: path is None or check_parameters_path(path))
    def load_processed_data_file(self, path: str = None, data_bytes: bytes = None):
        """
        Load processed data from a file or a byte string

        Parameters:
            - path (str): The path to the file. Defaults to None.
            - data_bytes (bytes): The byte string containing the data. Defaults to None.

        Return:
            - The processed data in a DataFrame object.
        """
        if path is None and data_bytes is None:
            return put_err('No processed data file specified, return None')
        elif path is not None and data_bytes is not None:
            put_err('Both path and data_bytes are specified, only act with path')
            data_bytes = None
        self.data_df = pd.read_excel(path or data_bytes, sheet_name='Data')
        return self.data_df
    
    def make_tag(self, tag: str = None, **kwargs):
        if tag is None and self.data_file_path is not None:
            tag = self.tag = Path(self.data_file_path).stem
        return tag
    
    def process_raw_data(self, y_scale: float = 1, **kwargs):
        """
        Process raw data, split the data into a list, and create a DataFrame from it, sorted in ascending order of the x-axis, multiplied by the y-axis scaling factor

        Parameters:
            - y_scale (float): The scaling factor for the y-axis. Default is 1.
            - **kwargs: Additional keyword arguments.

        Return:
            - The processed data in a DataFrame object.

        Raises:
            - Invalid file header: If the file header is invalid, an error message will be returned
            - Exception: If there is an error in the data processing process, an error message will be returned
        """
        try:
            lines = self.raw_data.splitlines()
            header = lines[0].split('\t')
            if header[0] == self.X_HEADER and header[1] == self.Y_HEADER:
                data_df = pd.DataFrame([line.split('\t') for line in lines[1:]],
                                       columns = [self.X_HEADER, self.Y_HEADER]).astype({self.X_HEADER: float, self.Y_HEADER: float})
                data_df[self.Y_HEADER] *= y_scale
                self.SUCCEED_LOADED = True
                self.data_df = data_df
                return data_df
            else:
                return put_err(f'Invalid file header, expected "{self.X_HEADER}" and "{self.Y_HEADER}", but got "{header[0]}" and "{header[1]}", return None')
        except:
            put_err('Failed to process raw data, return None')
            return None
    
    def save_processed_data(self, path: str = None):
        """
        Save the processed data to a file.

        Parameters:
            - path (str): The path of the file to be saved. If not provided, the function will use the original data file path with a different suffix.

        Return:
            - str: The path of the saved file.

        Side effects:
            - This function will save the processed data to the specified file path. If the file exists, it will be overwritten.
        """
        if self.check_processed_data_empty():
            self.process_raw_data()
        if path is None:
            path = self.processed_data_path = Path(self.data_file_path).with_suffix('.xlsx')
        write_sheets(path, {'Data': self.data_df}, index = False)
        return path
    
    def get_tick_by_minute(self, minute: float):
        """return the nearest tick to the given minute"""
        if self.TICKS_IN_MINUTE is not None:
            return minute * self.TICKS_IN_MINUTE
        else:
            data_df = self.get_processed_data()
            return np.argmin(np.abs(data_df[self.X_HEADER] - minute))
    
    def get_processed_data(self, *args, **kwargs):
        if not self.check_processed_data_empty():
            return self.processed_data
        else:
            self.process_raw_data()
    
    def get_tag(self, *args, **kwargs):
        return self.tag or self.make_tag()
    
    
__all__ = [
    'path_param_checker',
    'SciInstrumentData',
]
    
if __name__ == '__main__':
    pass