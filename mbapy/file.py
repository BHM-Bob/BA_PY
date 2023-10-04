'''
Author: BHM-Bob 2262029386@qq.com
Date: 2022-11-01 19:09:54
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-10-02 22:43:37
Description: 
'''
import collections
import os
import shutil
from glob import glob
from typing import Dict, List

import chardet

try:
    import ujson as json
except:
    import json

import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    # dev mode
    from mbapy.base import (check_parameters_path, format_secs,
                            get_default_for_bool, parameter_checker, put_err)

    # video and image functions assembly
    try:
        import cv2
        import torch

        from mbapy.file_utils.image import *
        from mbapy.file_utils.video import *
    except:
        pass
else:
    # release mode
    from .base import (check_parameters_path, format_secs,
                       get_default_for_bool, parameter_checker, put_err)

    # video and image functions assembly
    try:# mbapy package now does not require cv2 and torch installed forcibly
        import cv2
        import torch

        from .file_utils.image import *
        from .file_utils.video import *
    except:# if cv2 or torch is not installed, skip
        pass
    
def get_paths_with_extension(folder_path: str, file_extensions: List[str]) -> List[str]:
    """
    Returns a list of file paths within a given folder that have a specified extension.

    Args:
        folder_path (str): The path of the folder to search for files.
        file_extensions (List[str]): A list of file extensions to filter the search by.

    Returns:
        List[str]: A list of file paths that match the specified file extensions.
    """
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.endswith(extension) for extension in file_extensions):
                file_paths.append(os.path.join(root, file))
    return file_paths

def extract_files_from_dir(root: str, file_extensions: List[str] = None,
                           extract_sub_dir: bool = True, join_str:str = ' '):
    """
    Move all files in subdirectories to the root directory and add the subdirectory name as a prefix to the file name.

    Args:
        root (str): The root directory path.
        file_extensions (list[str]): specific file types string (without '.'), if None, means all types.
        extract_sub_dir (bool, optional): Whether to recursively extract files from subdirectories.
            If set to False, only files in the immediate subdirectories will be extracted. Defaults to True.
        join_str (str): string for link prefix and the file name.

    Returns:
        None
    """
    for dirpath, dirnames, filenames in os.walk(root):
        if not extract_sub_dir or dirpath == root:
            continue
        for filename in filenames:
            if file_extensions is None or any(filename.endswith(extension) for extension in file_extensions):
                if extract_sub_dir:
                    new_filename = dirpath.split(root)[1][1:] + join_str + filename
                else:
                    new_filename = filename
                src_path = os.path.join(dirpath, filename)
                dest_path = os.path.join(root, new_filename)
                shutil.move(src_path, dest_path)

def replace_invalid_path_chr(path:str, valid_chrs:str = '_'):
    """
    Replaces any invalid characters in a given path with a specified valid character.

    Args:
        path (str): The path string to be checked for invalid characters.
        valid_chrs (str, optional): The valid characters that will replace any invalid characters in the path. Defaults to '_'.

    Returns:
        str: The path string with all invalid characters replaced by the valid character.
    """
    invalid_chrs = ':*?"<>|\n'
    for invalid_chr in invalid_chrs:
        path = path.replace(invalid_chr, valid_chrs)
    return path

def get_valid_file_path(path:str, valid_chrs:str = '_', valid_len = 250):
    return replace_invalid_path_chr(path, valid_chrs)[:valid_len]

def opts_file(path:str, mode:str = 'r', encoding:str = 'utf-8', way:str = 'str', data = None, **kwargs):
    """
    A function that reads or writes data to a file based on the provided options.

    Parameters:
        path (str): The path to the file.
        mode (str, optional): The mode in which the file should be opened. Defaults to 'r'.
        encoding (str, optional): The encoding of the file. Defaults to 'utf-8'.
        way (str, optional): The way in which the data should be read or written. Defaults to 'lines'.
        data (Any, optional): The data to be written to the file. Only applicable in write mode. Defaults to None.

    Returns:
        list or str or dict or None: The data read from the file, or None if the file was opened in write mode and no data was provided.
    """
    if 'b' not in mode:
        kwargs.update(encoding=encoding)
    with open(path, mode, **kwargs) as f:
        if 'r' in mode:
            if way == 'lines':
                return f.readlines()
            elif way == 'str':
                return f.read()
            elif way == 'json':
                return json.loads(f.read())
        elif 'w' in mode and data is not None:
            if way == 'lines':
                return f.writelines(data)
            elif way == 'str':
                return f.write(data)
            elif way == 'json':
                return f.write(json.dumps(data))

def read_bits(path:str):
    return opts_file(path, 'rb')

def read_text(path:str, decode:str = 'utf-8', way:str = 'lines'):
    return opts_file(path, 'r', decode, way)

def detect_byte_coding(bits:bytes):
    """
    This function takes a byte array as input and detects the encoding of the first 1000 bytes (or less if the input is shorter).
    It then decodes the byte array using the detected encoding and returns the resulting text.

    Parameters:
        bits (bytes): The byte array to be decoded.

    Returns:
        str: The decoded text.
    """
    adchar = chardet.detect(bits[:(1000 if len(bits) > 1000 else len(bits))])['encoding']
    if adchar == 'gbk' or adchar == 'GBK' or adchar == 'GB2312':
        true_text = bits.decode('GB2312', "ignore")
    else:
        true_text = bits.decode('utf-8', "ignore")
    return true_text

def get_byte_coding(bits:bytes, max_detect_len = 1000):
    """
    Calculate the byte coding of a given sequence of bits.

    Args:
        bits (bytes): The sequence of bits to be analyzed.
        max_detect_len (int, optional): The maximum number of bits to consider when determining the byte coding. Defaults to 1000.

    Returns:
        str: The detected byte coding of the input sequence.
    """
    return chardet.detect(bits[ : min(max_detect_len, len(bits))])['encoding']

def decode_bits_to_str(bits:bytes):
    """
    Decodes a bytes object to a string using either GB2312 or utf-8 encoding.
    
    Args:
        bits (bytes): The bytes object to decode.
    
    Returns:
        str: The decoded string.
    """
    adchar = get_byte_coding(bits, max_detect_len = 1000)
    if adchar == 'gbk' or adchar == 'GBK' or adchar == 'GB2312':
        true_text = bits.decode('GB2312', "ignore")
    else:
        true_text = bits.decode('utf-8', "ignore")
    return true_text

def is_jsonable(data):
    if isinstance(data, str) or isinstance(data, int) or isinstance(data, float) or isinstance(data, bool) or data is None:
        return True
    elif isinstance(data, collections.abc.Mapping):
        return all(is_jsonable(value) for value in data.values())
    elif isinstance(data, collections.abc.Sequence):
        return all(is_jsonable(item) for item in data)
    else:
        return False

def save_json(path:str, obj, encoding:str = 'utf-8', forceUpdate = True):
    """
    Saves an object as a JSON file at the specified path.

    Parameters:
        - path (str): The path where the JSON file will be saved.
        - obj: The object to be saved as JSON.
        - encoding (str): The encoding of the JSON file. Default is 'utf-8'.
        - forceUpdate (bool): Determines whether to overwrite an existing file at the specified path. Default is True.

    Returns:
        None
    """
    if forceUpdate or not os.path.isfile(path):
        json_str = json.dumps(obj, indent=1)
        with open(path, 'w' ,encoding=encoding, errors='ignore') as json_file:
            json_file.write(json_str)
            
def read_json(path:str, encoding:str = 'utf-8', invalidPathReturn = None):
    """
    Read a JSON file from the given path and return the parsed JSON data.

    Parameters:
        path (str): The path to the JSON file.
        encoding (str, optional): The encoding of the file. Defaults to 'utf-8'.
        invalidPathReturn (any, optional): The value to return if the path is invalid. Defaults to None.

    Returns:
        dict: The parsed JSON data.
        invalidPathReturn (any): The value passed as `invalidPathReturn` if the path is invalid.
    """
    if os.path.isfile(path):
        with open(path, 'r' ,encoding=encoding, errors='ignore') as json_file:
            json_str = json_file.read()
        return json.loads(json_str)
    return invalidPathReturn

def save_excel(path:str, obj:List[List[str]], columns:List[str], encoding:str = 'utf-8', forceUpdate = True):
    """
    Save a list of lists as an Excel file.

    Args:
        path (str): The path where the Excel file will be saved.
        obj (List[List[str]]): The list of lists to be saved as an Excel file.
        columns (List[str]): The column names for the Excel file.
        encoding (str, optional): The encoding of the Excel file. Defaults to 'utf-8'.
        forceUpdate (bool, optional): If True, the file will be saved even if it already exists. Defaults to True.

    Returns:
        bool: True if the file was successfully saved, False otherwise.
    """
    if forceUpdate or not os.path.isfile(path):
        df = pd.DataFrame(obj, columns=columns)
        df.to_excel(path, encoding = encoding)
        return True
    return False

def read_excel(path:str, sheet_name:str = None, ignore_head:bool = True,
                  ignore_first_col:bool = True, invalid_path_return = None):
    """
    Reads an Excel file and returns a pandas DataFrame.
    
    Args:
        path (str): The path to the Excel file.
        sheet_name (str, optional): The name of the sheet to read. Defaults to None.
        ignore_head (bool, optional): Whether to ignore the first row (header) of the sheet. Defaults to True.
        ignore_first_col (bool, optional): Whether to ignore the first column of the sheet. Defaults to True.
        invalid_path_return (Any, optional): The value to return if the path is invalid. Defaults to None.
    
    Returns:
        pandas.DataFrame: The DataFrame containing the data from the Excel file.
            or:
        invalid_path_return (Any): The value specified if the path is invalid.
    """
    if os.path.isfile(path):
        df = pd.read_excel(path, sheet_name)
        if ignore_head:
            df = df.iloc[1:]  # 忽略第一行（表头）
        if ignore_first_col:
            df = df.iloc[:, 1:]  # 忽略第一列
        return df
    return invalid_path_return

def write_sheets(path:str, sheets:Dict[str, pd.DataFrame]):
    """
    Write multiple sheets to an Excel file.

    Args:
        path (str): The path to the Excel file.
        sheets (Dict[str, pd.DataFrame]): A dictionary mapping sheet names to dataframes.

    Returns:
        None
    """
    with pd.ExcelWriter(path) as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name)

def update_excel(path:str, sheets:Dict[str, pd.DataFrame] = None):
    """
    Updates an Excel file with the given path by adding or modifying sheets.

    Args:
        path (str): The path of the Excel file.
        sheets (Dict[str, pd.DataFrame], optional): A dictionary of sheets to add or modify. 
            The keys are sheet names and the values are pandas DataFrame objects. 
            Defaults to None.

    Returns:
        Union[Dict[str, pd.DataFrame], None]: If the Excel file exists and sheets is None, 
            returns a dictionary containing all the sheets in the Excel file. 
            Otherwise, returns None.

    Raises:
        None
    """
    if os.path.isfile(path):
        dfs = pd.read_excel(path, sheet_name=None)
        if sheets is None:
            return dfs
        else:
            for sheet in sheets:
                if isinstance(sheets[sheet], pd.DataFrame):
                    dfs[sheet] = sheets[sheet]
            write_sheets(path, dfs)
    elif sheets is not None:
        print(f'path is not a file : {path:s}, writing sheets to the file of path')
        write_sheets(path, sheets)
        
def convert_pdf_to_txt(path: str, backend = 'PyPDF2') -> str:
    """
    Convert a PDF file to a text file.

    Args:
        path: The path to the PDF file.
        backend: The backend library to use for PDF conversion. 
            - 'PyPDF2' is the default.
            - 'pdfminer'.

    Returns:
        The extracted text from the PDF file as a string.

    Raises:
        NotImplementedError: If the specified backend is not supported.
    """
    if not os.path.isfile(path):
        return put_err(f'{path:s} does not exist', f'{path:s} does not exist')
    if backend == 'PyPDF2':
        import PyPDF2
        with open(path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            return '\n'.join([page.extract_text() for page in reader.pages])
    elif backend == 'pdfminer':        
        from pdfminer.high_level import extract_text
        return extract_text(path)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    # dev code
    pass
        