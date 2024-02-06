'''
Author: BHM-Bob 2262029386@qq.com
Date: 2022-11-01 19:09:54
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2024-02-06 15:02:47
Description: 
'''
import collections
import os
import pickle
import platform
import shutil
from pathlib import Path
from typing import Dict, List, Union

try:
    import ujson as json
except:
    import json

import pandas as pd

if __name__ == '__main__':
    # dev mode
    from mbapy.base import (check_parameters_path, format_secs,
                            get_default_for_bool, parameter_checker, put_err)

    # video and image functions assembly
    try:
        if 'MBAPY_AUTO_IMPORT_TORCH' in os.environ and\
            os.environ['MBAPY_AUTO_IMPORT_TORCH'] == 'True':
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
        if 'MBAPY_AUTO_IMPORT_TORCH' in os.environ and\
            os.environ['MBAPY_AUTO_IMPORT_TORCH'] == 'True':
                import cv2
                import torch

                from .file_utils.image import *
                from .file_utils.video import *
    except:# if cv2 or torch is not installed, skip
        pass
    
def get_paths_with_extension(folder_path: str, file_extensions: List[str],
                             recursive: bool = True, name_substr: str = '') -> List[str]:
    """
    Returns a list of file paths within a given folder that have a specified extension.

    Args:
        - folder_path (str): The path of the folder to search for files.
        - file_extensions (List[str]): A list of file extensions to filter the search by.
            If it is empty, accept all extensions.
        - recursive (bool, optional): Whether to search subdirectories recursively. Defaults to True.
        - name_substr (str, optional): Sub-string in file-name. Defualts to '', means not specific.

    Returns:
        List[str]: A list of file paths that match the specified file extensions.
    """
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.endswith(extension) for extension in file_extensions) or (not file_extensions):
                if (not name_substr) or (name_substr and name_substr in file.split(os.path.sep)[-1]):
                    file_paths.append(os.path.join(root, file))
        if recursive:
            for dir in dirs:
                file_paths.extend(get_paths_with_extension(os.path.join(root, dir),
                                                           file_extensions, recursive, name_substr))
    return file_paths

def format_file_size(size_bits: int, return_str: bool = True):
    n = 0
    units = {0: '', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB', 5: 'PB', 6: 'EB', 7: 'ZB', 8: 'YB'}
    
    while size_bits > 1024:
        size_bits /= 1024
        n += 1
        
    if return_str:
        return f"{round(size_bits, 2)} {units[n]}"
    else:
        return size_bits, units[n]

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
    invalid_chrs = ':*?"<>|\n\t'
    for invalid_chr in invalid_chrs:
        path = path.replace(invalid_chr, valid_chrs)
    return path

def get_valid_file_path(path:str, valid_chrs:str = '_', valid_len:int = 250,
                        return_Path: bool = False):
    """
    Get a valid file path.

    Args:
        path (str): The path to process.
        valid_chrs (str, optional): Valid characters for the path. Default is '_'.
        valid_len (int, optional): The maximum valid length of the path. Default is 250.
        return_Path (bool, optional): Whether to return a Path object or not. Default is False.

    Returns:
        Union[str, Path]: The validated file path.

    """
    path = replace_invalid_path_chr(path, valid_chrs)
    if platform.system().lower() == 'windows' and len(path) > valid_len:
        suffix = Path(path).suffix
        valid_len = valid_len - len(suffix)
        path = path[:valid_len] + suffix
    return path if not return_Path else Path(path)

def opts_file(path:str, mode:str = 'r', encoding:str = 'utf-8',
              way:str = 'str', data = None, kwgs: Dict = {}, **kwargs):
    """
    A function that reads or writes data to a file based on the provided options.

    Parameters:
        - path (str): The path to the file.
        - mode (str, optional): The mode in which the file should be opened. Defaults to 'r'.
            - 'r': Read mode.
            - 'w': Write mode.
            - 'a': Append mode. Only applicable with 'str' and 'lines' way.
        - encoding (str, optional): The encoding of the file. Defaults to 'utf-8'.
        - way (str, optional): The way in which the data should be read or written. Defaults to 'str'.
            - 'str': Read/write the data as a string.
            - 'lines': Read/write the data as a list of lines.
            - 'json': Read/write the data as a JSON object.
            - 'yml'/'yaml': Read/write the data as a YAML file.
            - 'pkl': Read/write the data as a Python object (using pickle).
            - 'csv': Read/write the data as a CSV file.
            - 'excel' or 'xlsx' or 'xls': Read/write the data as an Excel file.
        - data (Any, optional): The data to be written to the file. Only applicable in write mode. Defaults to None.
        - kwgs (dict): Additional keyword arguments to be passed to the third-party read/write function.
        - kwargs (dict): Additional arguments to be passed to the open() function.

    Returns:
        list or str or dict or None: The data read from the file, or None if the file was opened in write mode and no data was provided.
        
    Errors:
        - return None if the path is not a valid file path for read.
        - return None if the mode or way is not valid.
    """
    if 'b' not in mode:
        kwargs.update(encoding=encoding)
    with open(path, mode, **kwargs) as f:
        if 'r' in mode and os.path.isfile(path):
            if way == 'lines':
                return f.readlines()
            elif way == 'str':
                return f.read()
            elif way == 'json':
                return json.loads(f.read(), **kwgs)
            elif way in ['yml', 'yaml']:
                import yaml
                return yaml.load(f, **kwgs)
            elif way == 'pkl':
                return pickle.load(f, **kwgs)
            elif way == 'csv':
                return pd.read_csv(f, **kwgs)
            elif way in ['excel', 'xlsx', 'xls']:
                return pd.read_excel(f, **kwgs)
        elif ('w' in mode or 'a' in mode) and data is not None\
                and way in ['lines','str']: 
            if way == 'lines':
                return f.writelines(data)
            elif way == 'str':
                return f.write(data)
        elif 'w' in mode and data is not None:
            if way == 'json':
                return f.write(json.dumps(data, **kwgs))
            elif way == 'pkl':
                return pickle.dump(data, f, **kwgs)
            elif way in ['yml', 'yaml']:
                import yaml
                return yaml.dump(data, f, **kwgs)
            elif way == 'csv':
                return data.to_csv(f, **kwgs)
            elif way in ['excel', 'xlsx', 'xls']:
                return data.to_excel(f, **kwgs)
        else:
            return put_err(f"Invalid mode or way for file {path}. mode={mode}, way={way}.")

def detect_byte_coding(bits:bytes):
    """
    This function takes a byte array as input and detects the encoding of the first 1000 bytes (or less if the input is shorter).
    It then decodes the byte array using the detected encoding and returns the resulting text.

    Parameters:
        bits (bytes): The byte array to be decoded.

    Returns:
        str: The decoded text.
    """
    # for FAST LOAD
    import chardet
    
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
    # for FAST LOAD
    import chardet
    
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
        return all((isinstance(k, str) and is_jsonable(v)) for k, v in data.items())
    elif isinstance(data, collections.abc.Sequence):
        return all(is_jsonable(item) for item in data)
    else:
        return False

def save_json(path:str, obj, encoding:str = 'utf-8', forceUpdate = True, ensure_ascii = False):
    """
    Saves an object as a JSON file at the specified path.

    Parameters:
        - path (str): The path where the JSON file will be saved.
        - obj: The object to be saved as JSON.
        - encoding (str): The encoding of the JSON file. Default is 'utf-8'.
        - forceUpdate (bool): Determines whether to overwrite an existing file at the specified path. Default is True.
        - ensure_ascii (bool): param for json.dumps

    Returns:
        None
    """
    if forceUpdate or not os.path.isfile(path):
        json_str = json.dumps(obj, indent=1, ensure_ascii=ensure_ascii)
        with open(path, 'w', encoding=encoding, errors='ignore') as f:
            f.write(json_str)
            
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
        with open(path, 'r' ,encoding=encoding, errors='ignore') as f:
            json_str = f.read()
        return json.loads(json_str)
    return invalidPathReturn

def save_yaml(path:str, obj, indent = 2, encoding:str = 'utf-8',
              force_update = True, allow_unicode = True):
    """
    Saves an object as a YAML file at the specified path.

    Parameters:
        - path (str): The path where the YAML file will be saved.
        - obj: The object to be saved as YAML.
        - indent (int): indent for YAML.
        - encoding (str): The encoding of the YAML file. Default is 'utf-8'.
        - forceUpdate (bool): Determines whether to overwrite an existing file at the specified path. Default is True.
        - allow_unicode (bool): param for yaml.dump

    Returns:
        None
    """
    import yaml
    if force_update or not os.path.isfile(path):
        with open(path, 'w' ,encoding=encoding, errors='ignore') as f:
            f.write(yaml.dump(obj, indent=indent, allow_unicode=allow_unicode))
            
def read_yaml(path:str, encoding:str = 'utf-8', invalidPathReturn = None):
    """
    Read a YMAL file from the given path and return the parsed YMAL data.

    Parameters:
        path (str): The path to the YMAL file.
        encoding (str, optional): The encoding of the file. Defaults to 'utf-8'.
        invalidPathReturn (any, optional): The value to return if the path is invalid. Defaults to None.

    Returns:
        dict: The parsed YMAL data.
        invalidPathReturn (any): The value passed as `invalidPathReturn` if the path is invalid.
    """
    import yaml
    if os.path.isfile(path):
        with open(path, 'r' ,encoding=encoding, errors='ignore') as fh:
            return yaml.load(fh, Loader=yaml.FullLoader)
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

def read_excel(path:str, sheet_name:Union[None, str, List[str]] = None, ignore_first_row:bool = False,
               ignore_first_col:bool = False, invalid_path_return = None, **kwargs):
    """
    Reads an Excel file and returns a pandas DataFrame.
    
    Args:
        path (str): The path to the Excel file.
        sheet_name (str, list[str], optional): The name of the sheet to read. Defaults to None.
        ignore_first_row (bool, optional): Whether to ignore the first row (header) of the sheet. Defaults to True.
        ignore_first_col (bool, optional): Whether to ignore the first column of the sheet. Defaults to True.
        invalid_path_return (Any, optional): The value to return if the path is invalid. Defaults to None.
    
    Returns:
        pandas.DataFrame: The DataFrame containing the data from the Excel file.
            or:
        invalid_path_return (Any): The value specified if the path is invalid.
    """
    # Forward Compatibility
    if 'ignore_head' in kwargs:
        ignore_first_row = kwargs['ignore_head']
    # if read head
    header = None if ignore_first_row else 'infer'
    # read excel
    if os.path.isfile(path):
        df = pd.read_excel(path, sheet_name, header=header)
        if isinstance(df, dict):
            return {k:v.iloc[int(ignore_first_row):, int(ignore_first_col):] for k,v in df.items()}
        else:
            return df.iloc[int(ignore_first_row):, int(ignore_first_col):]
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
    convert_pdf_to_txt(r'./data_tmp\papers\A review of the clinical efficacy of linaclotide in irritable bowel syndrome with constipation.pdf')