'''
Author: BHM-Bob 2262029386@qq.com
Date: 2022-11-01 19:09:54
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2025-01-09 16:10:25
Description: 
'''
import collections
import gzip
import os
import pickle
import platform
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, List, Union
from zipfile import ZipFile

try:
    import ujson as json
except:
    import json

import pandas as pd

if __name__ == '__main__':
    # dev mode
    from mbapy_lite.base import put_err
else:
    from .base import put_err


def get_paths_with_extension(folder_path: str, file_extensions: List[str],
                             recursive: bool = True, name_substr: str = '',
                             search_name_in_dir: bool = False) -> List[str]:
    """
    Returns a list of file paths within a given folder that have a specified extension.

    Args:
        - folder_path (str): The path of the folder to search for files.
        - file_extensions (List[str]): A list of file extensions to filter the search by.
            If it is empty, accept all extensions.
        - recursive (bool, optional): Whether to search subdirectories recursively. Defaults to True.
        - name_substr (str, optional): Sub-string in file-name (NOT INCLUDE TYPE SUFFIX). Defualts to '', means not specific.
        - search_name_in_dir (bool, optional): Whether to search file names in directory. Defaults to False.

    Returns:
        List[str]: A list of file paths that match the specified file extensions.
    """
    file_paths = []
    for name in os.listdir(folder_path): # do not use os.walk() to avoid FXXK files updates
        path = os.path.join(folder_path, name)
        if os.path.isfile(path) and (any(path.endswith(extension) for extension in file_extensions) or (not file_extensions)):
            if (not name_substr) or (name_substr and name_substr in path.split(os.path.sep)[-1]):
                file_paths.append(path)
        elif search_name_in_dir and os.path.isdir(path) and name_substr in path:
            file_paths.append(path)
        if recursive and os.path.isdir(path):
            file_paths.extend(get_paths_with_extension(path, file_extensions, recursive, name_substr, search_name_in_dir))
    return file_paths

def get_dir(root: str, min_item_num: int = 0, max_item_num: int = None,
            file_extensions: List[str] = [],
            recursive: bool = True,
            dir_name_substr: str = '', item_name_substr: str = '',) -> List[str]:
    """
    Returns a list of file paths within a given folder that have a specified extension.

    Args:
        - root (str): The root directory path.
        - min_item_num (int, optional): The minimum number of items in a directory to be included. Defaults to 0.
        - max_item_num (int, optional): The maximum number of items in a directory to be included. Defaults to None, means no limit.
        - file_extensions (list[str]): specific file types string (without '.'), if None or [], means all types.
        - recursive (bool, optional): Whether to recursively search subdirectories. Defaults to True.
        - dir_name_substr (str): Sub-string in directory name.
        - item_name_substr (str): Sub-string in file name.

    Returns:
        List[str]: A list of file paths that match the specified file extensions.
    """
    file_paths = []
    for name in os.listdir(root): # do not use os.walk() to avoid FXXK files updates
        path = os.path.join(root, name)
        # check is dir
        if not os.path.isdir(path):
            continue
        items = os.listdir(path)
        items_num = len(items)
        # recursive search before skip 
        if os.path.isdir(path) and recursive and any(os.path.isdir(os.path.join(path, item)) for item in items):
            file_paths.extend(get_dir(path, min_item_num, max_item_num,
                                      file_extensions, recursive,
                                      dir_name_substr, item_name_substr))
        # check dir's name sub-string
        if dir_name_substr not in name:
            continue
        # check items num
        if items_num < min_item_num or (max_item_num is not None and items_num > max_item_num):
            continue
        # check items' name sub-string
        if item_name_substr and not any(item_name_substr in item for item in items):
            continue
        # check files' type
        if file_extensions and not any(any(filename.endswith(extension) for extension in file_extensions) \
            for filename in items if os.path.isfile(os.path.join(path, filename))):
                continue
        # add dir path to result
        file_paths.append(path)
    return file_paths

def format_file_size(size_bits: int, return_str: bool = True):
    """
    Formats a file size in bits to a human-readable format.
    
    Parameters:
        - size_bits (int): The size of the file in bits.
        - return_str (bool, optional): Whether to return the size as a string or as a tuple of size and unit. Defaults to True.

    Returns:
        - If return_str is True, returns a string of the size and unit.
        - If return_str is False, returns a tuple of the size and unit.
    """
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
                           extract_sub_dir: bool = True, join_str:str = ' ',
                           preffix: str = '', file_result: List[str] = []):
    """
    Move all files in subdirectories to the root directory and add the subdirectory name as a prefix to the file name.

    Args:
        - root (str): The root directory path.
        - file_extensions (list[str]): specific file types string (without '.'), if None, means all types.
        - extract_sub_dir (bool, optional): Whether to recursively extract files from subdirectories.
            If set to False, only files in the immediate subdirectories will be extracted. Defaults to True.
        - join_str (str): string for link prefix and the file name.
        - preffix (str): the prefix for the file name.
        - file_result (list): the list to store the extracted file paths.

    Returns:
        None
    """
    for dirpath, dirnames, filenames in os.walk(root):
        if not extract_sub_dir or dirpath == root:
            continue
        else:
            for dirname in dirnames:
                preffix += (dirname + join_str)
                file_result.extend(extract_files_from_dir(
                    os.path.join(root, dirname), file_extensions,
                    extract_sub_dir, join_str, preffix))
        for filename in filenames:
            if not file_extensions or any(filename.endswith(extension) for extension in file_extensions):
                new_filename = preffix + join_str + filename
                src_path = os.path.join(dirpath, filename)
                dest_path = os.path.join(root, new_filename)
                shutil.move(src_path, dest_path)
            file_result.append(dest_path)
    return file_result

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
    win_prefix = ''
    # AVOID WINDOWS PATH PREFIX: such as 'C:\'
    if platform.system().lower() == 'windows' and os.path.basename(path) != path:
        if len(path) >= 2 and path[1] == ':':
            win_prefix, path = path[:2], path[2:]
    for invalid_chr in invalid_chrs:
        path = path.replace(invalid_chr, valid_chrs)
    return win_prefix + path

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


_filetype2opts_ = {
    'txt': {'mode': '', 'way': 'str', 'encoding': 'utf-8'},
    'pdb': {'mode': '', 'way': 'str', 'encoding': 'utf-8'},
    'json': {'mode': '', 'way': 'json', 'encoding': 'utf-8'},
    'yml': {'mode': '', 'way': 'yml', 'encoding': 'utf-8'},
    'yaml': {'mode': '', 'way': 'yaml', 'encoding': 'utf-8'},
    'pkl': {'mode': 'b', 'way': 'pkl', 'encoding': None},
    'csv': {'mode': '', 'way': 'csv', 'encoding': 'utf-8'},
    'xlsx': {'mode': 'b', 'way': 'excel', 'encoding': 'utf-8'},
}


def opts_file(path:str, mode:str = 'r', encoding:str = 'utf-8',
              way:str = 'str', data = None, kwgs: Dict = None, **kwargs):
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
            - 'zip': Read/write the data as a ZIP file, return dict: key is file path in zip, value is the data in the file.
            - '__auto__': Automatically determine the way based on the file extension, support in _filetype2opts_.
        - data (Any, optional): The data to be written to the file. Only applicable in write mode. Defaults to None.
        - kwgs (dict): Additional keyword arguments to be passed to the third-party read/write function.
        - kwargs (dict): Additional arguments to be passed to the open() function.

    Returns:
        list or str or dict or None: The data read from the file, or None if the file was opened in write mode and no data was provided.
        
    Errors:
        - return None if the path is not a valid file path for read.
        - return None if the mode or way is not valid.
    """
    # check kwgs
    kwgs = {} if kwgs is None else kwgs
    # check mode
    if 'b' not in mode:
        kwargs.update(encoding=encoding)
    # set open_fn depend on way
    open_fn_dict = {'zip': ZipFile, 'tar': tarfile.open}
    if way in open_fn_dict:
        open_fn = open_fn_dict[way]
        if 'encoding' in kwargs:
            del kwargs['encoding']
    else:
        open_fn = open
    if way == '__auto__':
        opts_kwgs = _filetype2opts_.get(path.split('.')[-1],
                                        {'mode': 'b', 'way': 'str', 'encoding': None})
        mode, way, encoding = mode + opts_kwgs['mode'], opts_kwgs['way'], opts_kwgs['encoding']
        if opts_kwgs['encoding'] is None:
            del kwargs['encoding']
    # perform read or write
    with open_fn(path, mode, **kwargs) as f:
        if 'r' in mode and os.path.isfile(path):
            if way == 'lines':
                return f.readlines()
            elif way == 'str':
                return f.read()
            elif way == 'json':
                return json.loads(f.read(), **kwgs)
            elif way in ['yml', 'yaml']:
                import yaml
                kwgs['Loader'] = kwgs.get('Loader', yaml.FullLoader)
                return yaml.load(f, **kwgs)
            elif way == 'pkl':
                if kwargs.get('gzip', False):
                    f = gzip.GzipFile(fileobj=f)
                if os.path.getsize(path) == 0:
                    return None
                return pickle.load(f, **kwgs)
            elif way == 'csv':
                return pd.read_csv(f, **kwgs)
            elif way in ['excel', 'xlsx', 'xls']:
                return pd.read_excel(f, **kwgs)
            elif way in {'zip', 'tar'}:
                with tempfile.TemporaryDirectory() as tmpdirname:
                    f.extractall(tmpdirname)
                    files = {}
                    for path in get_paths_with_extension(tmpdirname, []):
                        sub_kwgs = kwgs.get(Path(path).suffix, {'way': '__auto__'})
                        files[path[len(tmpdirname)+1:]] = opts_file(path, **sub_kwgs)
                    return files
        elif ('w' in mode or 'a' in mode) and data is not None\
                and way in ['lines','str']: 
            if way == 'lines':
                return f.writelines(data)
            elif way == 'str':
                return f.write(data)
        elif 'w' in mode and (data is not None or way == 'pkl'):
            if way == 'json':
                return f.write(json.dumps(data, **kwgs))
            elif way == 'pkl':
                data = pickle.dumps(data, **kwgs)
                if kwargs.get('gzip', False):
                    data = gzip.compress(data)
                f.write(data)
                return data
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
    """
    Check if the given data can be serialized as JSON.

    Parameters:
        - data: The data to be checked.

    Returns:
        - bool: True if the data can be serialized as JSON, False otherwise.
    """
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

def write_sheets(path:str, sheets:Dict[str, pd.DataFrame], writer_kwgs = {}, **kwargs):
    """
    Write multiple sheets to an Excel file.

    Args:
        - path (str): The path to the Excel file.
        - sheets (Dict[str, pd.DataFrame]): A dictionary mapping sheet names to dataframes.
        - writer_kwgs (dict): Additional keyword arguments to be passed to the ExcelWriter.
        - kwargs (dict): Additional keyword arguments to be passed to the to_excel() method of pandas.

    Returns:
        None
    """
    with pd.ExcelWriter(path, **writer_kwgs) as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, **kwargs)

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
    
    
__all__ = [
    'get_paths_with_extension',
    'get_dir',
    'format_file_size',
    'extract_files_from_dir',
    'replace_invalid_path_chr',
    'get_valid_file_path',
    'opts_file',
    'detect_byte_coding',
    'get_byte_coding',
    'decode_bits_to_str',
    'is_jsonable',
    'save_json',
    'read_json',
    'write_sheets',
    'update_excel',
]
    

if __name__ == '__main__':
    # dev code
    contents = opts_file('data_tmp/files/result.zip', way='zip')
    dirs = get_dir('.', min_item_num=10, dir_name_substr='scripts', recursive=True)