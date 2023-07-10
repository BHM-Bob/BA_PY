'''
Author: BHM-Bob 2262029386@qq.com
Date: 2022-11-01 19:09:54
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-07-10 16:35:46
Description: 
'''
import chardet
import json
import os
from typing import List, Dict

import pandas as pd

if __name__ == '__main__':
    # dev mode
    from mbapy.base import put_err
else:
    # release mode
    from .base import put_err

def replace_invalid_path_chr(path:str, valid_chrs:str = '_'):
    """
    Replaces any invalid characters in a given path with a specified valid character.

    Args:
        path (str): The path string to be checked for invalid characters.
        valid_chrs (str, optional): The valid characters that will replace any invalid characters in the path. Defaults to '_'.

    Returns:
        str: The path string with all invalid characters replaced by the valid character.
    """
    invalid_chrs = ':*?"<>|'
    for invalid_chr in invalid_chrs:
        path = path.replace(invalid_chr, '_')
    return path

def read_bits(path:str):
    with open(path, 'rb') as f:
        return f.read()

def read_text(path:str, decode:str = 'utf-8', way:str = 'lines'):
    """
    Reads the contents of a text file at the specified path.

    Parameters:
        path (str): The path of the text file to read.
        decode (str, optional): The decoding format to use. Defaults to 'utf-8'.
        way (str, optional): The way to read the file. Defaults to 'lines'.
            Possible values:
                - 'lines': Returns a list of lines from the file.
                - 'str': Returns the entire contents of the file as a string.
                - 'json': Returns the contents of the file as a JSON object.

    Returns:
        list or str or dict: The contents of the file based on the specified 'way' parameter.
    """
    with open(path, 'r', encoding=decode) as f:
        if way == 'lines':
            return f.readlines()
        elif way == 'str':
            return f.read()
        elif way == 'json':
            return json.loads(f.read())

def detect_byte_coding(bits:bytes):
    """
    Detects the byte coding of a given byte sequence and decodes it accordingly. 

    :param bits: The byte sequence to be decoded.
    :type bits: bytes
    
    :return: The decoded string.
    :rtype: str
    """
    adchar = chardet.detect(bits[:(1000 if len(bits) > 1000 else len(bits))])['encoding']
    if adchar == 'gbk' or adchar == 'GBK' or adchar == 'GB2312':
        true_text = bits.decode('GB2312', "ignore")
    else:
        true_text = bits.decode('utf-8', "ignore")
    return true_text

def get_byte_coding(bits:bytes, max_detect_len = 1000):
    """
    Given a bytes object 'bits' and an optional integer 'max_detect_len', 
    this function detects the encoding of the byte string 'bits' using 
    the chardet module. It returns a string representing the detected 
    encoding of the byte string. If 'max_detect_len' is given, it limits the 
    number of bytes that chardet analyzes for encoding detection.
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

def save_json(path:str, obj, encoding:str = 'utf-8', forceUpdate = True):
    """
    Saves a Python object to a JSON file located at `path`. If the file does not exist or `forceUpdate` is `True`, 
    it will overwrite the existing file with the provided object. The encoding of the file can be specified using the 
    `encoding` parameter (default is 'utf-8').

    :param path: A string representing the file path to save the JSON object.
    :param obj: The Python object to be saved as a JSON file.
    :param encoding: A string representing the encoding of the file (default is 'utf-8').
    :param forceUpdate: A boolean indicating whether to overwrite the file if it already exists (default is True).
    :return: None
    """
    if forceUpdate or not os.path.isfile(path):
        json_str = json.dumps(obj, indent=1)
        with open(path, 'w' ,encoding=encoding, errors='ignore') as json_file:
            json_file.write(json_str)
def read_json(path:str, encoding:str = 'utf-8', invalidPathReturn = None):
    """
    Given a path to a JSON file, reads its contents and returns the parsed JSON object.
    
    :param path: The path to the JSON file.
    :type path: str
    :param encoding: The encoding of the file. Default is 'utf-8'.
    :type encoding: str
    :param invalidPathReturn: The value to return in case the file path is invalid. Default is None.
    :return: The parsed JSON object if the file exists, otherwise invalidPathReturn.
    :rtype: dict or list or None
    """
    if os.path.isfile(path):
        with open(path, 'r' ,encoding=encoding, errors='ignore') as json_file:
            json_str = json_file.read()
        return json.loads(json_str)
    return invalidPathReturn

def save_excel(path:str, obj:List[List[str]], columns:List[str], encoding:str = 'utf-8', forceUpdate = True):
    """
    Saves a list of lists as an Excel file at the given path. 

    :param path: A string representing the path where the Excel file will be saved.
    :param obj: A list of lists that will be saved as the Excel file.
    :param columns: A list of strings representing the column names.
    :param encoding: A string representing the encoding to use when saving the Excel file. Default is 'utf-8'.
    :param forceUpdate: A boolean indicating whether to update the Excel file if it already exists. Default is True.

    :return: This function does not return anything.
    """
    if forceUpdate or not os.path.isfile(path):
        df = pd.DataFrame(obj, columns=columns)
        df.to_excel(path, encoding = encoding)
def read_excel(path:str, ignoreHead:bool = True,
                  ignoreFirstCol:bool = True, invalidPathReturn = None):
    """
    Reads an excel file located at the given path and returns a pandas DataFrame object.
    
    Args:
        path (str): The path to the excel file.
        ignoreHead (bool, optional): Whether or not to ignore the first row of the file. Defaults to True.
        ignoreFirstCol (bool, optional): Whether or not to ignore the first column of the file. Defaults to True.
        invalidPathReturn (optional): The object to return when the path is invalid. Defaults to None.
    
    Returns:
        pandas.DataFrame: A DataFrame object containing the data from the excel file, or invalidPathReturn if the path is invalid.
    """
    if os.path.isfile(path):
        df = pd.read_excel(path, )
        return df
    return invalidPathReturn

def write_sheets(path:str, sheets:Dict[str, pd.DataFrame]):
    """
    Writes multiple pandas DataFrames to an Excel file with multiple sheets.

    :param path: A string representing the path to the Excel file.
    :param sheets: A dictionary where the keys are strings representing the sheet names and the values are pandas DataFrames to be written to the sheet.
    :return: None
    """
    with pd.ExcelWriter(path) as f:
        for sheet in sheets:
            sheets[sheet].to_excel(path, sheet_name=sheet)    

def update_excel(path:str, sheets:Dict[str, pd.DataFrame] = None):
    """
    Updates an Excel file with new data.

    :param path: A string representing the file path of the Excel file.
    :param sheets: A dictionary containing the sheet names and corresponding pandas DataFrames to be written to the Excel file. Defaults to None.
    :return: If the Excel file is found, a dictionary of pandas DataFrames containing the sheets in the Excel file. If the Excel file is not found, None is returned.
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
        backend: The backend library to use for PDF conversion. Defaults to 'PyPDF2'.

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
    else:
        raise NotImplementedError

if __name__ == '__main__':
    # dev code
    pdf_path = r"./data_tmp/DiffBP Generative Diffusion of 3D Molecules for Target Protein Binding.pdf"
    print(convert_pdf_to_txt(pdf_path))
    
