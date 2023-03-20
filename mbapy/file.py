'''
Author: BHM-Bob 2262029386@qq.com
Date: 2022-11-01 19:09:54
LastEditors: BHM-Bob
LastEditTime: 2023-03-21 00:35:24
Description: 
'''
import chardet
import json
import os

import pandas as pd


def detect_byte_coding(bits:bytes):
    adchar = chardet.detect(bits[:(1000 if len(bits) > 1000 else len(bits))])['encoding']
    if adchar == 'gbk' or adchar == 'GBK' or adchar == 'GB2312':
        true_text = bits.decode('GB2312', "ignore")
    else:
        true_text = bits.decode('utf-8', "ignore")
    return true_text

def save_json(path:str, obj, encoding:str = 'utf-8', forceUpdate = True):
    if forceUpdate or not os.path.isfile(path):
        json_str = json.dumps(obj, indent=1)
        with open(path, 'w' ,encoding=encoding, errors='ignore') as json_file:
            json_file.write(json_str)
def read_json(path:str, encoding:str = 'utf-8', invalidPathReturn = None):
    if os.path.isfile(path):
        with open(path, 'r' ,encoding=encoding, errors='ignore') as json_file:
            json_str = json_file.read()
        return json.loads(json_str)
    return invalidPathReturn
def save_excel(path:str, obj:list[list[str]], columns:list[str], encoding:str = 'utf-8', forceUpdate = True):
    if forceUpdate or not os.path.isfile(path):
        df = pd.DataFrame(obj, columns=columns)
        df.to_excel(path, encoding = encoding)
def read_excel(path:str, ignoreHead:bool = True,
                  ignoreFirstCol:bool = True, invalidPathReturn = None):
    if os.path.isfile(path):
        df = pd.read_excel(path, )
        return df
    return invalidPathReturn