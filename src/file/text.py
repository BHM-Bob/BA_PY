'''
Author: BHM-Bob 2262029386@qq.com
Date: 2022-04-06 09:20:45
LastEditors: BHM-Bob
LastEditTime: 2022-10-19 23:06:34
Description: 
'''
import chardet


def DecodeByteStr(bits:bytes):
    adchar = chardet.detect(bits[:(1000 if len(bits) > 1000 else len(bits))])['encoding']
    if adchar == 'gbk' or adchar == 'GBK' or adchar == 'GB2312':
        true_text = bits.decode('GB2312', "ignore")
    else:
        true_text = bits.decode('utf-8', "ignore")
    return true_text