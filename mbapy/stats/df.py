'''
Author: BHM-Bob 2262029386@qq.com
Date: 2023-04-10 20:59:26
LastEditors: BHM-Bob
LastEditTime: 2023-04-30 18:56:49
Description: pd.dataFrame utils
'''

import pandas as pd
import numpy as np

from mbapy.plot import pro_bar_data, get_df_data, sort_df_factors
from mbapy.file import update_excel

def remove_simi(tag:str, df:pd.DataFrame, sh:float = 1., 
                backend:str = 'numpy-array', tensor = None, device = 'cuda'):
    """
    给定一组数，去除一些(最小数目)数，使任意两数差的绝对值大于或等于阈值\n
    Given a set of numbers, remove some (minimum number) of numbers so that the absolute value of the difference between any two numbers is greater than or equal to the threshold\n
    算法模仿自Needleman-Wushsch序列对比算法\n
    Parameters
    ----------
    backend : 
        'numpy-mat': a n-n mat will be alloc\n
        'numpy-array': only operate on a n shape arrray\n
        'torch-array': only operate on a n shape arrray\n
    Examples
    --------
    >>> df = pd.DataFrame({'d':[1, 2, 3, 3, 5, 6, 8, 13]})\n
    >>> print(remove_simi('d', df, 2.1, 'numpy'))\n
        d\n
    0   1\n
    4   5\n
    6   8\n
    7  13\n
    """
    ndf = df.sort_values(by = tag, ascending=True)
    to_remove_idx = []
    if backend  == 'numpy-mat':
        arr = np.array(ndf[tag]).reshape([1, len(ndf[tag])])
        mat = arr.repeat(arr.shape[1], axis = 0) - arr.transpose(1, 0).repeat(arr.shape[1], axis = 1)
        i, j, k = 0, 0, mat.shape[0]
        while i < k and j < k:
            if i == j:
                j += 1
            elif mat[i][j] < sh:
                to_remove_idx.append(j)
                mat[i][j] = mat[i][j-1]#skip for next element in this row
                mat[j] = arr - mat[i][j]#skip for row j
                j += 1
            elif mat[i][j] >= sh:
                i += 1
    elif backend == 'torch-array':
        try:
            import torch
        except:
            raise ImportError('no torch available')
        arr = tensor if tensor is not None else torch.tensor(ndf[tag], device = device, dtype = torch.float32).view(-1)
        @torch.jit.script
        def step_scan(x:torch.Tensor, to_remove:list[int], sh:float):
            i = 0
            while i < x.shape[0]-1:
                if x[i+1] - x[i] < sh:
                    x[i+1] = x[i]
                    to_remove.append(i+1)
                i += 1
            return to_remove
        to_remove_idx = step_scan(arr, to_remove_idx, sh)
    elif backend == 'numpy-array':
        arr = np.array(ndf[tag]).reshape([len(ndf[tag])])
        i = 0
        while i < arr.shape[0]-1:
            if arr[i+1] - arr[i] < sh:
                arr[i+1] = arr[i]
                to_remove_idx.append(i+1)
            i += 1
    ndf.drop(labels = to_remove_idx, inplace=True)
    return ndf, to_remove_idx

def interp(long_one:pd.Series, short_one:pd.Series):
    """
    给定两个pd.Series，一长一短，用线性插值给短的插值，使其长度与长的pd.Series一样\n
    Given two pd.Series, one long and one short, use linear interpolation to give the short one the same length as the long pd.Series\n
    """
    assert len(long_one) > len(short_one)
    short_one_idx = np.array(np.arange(short_one.shape[0])*(long_one.shape[0]/short_one.shape[0]), dtype=np.int32)
    if short_one_idx[-1] < long_one.shape[0]-1:
        short_one_idx[-1] = long_one.shape[0]-1
    return np.interp(np.arange(long_one.shape[0]), short_one_idx, short_one)

