'''
Author: BHM-Bob 2262029386@qq.com
Date: 2023-04-10 20:59:26
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-07-10 16:42:29
Description: pd.dataFrame utils
'''
import itertools
from functools import wraps
from typing import Dict, List

import numpy as np
import pandas as pd
import scipy

if __name__ == '__main__':
    # dev mode
    from mbapy.base import CDLL, get_dll_path_for_sys, put_err
    from mbapy.file import update_excel
else:
    # release mode
    from ..base import CDLL, get_dll_path_for_sys, put_err
    from ..file import update_excel

def get_value(df:pd.DataFrame, column:str, mask:np.array)->list:
    return df.loc[mask, column].tolist()


# TODO : not use itertools.product
def pro_bar_data(factors:List[str], tags:List[str], df:pd.DataFrame, **kwargs):
    """
    cacu mean and SE for each combinations of facotors\n
    data should be like this:\n
    | factor1 | factor2 | y1 | y2 |...\n
    |  f1_1   |   f2_1  |2.1 |-2  |...\n
    after process\n
    | factor1 | factor2 | y1(mean) | y1_SE(SE) | y1_N(sum_data) |...\n
    |  f1_1   |   f2_1  |2.1       |   -2      |   32           |...\n
    kwargs:
        min_sample_N:int : min N threshold(>=)
    """
    # kwargs
    min_sample_N = 1 if 'min_sample_N' not in kwargs else kwargs['min_sample_N']
    assert min_sample_N > 0, 'min_sample_N <= 0'
    # pro
    if len(tags) == 0:
        tags = list(df.columns)[len(factors):]
    factor_contents:list[list[str]] = [ df[f].unique().tolist() for f in factors ]
    ndf = [factors.copy()]
    for tag in tags:
        ndf[0] += [tag, tag+'_SE', tag+'_N']
    for factorCombi in itertools.product(*factor_contents):
        factorMask = np.array(df[factors[0]] == factorCombi[0])
        for i in range(1, len(factors)):
            factorMask &= np.array(df[factors[i]] == factorCombi[i])
        if factorMask.sum() >= min_sample_N:
            line = []
            for idx, tag in enumerate(tags):
                values = np.array(df.loc[factorMask, [tag]])
                line.append(values.mean())
                if values.shape[0] > 1:
                    line.append(values.std(ddof = 1)/np.sqrt(values.shape[0]))
                else:
                    line.append(np.NaN)
                line.append(values.shape[0])
            ndf.append(list(factorCombi) + line)
    return pd.DataFrame(ndf[1:], columns=ndf[0])

def pro_bar_data_R(factors:List[str], tags:List[str], df:pd.DataFrame, suffixs:List[str], **kwargs):
    """
    wrapper\n
    @pro_bar_data_R(['solution', 'type'], ['root', 'leaf'], ndf)\n
    def plot_func(values, **kwargs):
        return produced vars in list format whose length equal to len(suffix)
    """
    def ret_wrapper(core_func):
        def core_wrapper(**kwargs):
            nonlocal tags
            if len(tags) == 0:
                tags = list(df.columns)[len(factors):]
            factor_contents:List[List[str]] = [ df[f].unique().tolist() for f in factors ]
            ndf = [factors.copy()]
            for tag in tags:
                for suffix in suffixs:
                    ndf[0] += [tag+suffix]
            for factorCombi in itertools.product(*factor_contents):
                factorMask = np.array(df[factors[0]] == factorCombi[0])
                for i in range(1, len(factors)):
                    factorMask &= np.array(df[factors[i]] == factorCombi[i])
                if(factorMask.sum() > 0):
                    line = []
                    for idx, tag in enumerate(tags):
                        values = np.array(df.loc[factorMask, [tag]])
                        ret_line = core_func(values)
                        assert len(ret_line) == len(suffixs), 'length of return value of core_func != len(suffixs)'
                        line += ret_line
                    ndf.append(list(factorCombi) + line)
            return pd.DataFrame(ndf[1:], columns=ndf[0])
        return core_wrapper
    return ret_wrapper

def get_df_data(factors:Dict[str, List[str]], tags:List[str], df:pd.DataFrame,
                include_factors:bool = True):
    """
    Return a subset of the input DataFrame, filtered by the given factors and tags.

    Args:
        factors (dict[str, list[str]]): A dictionary containing the factors to filter by.
            The keys are column names in the DataFrame and the values are lists of values
            to filter by in that column.
        tags (list[str]): A list of column names to include in the output DataFrame.
        df (pd.DataFrame): The input DataFrame to filter.
        include_factors (bool, optional): Whether to include the factors in the output DataFrame.
            Defaults to True.

    Returns:
        pd.DataFrame: A subset of the input DataFrame, filtered by the given factors and tags.
        
    Examples:
        >>> sub_df = ndf.loc[(ndf['size'] == size1) & (ndf['light'] == light1), ['c', 'w', 'SE']]
        >>> sub_df = get_df_data([{'size':[size1], 'light':[light1]}, ['c', 'w', 'SE'])
    """
    def update_mask(mask, other:np.ndarray, method:str = '&'):
        return other if mask is None else (mask&other if method == '&' else mask|other)
    if len(tags) == 0:
        tags = list(set(df.columns.to_list())-set(factors.keys()))
    if include_factors:
        tags = list(factors.keys()) + tags
    mask = None
    for factor_name in factors:
        sub_mask = None
        if len(factors[factor_name]) == 0:
            # factors[factor_name] asigned with [], get all sub factors
            factors[factor_name] = df[factor_name].unique().tolist()
        for sub_factor in factors[factor_name]:
            sub_mask = update_mask(sub_mask, np.array(df[factor_name] == sub_factor), '|')
        mask = update_mask(mask, sub_mask, '&')
    return df.loc[mask, tags]

def sort_df_factors(factors:List[str], tags:List[str], df:pd.DataFrame):
    """UnTested
    sort each combinations of facotors\n
    data should be like this:\n
    | factor1 | factor2 | y1 | y2 |...\n
    |  f1_1   |   f2_1  |2.1 |-2  |...\n
    |  f1_1   |   f2_2  |2.1 |-2  |...\n
    ...\n
    after sort if given facotors=['factor2', 'factor1']\n
    | factor2 | factor1 | y1 | y2 |...\n
    |  f2_1   |   f1_1  |2.1 |-2  |...\n
    |  f2_1   |   f1_2  |2.1 |-2  |...\n
    ...\n
    """
    if len(tags) == 0:
        tags = list(df.columns)[len(factors):]
    factor_contents:List[List[str]] = [ df[f].unique().tolist() for f in factors ]
    ndf = [factors.copy()]
    ndf[0] += tags
    for factorCombi in itertools.product(*factor_contents):
        factorMask = df[factors[0]] == factorCombi[0]
        for i in range(1, len(factors)):
            factorMask &= df[factors[i]] == factorCombi[i]
        ndf.append(list(factorCombi) + np.array(df.loc[factorMask, tags].values))
    return pd.DataFrame(ndf[1:], columns=ndf[0])

def remove_simi(tag:str = None, df:pd.DataFrame = None, arr = None, tensor = None, sh:float = 1., 
                backend:str = 'numpy-array', device = 'cuda'):
    """
    给定一组数, 去除一些(最少数目)数, 使任意两数差的绝对值大于或等于阈值. numpy-mat实现模仿自动态规划Needleman-Wushsch序列对比算法.\n
    Given a set of numbers, remove some (minimum number) of numbers so that the absolute value of the difference between any two numbers is greater than or equal to the threshold\n
     
    Parameters: 
    --------
        -  tag : a string representing the column name in the dataframe to sort and remove similar elements from. 
        -  df : a pandas dataframe from which similar elements are to be removed. 
        -  arr : an array from which similar elements are to be removed. 
        -  sh : a float value that defines the threshold for similarity. If the difference between two elements is less than this value, one of them is considered for removal. 
        -  backend : a string that specifies the method to use for the operation. The options are 'numpy-mat', 'torch-array', and 'cpp-array', if not specified, the default is python list. 
        -  tensor : a tensor that can be used instead of  arr  when the backend is 'torch-array'. 
        -  device : a string that specifies the device to use for computation when the backend is 'torch-array'. The default is 'cuda'. 
    
    backend:
    --------
        - For 'numpy-mat', a n-n mat will be alloc. it creates a matrix where each element is the difference between two elements in  arr . It then iterates over this matrix to find elements that are similar according to the threshold  sh . 
        - For 'torch-array', only operate on a n shape arrray. it uses a PyTorch script to iterate over a tensor version of  arr  and find similar elements. 
        - For 'cpp-array', only operate on a n shape arrray. it uses a C++ function to find similar elements. 
        - Otherwise, use python list as array.
            
    Returns:
    --------
        the updated dataframe or array and the list of indices of removed elements.
        
    Raise:
    --------
        If meets numpy array memory error, return None(by put_err func).
        
    Examples
    --------
    >>> df = pd.DataFrame({'d':[1, 2, 3, 3, 5, 6, 8, 13]})\n
    >>> print(remove_simi('d', df, sh = 2.1))\n
    >>>     d
    >>> 0   1
    >>> 4   5
    >>> 6   8
    >>> 7  13
    """
    if tag is not None and df is not None:
        ndf = df.sort_values(by = tag, ascending=True)
        arr = ndf[tag]
    to_remove_idx = []
    if backend  == 'numpy-mat':
        arr = np.array(arr).reshape([1, len(arr)])
        try:
            mat = arr.repeat(arr.shape[1], axis = 0) - arr.transpose(1, 0).repeat(arr.shape[1], axis = 1)
        except np.core._exceptions._ArrayMemoryError:
            return put_err('OOM, retrun None', None)
        i, j, k = 0, 1, mat.shape[0] # start from the second col(number) to let the first number be lefted
        while i < k and j < k:
            if i == j:
                j += 1
            elif mat[i][j] < sh:
                to_remove_idx.append(j)
                j += 1 # move to next col(number)
            elif mat[i][j] >= sh:
                j += 1 # move to next col(number)
                i = j - 1 # set minuend to be previous col(number)
    elif backend == 'torch-array':
        try:
            import torch
        except:
            raise ImportError('no torch available')
        arr = tensor if tensor is not None else torch.tensor(arr, device = device).view(-1)
        @torch.jit.script
        def step_scan(x:torch.Tensor, to_remove:List[int], sh:float):
            i = 0 # start from the second col(number) to let the first number be lefted
            while i < x.shape[0]-1:
                if x[i+1] - x[i] < sh:
                    x[i+1] = x[i]
                    to_remove.append(i+1)
                i += 1
            return to_remove
        to_remove_idx = step_scan(arr, to_remove_idx, sh)
        arr = arr.to(device = 'cpu').numpy()
    elif backend == 'cpp-array':
        arr = list(arr)
        dll = CDLL(get_dll_path_for_sys('stats'))
        c_size = dll.ULL(len(arr))
        c_arr = dll.convert_c_lst(arr, dll.FLOAT)
        c_remove_simi = dll.get_func('remove_simi',
                                    [dll.PTR(dll.FLOAT), dll.PTR(dll.ULL), dll.FLOAT],
                                    dll.PTR(dll.ULL * 1))
        c_to_remove_idx = c_remove_simi(c_arr, dll.ptr(c_size), sh)
        to_remove_idx = dll.convert_py_lst(c_to_remove_idx, c_size.value)
        dll.get_func('freePtr', [dll.VOID])(ctypes.cast(c_to_remove_idx, dll.VOID))
    else:
        arr = list(arr)
        i = 0 # start from the second col(number) to let the first number be lefted
        while i < len(arr)-1:
            if arr[i+1] - arr[i] < sh:
                arr[i+1] = arr[i]
                to_remove_idx.append(i+1)
            i += 1
    
    if tag is not None and df is not None:
        ndf.drop(labels = to_remove_idx, inplace=True)
        return ndf, to_remove_idx
    elif arr is not None:
        arr = np.delete(arr, to_remove_idx)
        return arr, to_remove_idx

def interp(long_one:pd.Series, short_one:pd.Series):
    """
    给定两个pd.Series,一长一短,用线性插值给短的插值,使其长度与长的pd.Series一样\n
    Given two pd.Series, one long and one short, use linear interpolation to give the short one the same length as the long pd.Series\n
    """
    assert len(long_one) > len(short_one), 'len(long_one) <= len(short_one)'
    short_one_idx = np.array(np.arange(short_one.shape[0])*(long_one.shape[0]/short_one.shape[0]),
                             dtype=np.int32)
    if short_one_idx[-1] < long_one.shape[0]-1:
        short_one_idx[-1] = long_one.shape[0]-1
    return np.interp(np.arange(long_one.shape[0]), short_one_idx, short_one)

def merge_col2row(df:pd.DataFrame, cols:List[str],
                  new_cols_name:str, value_name:str):
    """
    Given a pandas.dataFrame, it has some colums, this func will replicate these colums to row\n
    Parameters
    ----------
    df: a pd.dataFrame
    cols: colums which need be merged to rows
    new_cols_name: new column contain cols name
    value_name: new column contain values of cols\n
    Return
    --------
    new_df: a new dataFrame
    """
    # 将需要转换的列转换为行，并将结果存储在一个新的数据框中
    new_df = pd.melt(df, id_vars=df.columns.difference(cols), value_vars=cols,
                     var_name=new_cols_name, value_name=value_name)
    # 重新设置索引
    new_df = new_df.reset_index(drop=True)
    return new_df

def make_three_line_table(factors:List[str], tags:List[str], df:pd.DataFrame,
                          float_fmt: str = '.3f', t_samples: int = 30):
    ndf = pro_bar_data(factors, tags, df)
    for tag in tags:
        coff = ndf[tag+'_SE'].copy()
        t_value = ndf[tag+'_N'].apply(lambda x:scipy.stats.t.ppf(0.975, df=x-1))
        coff[ndf[tag+'_N']<=t_samples] = t_value[ndf[tag+'_N']<=t_samples]
        ndf[tag] = ndf[tag].apply(lambda x: f'{x:{float_fmt}}') + ' (±' + (coff*ndf[tag+'_SE']).apply(lambda x: f'{x:{float_fmt}}') + ')'
        ndf.drop(tag+'_SE', axis=1, inplace = True)
    return ndf


if __name__ == '__main__':
    # dev code
    import ctypes
    dll = CDLL(get_dll_path_for_sys('stats'))
    c_size = dll.INT(1000000)
    arr = np.random.randn(c_size.value)
    arr.sort()
    
    from mbapy.base import TimeCosts
    @TimeCosts(10, True)
    def func(times, arr, backend, device):
        remove_simi(arr = arr, backend = backend, device = device)
        
    backends = ['', 'cpp-array']
    for backend in backends:
        func(arr = arr, backend = backend, device = 'cpu')
