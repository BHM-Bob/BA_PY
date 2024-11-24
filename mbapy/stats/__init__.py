'''
Author: BHM-Bob 2262029386@qq.com
Date: 2022-12-09 17:24:18
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2024-11-24 21:47:54
Description: 
'''

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    from mbapy.base import get_default_for_None, put_err
    from mbapy.stats import cluster, df, geography, reg, test
else:
    from ..base import get_default_for_None, put_err
    from . import cluster, df, geography, reg, test

def pca(df:pd.DataFrame, out_dim:int, scale: bool = False, return_model: bool = False) -> np.ndarray:
    """
    performs PCA on a dataframe and returns the transformed data.  
    
    Returns:
        - np.ndarray: The transformed data.
        - sklearn.decomposition.PCA: The PCA model if return_model is True.
        
    Notes:
        - access model.explained_variance_ratio_ to get the explained variance ratio.
    """
    pca = PCA(n_components=out_dim)
    if scale:
        scaler = StandardScaler()
        df = scaler.fit_transform(df)
    pca.fit(df)
    if return_model:
        return pca.transform(df), pca
    return pca.transform(df)

def max_pool2d(x:np.ndarray, pool_size:Tuple[int, int], stride: Tuple[int, int] = None):
    """
    Performs max pooling on a NDarray on last two dimensions.

    Args:
        - x (np.ndarray): The input array to perform max pooling on.
        - pool_size (Tuple[int, int], [h, w]): The size of the pooling window.
        - stride (Tuple[int, int], [h, w], optional): The stride to use for the pooling window. Defaults to None.

    Returns:
        np.ndarray: The result of the max pooling operation.

    Raises:
        mbapy inner error info: If pool_size does not have 2 dimensions.
        mbapy inner error info: If stride is provided and does not have the same shape as pool_size.

    Notes:
        - The input array x should have at least 2 dimensions.
        - If stride is None, it will be set to the same shape as pool_size.
    """
    # check param
    if len(pool_size) != 2:
        return put_err(f'pool_size must have at least 2 dimensions as it is pool2d, \
            but got {len(x.shape)} dimensions. return None', None)
    if stride is not None and len(stride) != len(pool_size):
        return put_err(f'stride must have the same shape as pool_size or simply None, \
            but got {len(stride)} and {len(pool_size)}', None)
    stride = get_default_for_None(stride, pool_size)
    # split x and do max pool
    pool_windows = np.lib.stride_tricks.sliding_window_view(x, pool_size, (-2, -1))
    pooled_data = np.max(pool_windows, axis=(-2, -1))
    # stride
    pooled_shape = pooled_data.shape
    pooled_data = pooled_data.reshape(-1, pooled_shape[-2], pooled_shape[-1]) # only apply last two dim
    strided_data = pooled_data[:, ::stride[0], ::stride[1]] # only apply last two dim
    # reshape
    return strided_data.reshape(pooled_shape[:-2] + strided_data.shape[-2:]) # recover original dims but keep last two

if __name__ == '__main__':
    # dev code
    input_data = np.array([[1, 2, 3, 4],
                           [5, 6, 7, 8],
                           [9, 10, 11, 12]])

    pool_size = (3, 2)

    pooled_data = max_pool2d(input_data, pool_size)
    print(pooled_data)