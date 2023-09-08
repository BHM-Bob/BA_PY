'''
Author: BHM-Bob 2262029386@qq.com
Date: 2022-12-09 17:24:18
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-09-03 10:17:19
Description: 
'''
from . import cluster, df, geography, reg, test

# def main():
#     pass
"""
var naming:

constant : CONSTANT_VAR_NAME
variable : var_name
func : func_name
global var: globalVarName
class : ClassName
"""

import pandas as pd
from sklearn.decomposition import PCA


def pca(df:pd.DataFrame, out_dim:int):
    pca = PCA(n_components=out_dim)
    pca.fit(df)
    return pca.transform(df)