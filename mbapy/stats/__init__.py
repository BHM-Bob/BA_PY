'''
Author: BHM-Bob 2262029386@qq.com
Date: 2022-12-09 17:24:18
LastEditors: BHM-Bob
LastEditTime: 2023-04-19 19:56:16
Description: 
'''
from . import geography, reg, test, df

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