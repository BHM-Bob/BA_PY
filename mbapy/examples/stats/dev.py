'''
Author: BHM-Bob 2262029386@qq.com
Date: 2022-12-05 16:18:43
LastEditors: BHM-Bob
LastEditTime: 2023-04-17 16:29:22
Description: 
'''
import sys
sys.path.append(r'../../../')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot
import stats

# create bars
df = pd.read_excel(r"data/plot.xlsx", sheet_name="MWM")
df['Animal Type'] = df['Animal Type'].astype('str')
result = stats.test.multicomp_turkeyHSD({'Animal Type':['Control', '0.0003', '0.001', '0.003']}, 'First Entry Len', df)
print(result)
# result.plot_simultaneous()
# plt.show()

# result = stats.test.multicomp_dunnett('Animal Type', ['Control', '0.0003', '0.001', '0.003'], 'First Entry Len', df)
# print(result)

result = stats.test.multicomp_bonferroni({'Animal Type':['Control', '0.0003', '0.001', '0.003']}, 'First Entry Len', df)
print(result)