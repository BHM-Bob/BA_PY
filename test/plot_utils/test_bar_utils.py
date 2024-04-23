'''
Date: 2024-04-23 10:57:21
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2024-04-23 10:58:39
Description: 
'''
import matplotlib.pyplot as plt
import pandas as pd

import mbapy.stats.test as mst
from mbapy.plot import get_palette, plot_turkey, pro_bar_data, plot_bar

df = pd.read_excel('./data/plot.xlsx', sheet_name='MWM')
df['Animal Type'] = df['Animal Type'].astype('str')
model = mst.multicomp_turkeyHSD({'Animal Type':[]}, 'Duration', df)
result = mst.turkey_to_table(model)
print(result)
# sub_df = get_df_data({'Animal Type':[]}, ['Duration'], df)
sub_df = pro_bar_data(['Animal Type'], ['Duration'], df)
# test err
cols = get_palette(n = 4, mode = 'hls')
plot_bar(['Animal Type'], ['Duration'], df, err = sub_df['Duration_SE'], jitter = True,
            edgecolor = [cols], linewidth = 5, colors = ['white'], jitter_kwargs = {'size': 10, 'alpha': 1, 'color': [cols]})
plt.show()
plot_turkey(sub_df['Duration'], sub_df['Duration_SE'], model)
plt.show()

df = pd.DataFrame({'Month': [5, 5, 6, 6, 7, 7, 8, 8, 9, 9],
                    'Ozone': [23.61538, 22.22445, 29.44444, 18.20790, 59.11538, 31.63584, 59.96154, 39.68121, 31.44828, 24.14182]})
model = mst.multicomp_turkeyHSD({'Month':[]}, 'Ozone', df)
result = mst.turkey_to_table(model)
print(result)
sub_df = pro_bar_data(['Month'], ['Ozone'], df)
plot_turkey(sub_df['Ozone'], sub_df['Ozone_SE'], model)
plt.show()