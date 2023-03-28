'''
Author: BHM-Bob 2262029386@qq.com
Date: 2023-03-25 23:13:03
LastEditors: BHM-Bob
LastEditTime: 2023-03-25 23:24:14
Description: 
'''
import sys

sys.path.append(r'../../../')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot

# create bars
ndf_peak = pd.read_excel(r"data/plot.xlsx", sheet_name="peak")
ndf_peak2 = plot.pro_bar_data(['time', 'event'], ['wave height'], ndf_peak)
pos, ax = plot.plot_bar(['event', 'time'], ['wave height'], ndf_peak2)
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.bar(ndf['event'], ndf['val'], color = 'grey')
plt.errorbar(pos, ndf_peak2['wave height'], yerr=1.96 * ndf_peak2['wave height_SE'],
             capsize=5, capthick = 2, elinewidth=2, fmt=' k')
# sns.violinplot(x="event", y='val', label = "event", data = ndf2, ax = ax)