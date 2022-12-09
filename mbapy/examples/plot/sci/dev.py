'''
Author: BHM-Bob 2262029386@qq.com
Date: 2022-11-04 12:33:19
LastEditors: BHM-Bob
LastEditTime: 2022-12-09 17:18:42
Description: 
'''
import sys

sys.path.append(r'../../../../mbapy/')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plot

plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['font.size'] = 20
plt.rcParams['figure.figsize'] = (7, 7)

df = pd.read_excel(r"data/plot.xlsx", sheet_name="xm")
solutions = df['solution'].unique().tolist()
ndf = plot.pro_bar_data(['type', 'solution'], [], df)
# print(plot.get_df_data({'type':['single', 'full'], 'solution':['KCl']}, ['root', 'leaf'], df))

@plot.plot_positional_hue(['solution', 'type'], ['root'], ndf)
def plot_a_bar(ax, x, y, label, label_idx, margs, **kwargs):
    ax.bar(x, y, width = margs.width, bottom = margs.bottom, label=label,
           edgecolor='white', color=margs.colors[label_idx], **kwargs)
    for i, n in enumerate(ndf['root_N']):
        ax.text(s = n, x = x[i], y = y[i], fontweight = 'bold')
pos, ax = plot_a_bar(xrotations=[5, 0])
# errorbar
ax.errorbar(pos, ndf['root'], yerr=1.96 * ndf['root_SE'],
            capsize=5, capthick = 2, elinewidth=2, fmt=' k')
# print
ax.legend(loc='best', title = "organ",
        title_fontsize = 20, ncol = 3, columnspacing = 0.6, handletextpad = 0.1)
title = f'A : Mean value of whole length to each solution'
# ax.set_title(title, fontsize=18, fontweight = 'bold')
ax.yaxis.set_label_text('Length (cm)')
plt.tight_layout()
# plt.gcf().savefig(f"ym {title.replace(':', '-'):s}.png", dpi=600)
plt.show()