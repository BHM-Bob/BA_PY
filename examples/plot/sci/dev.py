'''
Author: BHM-Bob 2262029386@qq.com
Date: 2022-11-04 12:33:19
LastEditors: BHM-Bob
LastEditTime: 2023-03-25 23:18:14
Description: 
'''
import sys

sys.path.append(r'../../../')
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
@plot.pro_bar_data_R(['type', 'solution'], [], df, ['', '_SUM'])
def get_sum(values):
    return [values.mean(), values.sum()]
ndf2 = get_sum()
print(ndf2)

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

df = pd.read_excel(r"data/plot.xlsx", sheet_name="ym")
solutions = df['solution'].unique().tolist()
ndf = plot.pro_bar_data(['time', 'solution'], [], df)
pos, ax = plot.plot_bar(['solution', 'time'], ['root', 'stem', 'leaf'], ndf,
                        xrotations=[35, 0])
# errorbar
ax.errorbar(pos, ndf['whole'], yerr=1.96 * ndf['whole_SE'],
            capsize=5, capthick = 2, elinewidth=2, fmt=' k')
# SNK.test
snk = [
['Mg',    'a'],
['Fe',   'ab'],
['full', 'ab'],
['K',    'ab'],
['P',    'ab'],
['Ca',   'ab'],
['N',     'b'],
]
for solution, group in snk:
    ax.text(s = group, x = pos[ndf['time']=='after'][solutions.index(solution)],
             y = 2+np.array(ndf.loc[(ndf['solution'] == solution) & (ndf['time'] == 'after'), ['whole']])[0],
             fontweight = 'bold')
# print
ax.legend(loc='upper left', title = "organ",
        title_fontsize = 20, ncol = 3, columnspacing = 0.6, handletextpad = 0.1)
title = f'A : Mean value of whole length to each solution'
ax.yaxis.set_label_text('Length (cm)')
plt.tight_layout()
plt.show()