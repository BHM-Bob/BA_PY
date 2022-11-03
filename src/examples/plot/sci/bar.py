import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mbapy import plot

plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['font.size'] = 20
plt.rcParams['figure.figsize'] = (7, 7)
    
df = pd.read_excel(r"E:\HC\Desktop\学科笔记\植物生理学\植物生理学实验\2-缺素\data\data.xlsx", sheet_name="ym")
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
# ax.set_title(title, fontsize=18, fontweight = 'bold')
ax.yaxis.set_label_text('Length (cm)')
plt.tight_layout()
# plt.gcf().savefig(f"ym {title.replace(':', '-'):s}.png", dpi=600)
plt.show()