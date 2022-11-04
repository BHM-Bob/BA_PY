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
pos, ax = plot.plot_bar(['solution', 'type'], ['root', 'leaf'], ndf,
                        xrotations=[35, 0])
# errorbar
ax.errorbar(pos, ndf['whole'], yerr=1.96 * ndf['whole_SE'],
            capsize=5, capthick = 2, elinewidth=2, fmt=' k')
# print
ax.legend(loc='upper left', title = "organ",
        title_fontsize = 20, ncol = 3, columnspacing = 0.6, handletextpad = 0.1)
title = f'A : Mean value of whole length to each solution'
# ax.set_title(title, fontsize=18, fontweight = 'bold')
ax.yaxis.set_label_text('Length (cm)')
plt.tight_layout()
# plt.gcf().savefig(f"ym {title.replace(':', '-'):s}.png", dpi=600)
plt.show()