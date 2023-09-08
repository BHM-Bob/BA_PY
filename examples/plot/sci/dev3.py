import sys

sys.path.append(r'../../../')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
import pandas as pd

import plot

ndf_weight = pd.read_excel(r"data/plot.xlsx", sheet_name="weight")
ndf_weight_2 = plot.pro_bar_data(['day', 'type'], ['weight'], ndf_weight, min_sample_N = 8)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig, ax = plt.subplots(figsize=(8, 6))
for exp_type, col, marker in zip(['CK', 'Exp'], colors[:2], ['^', 'v']):
    sub_df = plot.get_df_data({'type':[exp_type]}, ['day', 'weight', 'weight_SE'], ndf_weight_2)
    ax.plot(sub_df['day'], sub_df['weight'], label = exp_type,
                    color = col, marker = marker, markersize = 12,
                    linewidth=3)
    ax.errorbar(sub_df['day'], sub_df['weight'], yerr=1.96 * sub_df['weight_SE'],
                linewidth = 0, color = col, capsize=5, capthick = 2, elinewidth=2)
    
ax.xaxis.set_major_locator(MultipleLocator(2))
ax.xaxis.set_major_formatter(FormatStrFormatter('%2d'))
plt.xticks(size = 20)
plt.yticks(size = 20)
ax.set_xlabel(r'Day', fontsize=25)
ax.set_ylabel(f'Weight (g)', fontsize=25)
plt.legend(fontsize = 20)
plt.tight_layout()
plt.show()