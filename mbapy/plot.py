from itertools import chain, combinations
from typing import Callable, Dict, List, Tuple, Union, Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

if __name__ == '__main__':
    # dev mode
    import mbapy.stats.test as mst
    from mbapy.stats.df import (get_df_data, pro_bar_data, pro_bar_data_R,
                                sort_df_factors)
    from mbapy.stats.test import p_value_to_stars
    # Assembly of functions
    from mbapy.plot_utils.bar_utils import AxisLable, pro_bar_data, plot_bar, plot_positional_hue
    from mbapy.plot_utils.line_utils import *
    from mbapy.plot_utils.scatter_utils import plot_reg, plot_scatter, add_scatter_legend
else:
    # release mode
    from .stats import test as mst
    from .stats.df import (get_df_data, pro_bar_data, pro_bar_data_R,
                           sort_df_factors)
    from .stats.test import p_value_to_stars
    # Assembly of functions
    from .plot_utils.bar_utils import AxisLable, pro_bar_data, plot_bar, plot_positional_hue
    from .plot_utils.line_utils import *
    from .plot_utils.scatter_utils import plot_reg, plot_scatter, add_scatter_legend

# plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文
plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
PLT_MARKERS = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')

def rgb2hex(r, g, b):
  return '#'+('{:02X}' * 3).format(r, g, b)
def hex2rgb(hex:str):
  return [int(hex[i:i+2], 16) for i in (1, 3, 5)]
def rgbs2hexs(rgbs:List[Tuple[float]]):
    """
    Takes a list of RGB tuples and converts them to a list of hexadecimal color codes. 
    Each RGB tuple must contain three floats between 0 and 1 representing the red, green, and blue 
    components of the color. Returns a list of hexadecimal color codes as strings.
    """
    return list(map(lambda x : rgb2hex(*[int(x[i]*255) for i in range(3)]),
                    rgbs))
    
def get_palette(n:int = 10, mode:Union[None, str] = None, return_n = True) -> List[str]:
    """get a seq of hex colors    
    Parameters
    ----------
    n: how many colors required
    mode: kind of colors
        - hls : [default] sns.color_palette('hls', n)
        - green : sum 5 grenns
        - pair : plt.get_cmap('tab20')
        - others : plt.get_cmap(mode)
    """
    assert n >= 1, 'n < 1'
    ret = None
    if mode == 'hls':
        ret = rgbs2hexs(sns.color_palette('hls', n))
    elif n <= 5 and mode == 'green':
        ret = ['#80ab1c', '#405535', '#99b69b', '#92e4ce', '#72cb87'][0:n]
    elif n <= 9:
        ret = rgbs2hexs(plt.get_cmap('Set1').colors)
    elif n <= 12:
        ret = rgbs2hexs(plt.get_cmap('Set3').colors)
    elif n <= 20 and mode == 'pair':
        ret = rgbs2hexs(plt.get_cmap('tab20').colors)
    else:
        ret = rgbs2hexs(plt.get_cmap(mode).colors)
    if return_n and ret is not None:
        ret = ret[:n]
    return ret
    

def calcu_swarm_pos(x: float, y: np.ndarray, width: float, d: Optional[float] = None):
    """
   This function calculates the x-coordinates for the data points in a swarm plot.
   The x-coordinates are calculated based on the given x-coordinate, y-coordinates,
   and width of the swarm. If d is not None, it will be used as the distance between
   the data points. Otherwise, it will be calculated based on the number of data points.
   
    Parameters:
        - x: the x-coordinate of the center of the swarm.
        - y: the y-coordinates of the data points.
        - width: the width of the swarm.
        - d: the distance between the data points. If None, it will be calculated based on the number of data points.
        
    Returns:
        - A numpy array of x-coordinates for the data points.
    """
    def _calcu_arithmetic(x, n, w, d):
        if isinstance(d, float) or isinstance(d, int):
            a0 = x - (n-1)*d/2
            return np.linspace(a0, a0+d*(n-1), n)
        if n == 1:
            return x
        else:
            a0, d = x-w/2, w/(n-1)
            return np.linspace(a0, a0+d*(n-1), n)
    ret = np.zeros(len(y))
    for y_u in np.unique(y):
        y_idx = np.where(y == y_u)[0]
        ret[y_idx] = _calcu_arithmetic(x, len(y_idx), width, d)
    return ret

def qqplot(tags:List[str], df:pd.DataFrame, figsize = (12, 6), nrows = 1, ncols = 1, **kwargs):
    """
    Generate a QQ-plot for each column in the given DataFrame.

    Parameters:
        tags (List[str]): A list of column names to generate QQ-plots for.
        df (pd.DataFrame): The DataFrame containing the data.
        figsize (tuple, optional): The size of the figure. Defaults to (12, 6).
        nrows (int, optional): The number of rows in the figure grid. Defaults to 1.
        ncols (int, optional): The number of columns in the figure grid. Defaults to 1.
        **kwargs: Additional keyword arguments including xlim, ylim, title, tick_size, and label_size.

    Returns:
        None
    """
    axs = []
    fig = plt.figure(figsize = (12, 6))
    for fig_idx in range(1, ncols*nrows+1):
        axs.append(fig.add_subplot(nrows, ncols, fig_idx))
        if 'xlim' in kwargs:
            axs[-1].set_xlim(kwargs['xlim'])
        if 'ylim' in kwargs:
            axs[-1].set_ylim(kwargs['ylim'])            
        sm.qqplot(np.array(df[tags[fig_idx-1]]), fit=True, line="45", ax=axs[-1])
        axs[-1].set_title(tags[fig_idx-1]+' - QQPlot', fontdict={'fontsize':15})
        if 'title' in kwargs:
            axs[-1].set_ylim(kwargs['title'][fig_idx-1])
        if 'tick_size' in kwargs:
            axs[-1].tick_params(labelsize = kwargs['tick_size'])
        if 'label_size' in kwargs:
            axs[-1].set_xlabel('Theoretical Quantiles', fontsize = kwargs['label_size'])
            axs[-1].set_ylabel('Sample Quantiles', fontsize = kwargs['label_size'])
    return axs
            
def save_show(path:str, dpi = 300, bbox_inches = 'tight', show: bool = True, **kwargs):
    """
    Saves the current matplotlib figure to a file at the specified path and displays the figure.

    Parameters:
        - path (str): The path where the figure will be saved.
        - dpi (int, optional): The resolution of the saved figure in dots per inch. Default is 300.
        - bbox_inches (str or Bbox, optional): The portion of the figure to save. Default is 'tight'.
        - show (bool, optional): Whether to show the figure after saving. Default is True.
    """
    plt.tight_layout()
    plt.gcf().savefig(path, dpi=dpi, bbox_inches = bbox_inches, **kwargs)
    if show:
        plt.show()
    
def plot_stats_star(x1: float, x2: float, h: float, endpoint: float, p_value: float,
                    ax = plt, p2star: Callable[[float], str] = p_value_to_stars):
    """
    Params
        - x1: The x-coordinate of the left endpoint.
        - x2: The x-coordinate of the right endpoint.
        - h: The y-coordinate of the horizontal line.
        - endpoint: The height of the endpoint.
        - p_value: The p-value of the difference.
        - ax: The `Axes` instance to plot on. Default is `plt`.
        - p2star: A function that converts a p-value to a string of stars. Default is `p_value_to_stars`.

    Returns:
        None
    """
    ax.plot([x1, x2], [h, h], color='black')
    ax.plot([x1, x1], [h, h-endpoint], color='black')
    ax.plot([x2, x2], [h, h-endpoint], color='black')
    ax.text((x1+x2)/2, h, p2star(p_value), ha='center')    
    
def plot_turkey(means, std_errs, tukey_results, min_star = 1):
    """
    Plot a bar chart showing the means of different groups along with the standard errors.

    Parameters:
        - means: A list of mean values for each group.
        - std_errs: A list of standard errors for each group.
        - tukey_results: The Tukey's test results object.

    Returns:
        The current `Axes` instance.

    This function plots a bar chart using the given mean values and standard errors. It also marks the groups with significant differences based on the Tukey's test results.
    For each combination of groups, the function checks if the corresponding Tukey's test result indicates a significant difference. If so, it plots a horizontal line at the maximum height, vertical lines at the endpoints, and places a text label with stars indicating the p-value of the difference.
    """
    x = np.arange(len(means))
    plt.bar(x, means, yerr=std_errs, capsize=5)

    combins = np.array(list(combinations(range(len(means)), 2)))
    height = max(means) + max(std_errs)
    min_height = 0.05 * height
    endpoint_height = [height - 0.05 * min_height, height + 0.05 * min_height]
    for i, combination in enumerate(combins):
        if tukey_results.reject[i] \
                and len(p_value_to_stars(tukey_results.pvalues[i])) >= min_star:
            plt.plot(combination, [height, height], color='black')
            plt.plot([combination[0], combination[0]], endpoint_height, color='black')
            plt.plot([combination[1], combination[1]], endpoint_height, color='black')
            plt.text(np.mean(combination), height,
                     p_value_to_stars(tukey_results.pvalues[i]), ha='center')
            height += min_height

    plt.xticks(x, tukey_results.groupsunique)
    return plt.gca()


__all__ = [
    'AxisLable',
    'pro_bar_data',
    'plot_bar',
    'plot_positional_hue',
    
    'plot_reg',
    'plot_scatter',
    'add_scatter_legend',
    
    'rgb2hex',
    'hex2rgb',
    'rgbs2hexs',
    'get_palette',
    'calcu_swarm_pos',
    'qqplot',
    'save_show',
    'plot_stats_star',
    'plot_turkey'
]


if __name__ == '__main__':
    # dev code
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
    