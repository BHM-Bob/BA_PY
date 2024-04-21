from functools import wraps
from itertools import chain, combinations
from typing import Callable, Dict, List, Tuple, Union, Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from mpl_toolkits import axisartist
from mpl_toolkits.axes_grid1 import host_subplot

import mbapy.stats.test as mst
from mbapy.base import get_wanted_args
from mbapy.stats.df import (get_df_data, pro_bar_data, pro_bar_data_R,
                            sort_df_factors)
from mbapy.stats.test import p_value_to_stars

# plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文
plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

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
        - None : plt.get_cmap('Set1') for n<=9 or plt.get_cmap('Set3') for n<= 12
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
    if return_n and ret is not None:
        ret = ret[:n]
    return ret
    
class AxisLable():
    def __init__(self, name:str, hold_space:int = 1) -> None:
        self.name = name
        self.hold_space = hold_space
        self.father = None
        self.child = set()
        self._hash = id(self) # 固定hash值
    def __eq__(self, other: 'AxisLable') -> bool:
        return id(self) == id(other)
    def __hash__(self) -> int:
        return self._hash
    def add_space(self, space:int = 1):
        self.hold_space += space
    def add_father(self, father: 'AxisLable'):
        self.father = father
        self.father.child.add(self)

def pro_hue_pos(factors:List[str], df:pd.DataFrame, width:float,
                hue_space:float, bar_space:float):
    """
    Generate the position and labels for a grouped bar plot with multiple factors.

    Args:
        - factors (List[str]): A list of strings representing the factors to group the bars by.
        - df (pd.DataFrame): A pandas DataFrame containing the data for the bar plot.
        - width (float): The width of each individual bar.
        - hue_space (float): The space between each group of bars.
        - bar_space (float): The space between each bar in a group.

    Returns:
        Tuple[List[List[AxisLable]], List[List[float]]]: A tuple containing two lists. The first list contains the labels for each factor and each bar. The second list contains the x-positions for each bar.

    Notes:
        - `df` must be processed by `pro_bar_data` or `sort_df_factors`.
        - The LAST factor will be the TOP level x-axis.
    """
    xlabels, fj_old, pos = [ [] for _ in range(len(factors))], None, []
    xlabels_mat = df[factors].T.values.tolist()
    # build relationships between each level sub-factors. build xlabels and allocte space for each level.
    for i, fi in enumerate(factors[::-1]): # 从最高级开始
        for j, fj in enumerate(df[fi]):
            # 不同sub-factor 或 相同sub-factor但上级sub-factor不同
            if (str(fi)+str(fj)) != fj_old or (i > 0 and xlabels_mat[i-1][j] != xlabels_mat[i-1][j-1]):
                xlabels_mat[i][j] = AxisLable(fj)
                xlabels[i].append(AxisLable(fj))
                fj_old = str(fi)+str(fj) # 以防两个级别的factors中出现相同的sub-factors
                if i > 0:
                    xlabels_mat[i][j].add_father(xlabels_mat[i-1][j])
            else:
                xlabels_mat[i][j] = xlabels_mat[i][j-1]
                xlabels[i][-1].add_space()
    xlabels = xlabels[::-1] # 倒序，使最高级在最后
    xlabels.append([AxisLable(factors[-1], df.shape[0])])#master level has an extra total axis as x_title
    for axis_idx in range(len(xlabels)):
        pos.append([])
        if axis_idx == 0:
            st_pos = hue_space
            for h_fc_idx in range(len(xlabels[axis_idx+1])):
                sum_this_hue_bar = xlabels[axis_idx+1][h_fc_idx].hold_space
                pos[axis_idx] += [st_pos+width*(i+0.5)+bar_space*i for i in range(sum_this_hue_bar)]
                st_pos += (sum_this_hue_bar*width+(sum_this_hue_bar-1)*bar_space+hue_space)
        else:
            st_pos = 0
            for fc_idx in range(len(xlabels[axis_idx])):
                this_hue_per = xlabels[axis_idx][fc_idx].hold_space / df.shape[0]
                pos[axis_idx].append(st_pos+this_hue_per/2)
                st_pos += this_hue_per
    return xlabels, pos

def plot_bar(factors:List[str], tags:List[str], df:pd.DataFrame, **kwargs):
    """
    Stack bar plot with hue style

    Args:
        - factors (List[str]): A list of factors. [low_lever_factor, medium_lever_factor, ...] or just one.
        - tags (List[str]): A list of tags. [stack_low_y, stack_medium_y, ...] or just one.
        - df (pd.DataFrame): A pandas DataFrame. From `pro_bar_data` or `sort_df_factors`.
        - kwargs: Additional keyword arguments.
            - fig = None
            - jitter = Flase, IF True, pass the original df, the func will call `pro_bar_data` internally.
            - jitter_kwargs = {'size': 10, 'alpha': 0.5, 'color': None}
            - width = 0.4
            - bar_space = 0.05
            - hue_space = 0.2
            - xrotations = [0]*len(factors)
            - colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            - hatchs:['-', '+', 'x', '\\', '*', 'o', 'O', '.'],
            - font_size:None,
            - labels:None,
            - offset = [None] + [(i+1)*(plt.rcParams['font.size']+8) for i in range(len(factors)-1)], x-axis offset
            - xticks_pad: [5 for _ in range(len(factors)+1)], x-axis label pad from x-axis
            - edgecolor:['white'] * len(tags),
            - linewidth: 0,
            - err: (str|np.array), if str, will use df[err] as yerr. if np.array, will use it directly. will multiply 1.96 to yerr.
            - err_kwargs = {'capsize':5, 'capthick':2, 'elinewidth':2, 'fmt':' k', 'ecolor':'black'}

    Returns:
        - np.array: An array of positions.
        - ax1: An axis object.
        - df: If `jitter` is True, return the processed df.
        
    Notes:
        - the `FIRST factor` will be the `LOWEST level x-axis`.
        - y-axis tick labels' fontsize will be set same as first level's x-axis tick labels' fontsize.
    """
    # process args
    if len(tags) == 0:
        # TODO: 可能存在'Unbond: 0'等其他情况
        tags = list(df.columns)[len(factors):]
    args = get_wanted_args({'fig': None,
                            'jitter':False,
                            'jitter_kwargs': {'size': 10, 'alpha': 0.5, 'color': None},
                            'width':0.4, 'hue_space':0.2, 'bar_space':0.05,
                            'xrotations':[0]*len(factors),
                            'colors':plt.rcParams['axes.prop_cycle'].by_key()['color'],
                            'hatchs':['-', '+', 'x', '\\', '*', 'o', 'O', '.'],
                            'font_size':None,
                            'labels':None,
                            'offset':[None] + [(i+1)*(plt.rcParams['font.size']+8) for i in range(len(factors))],
                            'xticks_pad':[5 for _ in range(len(factors)+1)],
                            'edgecolor':['white'] * len(tags),
                            'linewidth': 0,
                            'log': False,
                            'err':None,
                            'err_kwargs':{'capsize':5, 'capthick':2, 'elinewidth':2, 'fmt':' k', 'ecolor':'black'}},
                            kwargs)
    args.xrotations.append(0)
    # make first level axis
    ax1 = host_subplot(111, axes_class=axisartist.Axes, figure=args.fig)
    # plot jitter using seaborn
    if args.jitter:
        jittor_color = args.jitter_kwargs['color']
        del args.jitter_kwargs['color']
        @pro_bar_data_R(factors[::-1], tags, df, ['', '_SE', '_N', '_V'])
        def pro_bar_jitter_data(v):
            if v.shape[0] > 1:
                se = v.std(ddof = 1)/np.sqrt(v.shape[0])
            else:
                se = np.NaN
            return [v.mean(), se, v.shape[0], v]
        df = pro_bar_jitter_data()
    # make xlabels and positions
    xlabels, pos = pro_hue_pos(factors, df, args.width, args.hue_space, args.bar_space)
    # plot bar
    bottom = kwargs['bottom'] if 'bottom' in kwargs else np.zeros(len(pos[0]))
    for yIdx, yName in enumerate(tags):
        # label
        label = args.labels[yIdx] if (args.labels and args.labels[yIdx] is not None) else yName
        # add jitter
        if args.jitter:
            for i, (n, y) in enumerate(zip(df[yName+'_N'], df[yName+'_V'])):
                x = [pos[0][i]] * n
                color = jittor_color[yIdx][i] if jittor_color is not None else 'black'
                sns.stripplot(x=x, y=y.reshape(-1), ax=ax1, jitter=True, native_scale=True,
                              color=color, zorder = 1, legend = False, **args.jitter_kwargs)
        # plot bar
        ax1.bar(pos[0], df[yName], width = args.width, bottom = bottom, label=label,
                edgecolor=args.edgecolor[yIdx], linewidth = args.linewidth,
                color=args.colors[yIdx], log = args.log, zorder = 0)
        bottom += df[yName]
    ax1.set_xlim(0, pos[0][-1]+args.hue_space+args.width/2)
    ax1.set_xticks(pos[0], [l.name for l in xlabels[0]])
    plt.setp(ax1.axis["bottom"].major_ticklabels, rotation=args.xrotations[0], pad = args.xticks_pad[0])
    if args.font_size is not None:
        plt.setp(ax1.axis["bottom"].major_ticklabels, fontsize=args.font_size[0])
        plt.setp(ax1.axis["left"].major_ticklabels, fontsize=args.font_size[0])
    
    axs = []
    for idx, sub_pos in enumerate(pos[1:]):
        axs.append(ax1.twiny())
        axs[-1].set_xticks(sub_pos, [l.name for l in xlabels[idx+1]])
        new_axisline = axs[-1].get_grid_helper().new_fixed_axis
        axs[-1].axis["bottom"] = new_axisline(loc="bottom", axes=axs[-1], offset=(0, -args.offset[idx+1]))
        plt.setp(axs[-1].axis["bottom"].major_ticklabels, rotation=args.xrotations[idx+1], pad = args.xticks_pad[idx+1])
        if args.font_size is not None:
            plt.setp(axs[-1].axis["bottom"].major_ticklabels, fontsize=args.font_size[idx+1])
        axs[-1].axis["top"].major_ticks.set_ticksize(0)
        # TODO : do not work
        axs[-1].axis["right"].major_ticks.set_ticksize(0)
        
    # err bar, put here because errorbar will change ax obj and occur errs
    if args.err is not None:
        if isinstance(args.err, str):
            args.err = df[args.err]
        if 'ecolor' in args.err_kwargs and isinstance(args.err_kwargs['ecolor'], list):
            err_cols = args.err_kwargs['ecolor']
            del args.err_kwargs['ecolor']
            for pos_i, y, err_i, col_i in zip(pos[0], bottom, args.err, err_cols):
                ax1.errorbar(pos_i, y, yerr=1.96 * err_i, zorder = 2, ecolor = col_i, **args.err_kwargs)
        else:
            ax1.errorbar(pos[0], bottom, yerr=1.96 * args.err, zorder = 2, **args.err_kwargs)
    
    if args.jitter:
        return np.array(pos[0]), ax1, df
    else:
        return np.array(pos[0]), ax1

def plot_positional_hue(factors:List[str], tags:List[str], df:pd.DataFrame, **kwargs):
    """
    wrapper\n
    support args: width, bar_space, xrotations, colors, offset, bottom\n
    xlabels is in margs
    @plot_positional_hue(['solution', 'type'], ['root', 'leaf'], ndf)\n
    def plot_func(ax, x, y, label, label_idx, margs, **kwargs):
        do something
    """
    def ret_wrapper(core_plot_func):
        def core_wrapper(**kwargs):
            ax1 = host_subplot(111, axes_class=axisartist.Axes)
            nonlocal tags
            if len(tags) == 0:
                tags = list(df.columns)[len(factors):]    
            margs = get_wanted_args({'width':0.4, 'bar_space':0.2, 'xrotations':[0]*len(factors),
                                    'colors':plt.rcParams['axes.prop_cycle'].by_key()['color'],
                                    'offset':[(i+1)*(plt.rcParams['font.size']+8) for i in range(len(factors))]},
                                   kwargs)
            margs.xrotations.append(0)
            xlabels, pos = pro_hue_pos(factors, df, margs.width, margs.bar_space)
            margs.add_arg('xlabels', xlabels)
            margs.add_arg('bottom', np.zeros(len(pos[0])), False)
            if 'bottom' in kwargs:
                del kwargs['bottom']
            for yIdx, yName in enumerate(tags):
                core_plot_func(ax1, pos[0], df[yName], yName, yIdx, margs, **kwargs)
            ax1.set_xlim(0, pos[0][-1]+margs.bar_space+margs.width/2)
            ax1.set_xticks(pos[0], [l.name for l in xlabels[0]])
            plt.setp(ax1.axis["bottom"].major_ticklabels, rotation=margs.xrotations[0])            
            axs = []
            for idx, sub_pos in enumerate(pos[1:]):
                axs.append(ax1.twiny())
                axs[-1].set_xticks(sub_pos, [l.name for l in xlabels[idx+1]])
                new_axisline = axs[-1].get_grid_helper().new_fixed_axis
                axs[-1].axis["bottom"] = new_axisline(loc="bottom", axes=axs[-1], offset=(0, -margs.offset[idx]))
                plt.setp(axs[-1].axis["bottom"].major_ticklabels, rotation=margs.xrotations[idx+1])
                axs[-1].axis["top"].major_ticks.set_ticksize(0)
                # TODO : do not work
                axs[-1].axis["right"].major_ticks.set_ticksize(0)            
            return np.array(pos[0]), ax1
        return core_wrapper
    return ret_wrapper

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
            
def save_show(path:str, dpi = 300, bbox_inches = 'tight'):
    """
    Saves the current matplotlib figure to a file at the specified path and displays the figure.

    Parameters:
        path (str): The path where the figure will be saved.
        dpi (int, optional): The resolution of the saved figure in dots per inch. Default is 300.
        bbox_inches (str or Bbox, optional): The portion of the figure to save. Default is 'tight'.

    Returns:
        None
    """
    plt.tight_layout()
    plt.gcf().savefig(path, dpi=dpi, bbox_inches = bbox_inches)
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
    