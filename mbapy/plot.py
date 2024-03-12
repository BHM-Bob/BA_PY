from functools import wraps
from itertools import combinations
from typing import Dict, List, Tuple, Union

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
    def add_space(self, space:int = 1):
        self.hold_space += space

def pro_hue_pos(factors:List[str], df:pd.DataFrame, width:float, bar_space:float):
    """
    Generate the position and labels for a grouped bar plot with multiple factors.

    Args:
        factors (List[str]): A list of strings representing the factors to group the bars by.
        df (pd.DataFrame): A pandas DataFrame containing the data for the bar plot.
        width (float): The width of each individual bar.
        bar_space (float): The space between each group of bars.

    Returns:
        Tuple[List[List[AxisLable]], List[List[float]]]: A tuple containing two lists. The first list contains the labels for each factor and each bar. The second list contains the x-positions for each bar.

    """
    xlabels, fc_old, pos = [ [] for _ in range(len(factors))], '', []
    for f_i, f in enumerate(factors):
        for fc_i, fc in enumerate(df[f]):
            if fc != fc_old:
                xlabels[f_i].append(AxisLable(fc))
                fc_old = fc
            else:
                xlabels[f_i][-1].add_space()
    xlabels.append([AxisLable(factors[-1], df.shape[0])])#master level has an extra total axis as x_title
    for axis_idx in range(len(xlabels)):
        pos.append([])
        if axis_idx == 0:
            st_pos = bar_space
            for h_fc_idx in range(len(xlabels[axis_idx+1])):
                sum_this_hue_bar = xlabels[axis_idx+1][h_fc_idx].hold_space
                pos[axis_idx] += [st_pos+width*(i+0.5) for i in range(sum_this_hue_bar)]
                st_pos += (sum_this_hue_bar*width+bar_space)
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
        factors (List[str]): A list of factors. [low_lever_factor, medium_lever_factor, ...] or just one.
        tags (List[str]): A list of tags. [stack_low_y, stack_medium_y, ...] or just one.
        df (pd.DataFrame): A pandas DataFrame. From pro_bar_data or sort_df_factors.
        **kwargs: Additional keyword arguments.
            width = 0.4\n
            bar_space = 0.2\n
            xrotations = [0]*len(factors)\n
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']\n
            hatchs:['-', '+', 'x', '\\', '*', 'o', 'O', '.'],\n
            labels:None,\n
            font_size:None, \n
            offset = [(i+1)*(plt.rcParams['font.size']+8) for i in range(len(factors))]\n
            err = None\n
            err_kwargs = {'capsize':5, 'capthick':2, 'elinewidth':2, 'fmt':' k'}

    Returns:
        np.array: An array of positions.
        ax1: An axis object.
    """
    ax1 = host_subplot(111, axes_class=axisartist.Axes)
    
    if len(tags) == 0:
        tags = list(df.columns)[len(factors):]
    args = get_wanted_args({'width':0.4, 'bar_space':0.2, 'xrotations':[0]*len(factors),
                            'colors':plt.rcParams['axes.prop_cycle'].by_key()['color'],
                            'hatchs':['-', '+', 'x', '\\', '*', 'o', 'O', '.'],
                            'font_size':None,
                            'labels':None,
                            'offset':[(i+1)*(plt.rcParams['font.size']+8) for i in range(len(factors))],
                            'edgecolor':'white',
                            'err':None,
                            'err_kwargs':{'capsize':5, 'capthick':2, 'elinewidth':2, 'fmt':' k'}},
                            kwargs)
    args.xrotations.append(0)
    xlabels, pos = pro_hue_pos(factors, df, args.width, args.bar_space)
    bottom = kwargs['bottom'] if 'bottom' in kwargs else np.zeros(len(pos[0]))
    
    for yIdx, yName in enumerate(tags):
        if args.labels is not None:
            label = args.labels[yIdx]
        else:
            label = yName
        ax1.bar(pos[0], df[yName], width = args.width, bottom = bottom, label=label,
                edgecolor=args.edgecolor, color=args.colors[yIdx])
        bottom += df[yName]
    ax1.set_xlim(0, pos[0][-1]+args.bar_space+args.width/2)
    ax1.set_xticks(pos[0], [l.name for l in xlabels[0]])
    plt.setp(ax1.axis["bottom"].major_ticklabels, rotation=args.xrotations[0])
    if args.font_size is not None:
        plt.setp(ax1.axis["bottom"].major_ticklabels, fontsize=args.font_size[0])
    
    axs = []
    for idx, sub_pos in enumerate(pos[1:]):
        axs.append(ax1.twiny())
        axs[-1].set_xticks(sub_pos, [l.name for l in xlabels[idx+1]])
        new_axisline = axs[-1].get_grid_helper().new_fixed_axis
        axs[-1].axis["bottom"] = new_axisline(loc="bottom", axes=axs[-1], offset=(0, -args.offset[idx]))
        plt.setp(axs[-1].axis["bottom"].major_ticklabels, rotation=args.xrotations[idx+1])
        if args.font_size is not None:
            plt.setp(axs[-1].axis["bottom"].major_ticklabels, fontsize=args.font_size[idx+1])
        axs[-1].axis["top"].major_ticks.set_ticksize(0)
        # TODO : do not work
        axs[-1].axis["right"].major_ticks.set_ticksize(0)
        
    # err bar, put here because errorbar will change ax obj and occur errs
    if args.err is not None:        
        ax1.errorbar(pos[0], bottom, yerr=1.96 * args.err, **args.err_kwargs)
    
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
    """dev code"""
    df = pd.read_excel('./data/plot.xlsx', sheet_name='MWM')
    df['Animal Type'] = df['Animal Type'].astype('str')
    model = mst.multicomp_turkeyHSD({'Animal Type':[]}, 'Duration', df)
    result = mst.turkey_to_table(model)
    print(result)
    sub_df = get_df_data({'Animal Type':[]}, ['Duration'], df)
    sub_df = pro_bar_data(['Animal Type'], ['Duration'], sub_df)
    # test err
    plot_bar(['Animal Type'], ['Duration'], sub_df, err = sub_df['Duration_SE'])
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
    