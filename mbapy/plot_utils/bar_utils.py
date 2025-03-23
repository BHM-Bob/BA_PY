from functools import wraps
from itertools import chain, combinations
from typing import Callable, Dict, List, Tuple, Union, Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits import axisartist
from mpl_toolkits.axes_grid1 import host_subplot

if __name__ == '__main__':
    # dev mode
    from mbapy.base import get_wanted_args, get_default_for_None
    from mbapy.stats.df import (get_df_data, pro_bar_data, pro_bar_data_R,
                                sort_df_factors)
else:
    # release mode
    from ..base import get_wanted_args, get_default_for_None
    from ..stats.df import (get_df_data, pro_bar_data, pro_bar_data_R,
                           sort_df_factors)

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
            - figsize = (8, 6)
            - jitter = Flase, IF True, pass the original df, the func will call `pro_bar_data` internally.
            - jitter_kwargs = {'size': 10, 'alpha': 0.5, 'color': None}
            - width = 0.4
            - bar_space = 0.05
            - hue_space = 0.2
            - xrotations = [0]*len(factors)
            - colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            - hatchs:None, could be [['-', '+', 'x', '\\'], ['*', 'o', 'O', '.']],
            - font_size:None,
            - labels:None,
            - ylabel: None, default is the first tag, fontsize will be set same as TOP level's x-axis' fontsize.
            - offset = [None] + [(i+1)*(plt.rcParams['font.size']+8) for i in range(len(factors)-1)], x-axis offset
            - xticks_pad: [5 for _ in range(len(factors)+1)], x-axis label pad from x-axis
            - edgecolor:['white'] * len(tags),
            - linewidth: 0,
            - err: (str|np.array), if str, will use df[err] as yerr. if np.array, will use it directly. will multiply 1.96 to yerr.
            - err_kwargs = 
                - 'capsize':5, # error bar cap size
                - 'capthick':2, # error bar cap thickness
                - 'elinewidth':2, # error bar line width
                - 'fmt':' k', # error bar format. 'k' means black.
                - 'ecolor':'black', # error bar color. support list of hex color.

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
    args = get_wanted_args({'figsize': (8, 6),
                            'dpi': 100,
                            'jitter':False,
                            'jitter_kwargs': {'size': 10, 'alpha': 0.5, 'color': None},
                            'width':0.4, 'hue_space':0.2, 'bar_space':0.05,
                            'xrotations':[0]*len(factors),
                            'colors':plt.rcParams['axes.prop_cycle'].by_key()['color'],
                            'hatchs': None,
                            'font_size':None,
                            'labels':None,
                            'ylabel': None,
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
    ax1 = host_subplot(111, axes_class=axisartist.Axes)
    ax1.figure.set_size_inches(args.figsize)
    ax1.figure.set_dpi(args.dpi)
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
        hatch = args.hatchs[yIdx] if (args.hatchs and args.hatchs[yIdx] is not None) else None
        ax1.bar(pos[0], df[yName], width = args.width, bottom = bottom, label=label,
                edgecolor=args.edgecolor[yIdx], linewidth = args.linewidth, hatch = hatch,
                color=args.colors[yIdx], log = args.log, zorder = 0)
        bottom += df[yName]
    ax1.set_xlim(0, pos[0][-1]+args.hue_space+args.width/2)
    ax1.set_xticks(pos[0], [l.name for l in xlabels[0]])
    plt.setp(ax1.axis["bottom"].major_ticklabels, rotation=args.xrotations[0], pad = args.xticks_pad[0])
    if args.font_size is not None:
        plt.setp(ax1.axis["bottom"].major_ticklabels, fontsize=args.font_size[0])
        plt.setp(ax1.axis["left"].major_ticklabels, fontsize=args.font_size[0])
        ax1.axis['left'].set_label(get_default_for_None(args.ylabel, tags[0]))
        ax1.axis['left'].label.set_fontsize(args.font_size[-1])
    
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
            margs = get_wanted_args({'width':0.4, 'hue_space': 0.2, 'bar_space':0.2, 'xrotations':[0]*len(factors),
                                    'colors':plt.rcParams['axes.prop_cycle'].by_key()['color'],
                                    'offset':[(i+1)*(plt.rcParams['font.size']+8) for i in range(len(factors))]},
                                   kwargs)
            margs.xrotations.append(0)
            xlabels, pos = pro_hue_pos(factors, df, margs.width, margs.hue_space, margs.bar_space)
            margs.add_arg('xlabels', xlabels)
            margs.add_arg('bottom', np.zeros(len(pos[0])))
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


__all__ = [
    'AxisLabel'
    'pro_hue_pos',
    'plot_bar'
    'plot_positional_hue',
    ]


if __name__ == '__main__':
    # dev code
    from mbapy.plot import get_palette
    df = pd.read_excel('./data/plot.xlsx', sheet_name='MWM')
    cols = get_palette(n = 4, mode = 'hls')
    plot_bar(['Animal Type'], ['Duration'], df,
             err = 'Duration_SE', err_kwargs = {'capsize':5, 'capthick':2, 'elinewidth':2, 'fmt':' k', 'ecolor':cols},
             figsize = (8, 6),
             edgecolor = [cols], linewidth = 5, colors = ['white'],
             font_size = [10, 20],
             jitter = True, jitter_kwargs = {'size': 10, 'alpha': 0.6, 'color': [cols]})
    plt.show()

    