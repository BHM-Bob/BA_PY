import itertools
import sys
from functools import wraps
from typing import Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from mpl_toolkits import axisartist
from mpl_toolkits.axes_grid1 import host_subplot

from mbapy.base import get_wanted_args

# plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文
plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def rgb2hex(r, g, b):
  return '#'+('{:02X}' * 3).format(r, g, b)
def hex2rgb(hex:str):
  return [int(hex[i:i+2], 16) for i in (1, 3, 5)]
def rgbs2hexs(rgbs:list[tuple[float]]):
    return list(map(lambda x : rgb2hex(*[int(x[i]*255) for i in range(3)]),
                    rgbs))
    
def get_palette(n:int = 10, mode:Union[None, str] = None) -> list[str]:
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
    assert n >= 1
    if mode == 'hls':
        return rgbs2hexs(sns.color_palette('hls', n))
    if n <= 5 and mode == 'green':
        return ['#80ab1c', '#405535', '#99b69b', '#92e4ce', '#72cb87'][0:n]
    elif n <= 9:
        return rgbs2hexs(plt.get_cmap('Set1').colors)
    elif n <= 12:
        return rgbs2hexs(plt.get_cmap('Set3').colors)
    elif n <= 20 and mode == 'pair':
        return rgbs2hexs(plt.get_cmap('tab20').colors)
    
# TODO : not use itertools.product
def pro_bar_data(factors:list[str], tags:list[str], df:pd.DataFrame, **kwargs):
    """
    cacu mean and SE for each combinations of facotors\n
    data should be like this:\n
    | factor1 | factor2 | y1 | y2 |...\n
    |  f1_1   |   f2_1  |2.1 |-2  |...\n
    after process\n
    | factor1 | factor2 | y1(mean) | y1_SE(SE) | y1_N(sum_data) |...\n
    |  f1_1   |   f2_1  |2.1       |   -2      |   32           |...\n
    kwargs:
        min_sample_N:int : min N threshold(>=)
    """
    # kwargs
    min_sample_N = 1 if 'min_sample_N' not in kwargs else kwargs['min_sample_N']
    assert min_sample_N > 0, 'min_sample_N <= 0'
    # pro
    if len(tags) == 0:
        tags = list(df.columns)[len(factors):]
    factor_contents:list[list[str]] = [ df[f].unique().tolist() for f in factors ]
    ndf = [factors.copy()]
    for tag in tags:
        ndf[0] += [tag, tag+'_SE', tag+'_N']
    for factorCombi in itertools.product(*factor_contents):
        factorMask = np.array(df[factors[0]] == factorCombi[0])
        for i in range(1, len(factors)):
            factorMask &= np.array(df[factors[i]] == factorCombi[i])
        if factorMask.sum() >= min_sample_N:
            line = []
            for idx, tag in enumerate(tags):
                values = np.array(df.loc[factorMask, [tag]])
                line.append(values.mean())
                if values.shape[0] > 1:
                    line.append(values.std(ddof = 1)/np.sqrt(values.shape[0]))
                else:
                    line.append(np.NaN)
                line.append(values.shape[0])
            ndf.append(list(factorCombi) + line)
    return pd.DataFrame(ndf[1:], columns=ndf[0])

def pro_bar_data_R(factors:list[str], tags:list[str], df:pd.DataFrame, suffixs:list[str], **kwargs):
    """
    wrapper\n
    @pro_bar_data_R(['solution', 'type'], ['root', 'leaf'], ndf)\n
    def plot_func(values, **kwargs):
        return produced vars in list format whose length equal to len(suffix)
    """
    def ret_wrapper(core_func):
        def core_wrapper(**kwargs):
            nonlocal tags
            if len(tags) == 0:
                tags = list(df.columns)[len(factors):]
            factor_contents:list[list[str]] = [ df[f].unique().tolist() for f in factors ]
            ndf = [factors.copy()]
            for tag in tags:
                for suffix in suffixs:
                    ndf[0] += [tag+suffix]
            for factorCombi in itertools.product(*factor_contents):
                factorMask = np.array(df[factors[0]] == factorCombi[0])
                for i in range(1, len(factors)):
                    factorMask &= np.array(df[factors[i]] == factorCombi[i])
                if(factorMask.sum() > 0):
                    line = []
                    for idx, tag in enumerate(tags):
                        values = np.array(df.loc[factorMask, [tag]])
                        ret_line = core_func(values)
                        assert len(ret_line) == len(suffixs), 'length of return value of core_func != len(suffixs)'
                        line += ret_line
                    ndf.append(list(factorCombi) + line)
            return pd.DataFrame(ndf[1:], columns=ndf[0])
        return core_wrapper
    return ret_wrapper

def get_df_data(factors:dict[str, list[str]], tags:list[str], df:pd.DataFrame,
                include_factors:bool = True):
    #sub_df = ndf.loc[(ndf['size'] == size1) & (ndf['light'] == light1), ['c', 'w', 'SE']]
    #sub_df = get_df_data([{'size':[size1], 'light':[light1]}, ['c', 'w', 'SE'])
    def update_mask(mask, other:np.ndarray, method:str = '&'):
        return other if mask is None else (mask&other if method == '&' else mask|other)
    if len(tags) == 0:
        tags = list(set(df.columns.to_list())-set(factors.keys()))
    if include_factors:
        tags = list(factors.keys()) + tags
    mask = None
    for factor_name in factors:
        sub_mask = None
        if len(factors[factor_name]) == 0:
            # factors[factor_name] asigned with [], get all sub factors
            factors[factor_name] = df[factor_name].unique().tolist()
        for sub_factor in factors[factor_name]:
            sub_mask = update_mask(sub_mask, np.array(df[factor_name] == sub_factor), '|')
        mask = update_mask(mask, sub_mask, '&')
    return df.loc[mask, tags]

def sort_df_factors(factors:list[str], tags:list[str], df:pd.DataFrame):
    """UnTested
    sort each combinations of facotors\n
    data should be like this:\n
    | factor1 | factor2 | y1 | y2 |...\n
    |  f1_1   |   f2_1  |2.1 |-2  |...\n
    |  f1_1   |   f2_2  |2.1 |-2  |...\n
    ...\n
    after sort if given facotors=['factor2', 'factor1']\n
    | factor2 | factor1 | y1 | y2 |...\n
    |  f2_1   |   f1_1  |2.1 |-2  |...\n
    |  f2_1   |   f1_2  |2.1 |-2  |...\n
    ...\n
    """
    if len(tags) == 0:
        tags = list(df.columns)[len(factors):]
    factor_contents:list[list[str]] = [ df[f].unique().tolist() for f in factors ]
    ndf = [factors.copy()]
    ndf[0] += tags
    for factorCombi in itertools.product(*factor_contents):
        factorMask = df[factors[0]] == factorCombi[0]
        for i in range(1, len(factors)):
            factorMask &= df[factors[i]] == factorCombi[i]
        ndf.append(list(factorCombi) + np.array(df.loc[factorMask, tags].values))
    return pd.DataFrame(ndf[1:], columns=ndf[0])

class AxisLable():
    def __init__(self, name:str, hold_space:int = 1) -> None:
        self.name = name
        self.hold_space = hold_space
    def add_space(self, space:int = 1):
        self.hold_space += space

def pro_hue_pos(factors:list[str], df:pd.DataFrame, width:float, bar_space:float):
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

def plot_bar(factors:list[str], tags:list[str], df:pd.DataFrame, **kwargs):
    """
    stack bar plot with hue style\n
    factors:[low_lever_factor, medium_lever_factor, ...] or just one
    tags:[stack_low_y, stack_medium_y, ...] or just one
    df:df from pro_bar_data or sort_df_factors
        kwargs:
    width = 0.4
    bar_space = 0.2
    xrotations = [0]*len(factors)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    offset = [(i+1)*(plt.rcParams['font.size']+8) for i in range(len(factors))]
    """
    ax1 = host_subplot(111, axes_class=axisartist.Axes)
    
    if len(tags) == 0:
        tags = list(df.columns)[len(factors):]    
    args = get_wanted_args({'width':0.4, 'bar_space':0.2, 'xrotations':[0]*len(factors),
                            'colors':plt.rcParams['axes.prop_cycle'].by_key()['color'],
                            'offset':[(i+1)*(plt.rcParams['font.size']+8) for i in range(len(factors))]},
                            kwargs, locals())
    args.xrotations.append(0)    
    xlabels, pos = pro_hue_pos(factors, df, args.width, args.bar_space)
    bottom = kwargs['bottom'] if 'bottom' in kwargs else np.zeros(len(pos[0]))
    
    for yIdx, yName in enumerate(tags):
        ax1.bar(pos[0], df[yName], width = args.width, bottom = bottom, label=yName,
                edgecolor='white', color=args.colors[yIdx])
        bottom += df[yName]
    ax1.set_xlim(0, pos[0][-1]+args.bar_space+args.width/2)
    ax1.set_xticks(pos[0], [l.name for l in xlabels[0]])
    plt.setp(ax1.axis["bottom"].major_ticklabels, rotation=args.xrotations[0])
    
    axs = []
    for idx, sub_pos in enumerate(pos[1:]):
        axs.append(ax1.twiny())
        axs[-1].set_xticks(sub_pos, [l.name for l in xlabels[idx+1]])
        new_axisline = axs[-1].get_grid_helper().new_fixed_axis
        axs[-1].axis["bottom"] = new_axisline(loc="bottom", axes=axs[-1], offset=(0, -args.offset[idx]))
        plt.setp(axs[-1].axis["bottom"].major_ticklabels, rotation=args.xrotations[idx+1])
        axs[-1].axis["top"].major_ticks.set_ticksize(0)
        # TODO : do not work
        axs[-1].axis["right"].major_ticks.set_ticksize(0)
    
    return np.array(pos[0]), ax1

def plot_positional_hue(factors:list[str], tags:list[str], df:pd.DataFrame, **kwargs):
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

def qqplot(tags:list[str], df:pd.DataFrame, figsize = (12, 6), nrows = 1, ncols = 1, **kwargs):
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
            
def save_show(path:str, dpi = 300, bbox_inches = 'tight'):
    plt.tight_layout()
    plt.gcf().savefig(path, dpi=dpi, bbox_inches = bbox_inches)
    plt.show()