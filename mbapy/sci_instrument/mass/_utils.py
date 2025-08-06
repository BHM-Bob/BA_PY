'''
Date: 2024-05-22 10:00:28
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2025-07-24 10:41:49
Description: 
'''

from typing import Callable, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd

if __name__ == '__main__':
    from mbapy.base import get_default_for_None, put_err, put_log
    from mbapy.plot import get_palette, PLT_MARKERS
    from mbapy.sci_instrument._utils import \
        process_num_label_col_marker as process_peak_labels
    from mbapy.sci_instrument.mass._base import MassData
else:
    from ...base import get_default_for_None, put_err, put_log
    from ...plot import get_palette, PLT_MARKERS
    from .._utils import process_num_label_col_marker as process_peak_labels
    from ._base import MassData
    
    
def _plot_vlines(ax, x, y, col, label = None, plot_scatter: bool = True,
                 marker: str = 'o', scatter_size: float = 120,
                 scatter_label: str = None):
    ax.vlines(x, 0, y, colors = [col] * len(x), label = label)
    if plot_scatter:
        ax.scatter(x, y, c = col, s = scatter_size, marker = marker, label = scatter_label)
    
def _plot_tag_by_string_label(ax: plt.Axes, df: pd.DataFrame, data: MassData,
                              labels_eps: float, labels: Dict[str, str],
                              color: str = 'black', tag_fontsize: int  = 15,
                              marker_size: int = 120, tag_monoisotopic_only: bool = False):
    labels_ms = np.array(list(labels.keys()))
    text_col = color
    has_label_matched = False
    charges = df[data.CHARGE_HEADER] if data.CHARGE_HEADER is not None else [None]*len(df)
    is_monoisotopic = df[data.MONOISOTOPIC_HEADER] if data.MONOISOTOPIC_HEADER is not None else [True]*len(df)
    for ms, h, charge, is_mono in zip(df[data.X_HEADER], df[data.Y_HEADER], charges, is_monoisotopic):
        matched = np.where(np.abs(labels_ms - ms) < labels_eps)[0]
        if matched.size > 0:
            label, text_col, maker = labels.get(labels_ms[matched[0]])
            _plot_vlines(ax, [ms], [h], text_col, scatter_label = label, marker = maker, plot_scatter = True, scatter_size=marker_size)
            has_label_matched = True
        elif h < data.plot_params['min_tag_lim']: # skip tag if h < data.plot_params['min_tag_lim']
            continue
        else:
            text_col = color
        charge_str = f'({charge})' if charge is not None else ''
        if (tag_monoisotopic_only and is_mono) or (not tag_monoisotopic_only):
            ax.text(ms, h, f'  {ms:.3f}{charge_str}', fontsize=tag_fontsize, color = text_col,
                    horizontalalignment='left', verticalalignment='center')
    return has_label_matched

def _plot_tag_by_match_df(ax: plt.Axes, df: pd.DataFrame, data: MassData,
                          color: str = 'black', tag_fontsize: int  = 15, marker_size: int = 120,
                          tag_monoisotopic_only: bool = False):
    _check_monoisotopic = lambda is_mono: (tag_monoisotopic_only and is_mono) or (not tag_monoisotopic_only)
    # get color and marker
    if 'color' not in df.columns:
        df['color'] = get_palette(len(data.match_df), 'hls')
    if 'marker' not in df.columns:
        if len(data.match_df) + 1 > len(PLT_MARKERS):
            put_log(f'Not enough markers for {len(data.match_df)} peaks, use default markers instead.')
            df['marker'] = PLT_MARKERS[0]
        else:
            df['marker'] = PLT_MARKERS[1:len(data.match_df)+1]
    match_df = data.match_df
    # plot normal
    charges = data.peak_df[data.CHARGE_HEADER] if data.CHARGE_HEADER is not None else [None]*len(data.peak_df)
    is_monoisotopic = data.peak_df[data.MONOISOTOPIC_HEADER] if data.MONOISOTOPIC_HEADER is not None else [True]*len(data.peak_df)
    for ms, h, charge, is_mono in zip(data.peak_df[data.X_HEADER], data.peak_df[data.Y_HEADER], charges, is_monoisotopic):
        if ms not in match_df['x'] and _check_monoisotopic(is_mono) and h >= data.plot_params['min_tag_lim']:
            charge_str = f'({charge})' if charge is not None else ''
            ax.text(ms, h, f'  {ms:.3f}{charge_str}', fontsize=tag_fontsize, color = color,
                    horizontalalignment='left', verticalalignment='center')
    # plot match
    for x, y, charge, mode, substance, col, marker in zip(match_df['x'], match_df['y'], match_df['c'], match_df['mode'], match_df['substance'], match_df['color'], match_df['marker']):
        if data.X_HEADER == data.X_M_HEADER:
            esi_mode = data.ESI_IRON_MODE[mode]
            x = (x*charge-esi_mode['im'])/esi_mode['m']
        _plot_vlines(ax, [x], [y], col, marker=marker,
                     scatter_label=f'{x:.3f}: {substance}{mode}',
                     plot_scatter=True, scatter_size=marker_size)
        charge_str = f'({charge})' if charge is not None else ''
        if _check_monoisotopic(is_mono):
            ax.text(x, y, f'{x:.3f}{charge_str}\n{mode}', fontsize=tag_fontsize, color = color,
                    horizontalalignment='center', verticalalignment='bottom')
    return True

def plot_mass(data: MassData, ax: plt.Axes = None, fig_size: Tuple[float, float] = (12, 7),
              xlim: Tuple[float, float] = None,
              show_legend: bool = True, legend_fontsize: float = 15,
              legend_pos: Union[str, int] = 'upper right', legend_bbox: Tuple[float, float] = (1.3, 0.75),
              min_height: float = None, min_height_percent: float = 1,
              verbose: bool = True, color: str = 'black',
              labels_eps: float = 0.5, labels: Dict[float, Tuple[str, str]] = {}, use_match_as_label: bool = True,
              tag_fontsize: float = 15, marker_size: float = 120, normal_marker: str = 'o',
              is_y_log: bool = True, tag_monoisotopic_only: bool = False,
              **kwargs):
    """
    Parameters
        - data: MassData object, include a data_df and peak_df as attributes
        - ax: matplotlib.pyplot.Axes object, default None, will create a new one according to fig_size if None
        - fig_size: tuple, default (10, 8), will be used to create a new matplotlib.pyplot.Axes object if ax is None
        - xlim: tuple, default None, will be set to (data.data_df[data.X_HEADER].min(), data.data_df[data.X_HEADER].max())
        - show_legend: bool, default True, will create a legend if there has any label matched
        - legend_fontsize: float, default 15, matplotlib.pyplot.legend() parameter
        - legend_pos: str or int, default 'upper right', matplotlib.pyplot.legend() parameter
        - legend_bbox: tuple, default (1.3, 0.75), matplotlib.pyplot.legend() parameter
        - min_height: float, default None, will be set to data.peak_df[data.Y_HEADER].min() if None
        - min_height_percent: float, default 1, will act as a percentage of max height if not None
        - verbose: bool, default True, FLAG
        - color: str, default 'black', peak-line color
        - labels_eps: float, default 0.5, will match labels within labels_eps
        - labels: Dict[float, Tuple[str, str]], default null dict, key is mass, value is a tuple of (label, text_color, marker)
        - tag_fontsize: float, default 15, peak-tag font size
        - maker_size: float, default 80, peak-tag marker size
        - normal_marker: str, default 'o', not matched peak's peak-tag marker
        - is_y_log: bool, default True, will set y-axis scale to log if True
        - tag_monoisotopic_only: bool, default False, True will only plot tag for monoisotopic peak
        - **kwargs: other keyword arguments for matplotlib.pyplot.Axes.vlines()
        
    Returns
        - ax: matplotlib.pyplot.Axes object
        - _bbox_extra_artists: list of matplotlib.offsetbox.AnchoredOffsetbox objects
    """
    # check ax
    if ax is None:
        fig, ax = plt.subplots(figsize = fig_size)
    # find peaks
    if data.check_processed_data_empty(data.peak_df):
        data.search_peaks(xlim, min_height, min_height_percent)
    df = data.peak_df.copy()
    # set xlim
    auto_xlim = True if xlim is None else False
    xlim = get_default_for_None(xlim, (data.peak_df[data.X_HEADER].min(),
                                       data.peak_df[data.X_HEADER].max()))
    if verbose:
        put_log(f'x-axis data limit set to {xlim}')
    df = df[(df[data.X_HEADER] >= xlim[0]) & (df[data.X_HEADER] <= xlim[1])] # repetitive code if data.peak_df is None
    # check if there has any peak
    if data.check_processed_data_empty(df):
        return put_err('no peaks found, return None')
    # plot
    _plot_vlines(ax, df[data.X_HEADER], df[data.Y_HEADER], color, scatter_size=marker_size//4, marker=normal_marker)
    # plot labels tag
    if use_match_as_label and len(data.match_df) > 0:
        if data.X_HEADER == data.X_M_HEADER:
            put_log(f'using x-data like "Mass (charge)" on matched peaks may cause unexpected mass transfer results')
        has_label_matched = _plot_tag_by_match_df(ax, data.match_df, data, color, tag_fontsize, marker_size, tag_monoisotopic_only)
    else:
        has_label_matched = _plot_tag_by_string_label(ax, df, data, labels_eps, labels, color, tag_fontsize, marker_size, tag_monoisotopic_only)
    # legend
    _bbox_extra_artists = []
    if show_legend and has_label_matched:
        legend = ax.legend(loc = legend_pos, bbox_to_anchor = legend_bbox,
                           fontsize = legend_fontsize, draggable = True)
        _bbox_extra_artists.append(legend)
    # fix style
    if not auto_xlim: # because when auto_xlim is True, the min x is in the edge of the figure
        put_log(f'x-axis plot limit set to {xlim}')
        ax.set_xlim(xlim)
    if is_y_log:
        ax.set_yscale('log')
    else:
        ax.set_yscale('linear')
    # return
    return ax, _bbox_extra_artists
    

__all__ = [
    'plot_mass',
    'process_peak_labels',
]


if __name__ == '__main__':
    from mbapy.plot import save_show
    from mbapy.sci_instrument.mass.SCIEX import SciexPeakListData
    data = SciexPeakListData(r'data_tmp\scripts\mass\pl.xlsx')
    labels = {362: ('M+H', 'blue', 'x'),}
    ax, bbox = plot_mass(data, labels = labels, tag_monoisotopic_only = True)
    save_show(r'data_tmp\scripts\mass\pl.png', bbox_extra_artists = bbox)