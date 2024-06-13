'''
Date: 2024-05-22 10:00:28
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2024-06-13 10:48:30
Description: 
'''

from typing import Callable, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np

if __name__ == '__main__':
    from mbapy.base import get_default_for_None, put_err
    from mbapy.sci_instrument._utils import \
        process_num_label_col_marker as process_peak_labels
    from mbapy.sci_instrument.mass._base import MassData
else:
    from ...base import get_default_for_None, put_err
    from .._utils import process_num_label_col_marker as process_peak_labels
    from ._base import MassData
    

def plot_mass(data: MassData, ax: plt.Axes = None, fig_size: Tuple[float, float] = (12, 7),
              xlim: Tuple[float, float] = None,
              show_legend: bool = True, legend_fontsize: float = 15,
              legend_pos: Union[str, int] = 'upper right', legend_bbox: Tuple[float, float] = (1.3, 0.75),
              min_height: float = None, min_height_percent: float = 1,
              verbose: bool = True, color: str = 'black',
              labels_eps: float = 0.5, labels: Dict[float, Tuple[str, str]] = {},
              tag_fontsize: float = 15, marker_size: float = 120, normal_marker: str = 'o',
              is_y_log: bool = True,
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
        - **kwargs: other keyword arguments for matplotlib.pyplot.Axes.vlines()
        
    Returns
        - ax: matplotlib.pyplot.Axes object
        - _bbox_extra_artists: list of matplotlib.offsetbox.AnchoredOffsetbox objects
    """
    # check ax
    if ax is None:
        fig, ax = plt.subplots(figsize = fig_size)
    # helper function
    def _plot_vlines(ax, x, y, col, label = None, plot_scatter: bool = True,
                     marker: str = normal_marker, scatter_size: float = marker_size,
                     scatter_label: str = None):
        ax.vlines(x, 0, y, colors = [col] * len(x), label = label)
        if plot_scatter:
            ax.scatter(x, y, c = col, s = scatter_size, marker = marker, label = scatter_label)
    # find peaks
    if data.check_processed_data_empty(data.peak_df):
        data.search_peaks(xlim, min_height, min_height_percent)
    df = data.peak_df.copy()
    # set xlim
    xlim = get_default_for_None(xlim, (data.peak_df[data.X_HEADER].min(),
                                       data.peak_df[data.X_HEADER].max()))
    if verbose:
        print(f'x-axis data limit set to {xlim}')
    df = df[(df[data.X_HEADER] >= xlim[0]) & (df[data.X_HEADER] <= xlim[1])] # repetitive code if data.peak_df is None
    # check if there has any peak
    if data.check_processed_data_empty(df):
        return put_err('no peaks found, return None')
    # plot
    _plot_vlines(ax, df[data.X_HEADER], df[data.Y_HEADER], color, scatter_size=marker_size//4)
    labels_ms = np.array(list(labels.keys()))
    text_col = color
    has_label_matched = False
    charges = df[data.CHARGE_HEADER] if data.CHARGE_HEADER is not None else [None]*len(df)
    for ms, h, charge in zip(df[data.X_HEADER], df[data.Y_HEADER], charges):
        matched = np.where(np.abs(labels_ms - ms) < labels_eps)[0]
        if matched.size > 0:
            label, text_col, maker = labels.get(labels_ms[matched[0]])
            _plot_vlines(ax, [ms], [h], text_col, scatter_label = label, marker = maker, plot_scatter = True, scatter_size=marker_size)
            has_label_matched = True
        else:
            text_col = color
        charge_str = f'({charge})' if charge is not None else ''
        ax.text(ms, h, f'  {ms:.2f}{charge_str}', fontsize=tag_fontsize, color = text_col)
    # legend
    _bbox_extra_artists = []
    if show_legend and has_label_matched:
        legend = ax.legend(loc = legend_pos, bbox_to_anchor = legend_bbox,
                           fontsize = legend_fontsize, draggable = True)
        _bbox_extra_artists.append(legend)
    # fix style
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
    from mbapy.sci_instrument.mass.SCIEX import SciexPeakListData
    data = SciexPeakListData(r'data_tmp\scripts\mass\B4 pl.txt')
    labels = {453: ('M+H', 'blue', 'x'),}
    plot_mass(data, labels=labels)