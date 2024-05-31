'''
Date: 2024-05-22 10:00:28
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2024-05-30 17:38:51
Description: 
'''

from typing import Callable, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy

if __name__ == '__main__':
    from mbapy.base import get_default_for_None, put_err
    from mbapy.sci_instrument._utils import \
        process_num_label_col as process_peak_labels
    from mbapy.sci_instrument.mass._base import MassData
else:
    from ...base import get_default_for_None, put_err
    from .._utils import process_num_label_col as process_peak_labels
    from ._base import MassData
    

def plot_mass(data: MassData, ax: plt.Axes = None, fig_size: Tuple[float, float] = (10, 8),
              xlim: Tuple[float, float] = None,
              show_legend: bool = True, legend_fontsize: float = 15,
              legend_pos: Union[str, int] = 'upper right', legend_bbox: Tuple[float, float] = (1.3, 0.75),
              min_height: float = None, min_height_percent: float = 1,
              verbose: bool = True, color: str = 'black',
              labels_eps: float = 0.5, labels: Dict[float, Tuple[str, str]] = {},
              tag_fontsize: float = 15, y_log_scale: bool = True,
              **kwargs):
    """
    Parameters
        - data: MassData object
        - ax: matplotlib.pyplot.Axes object, default None
        - fig_size: tuple, default (10, 8)
        - xlim: tuple, default None, will be set to (data.data_df[data.X_HEADER].min(), data.data_df[data.X_HEADER].max())
        - show_legend: bool, default True
        - legend_fontsize: float, default 15
        - legend_pos: str or int, default 'upper right'
        - legend_bbox: tuple, default (1.3, 0.75)
        - min_height: float, default None
        - min_height_percent: float, default 1
        - verbose: bool, default True
        - color: str, default 'black'
        - labels_eps: float, default 0.5
        - labels: Dict[float, Tuple[str, str]], default null dict
        - tag_fontsize: float, default 15
        - **kwargs: other keyword arguments for matplotlib.pyplot.Axes.vlines()
        
    Returns
        - ax: matplotlib.pyplot.Axes object
        - _bbox_extra_artists: list of matplotlib.offsetbox.AnchoredOffsetbox objects
    """
    # check ax
    if ax is None:
        _, ax = plt.subplots(figsize = fig_size)
    # helper function
    def _plot_vlines(ax, x, y, col, label = None):
        ax.vlines(x, 0, y, colors = [col] * len(x), label = label)
        ax.scatter(x, y, c = col)
    # find peaks
    xlim = get_default_for_None(xlim, (data.data_df[data.X_HEADER].min(),
                                       data.data_df[data.X_HEADER].max()))
    if verbose:
        print(f'x-axis data limit set to {xlim}')
    if data.check_processed_data_empty(data.peak_df):
        data.search_peaks(xlim, min_height, min_height_percent)
    df = data.peak_df.copy()
    df = df[(df[data.X_HEADER] >= xlim[0]) & (df[data.X_HEADER] <= xlim[1])] # repetitive code if data.peak_df is None
    if data.check_processed_data_empty(df):
        return put_err('no peaks found, return None')
    # plot
    _plot_vlines(df['Mass/Charge'], df['Intensity'], color)
    labels_ms = np.array(list(labels.keys()))
    text_col = color
    for ms, h in zip(df['Mass/Charge'], df['Intensity']):
        matched = np.where(np.abs(labels_ms - ms) < labels_eps)[0]
        if matched.size > 0:
            label, text_col = labels.get(labels_ms[matched[0]])
            _plot_vlines([ms], [h], text_col, label)
        else:
            text_col = color
        ax.text(ms, h, f'* {ms:.2f}', fontsize=tag_fontsize, color = text_col)
    # legend
    _bbox_extra_artists = []
    if show_legend:
        legend = ax.legend(loc = legend_pos, bbox_to_anchor = legend_bbox,
                           fontsize = legend_fontsize, draggable = True)
        _bbox_extra_artists.append(legend)
    # fix style
    ax.set_yscale('log')
    # return
    return ax, _bbox_extra_artists
    

__all__ = [
    'plot_mass',
    'process_peak_labels',
]