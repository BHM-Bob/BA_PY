
from typing import Callable, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

if __name__ == '__main__':
    from mbapy.base import put_err
    from mbapy.sci_instrument._utils import \
        process_label_col as process_file_labels
    from mbapy.sci_instrument._utils import \
        process_num_label_col as process_peak_labels
    from mbapy.sci_instrument.hplc._base import HplcData
else:
    from ...base import put_err
    from .._utils import process_label_col as process_file_labels
    from .._utils import process_num_label_col as process_peak_labels
    from ._base import HplcData
    

def plot_hplc(hplc_data: Union[HplcData, List[HplcData]],
              ax = None, fig_size = (12, 7), y_log_scale: bool = False,
              dfs_refinment_x: Dict[str, float] = {}, dfs_refinment_y: Dict[str, float] = {},
              file_labels: Union[str, Tuple[str, str, str]] = [], file_label_fn: Callable = process_file_labels,
              show_file_legend = True, file_legend_pos = 'upper right', file_legend_bbox = (1.3, 0.75),
              peak_labels: Union[str, Dict[float, Tuple[str, str]]] = {}, peak_label_fn: Callable = process_peak_labels,
              plot_peaks_underline: bool = False, plot_peaks_line: bool = False,
              plot_peaks_area: bool = False, peak_area_alpha = 0.3,
              show_tag_legend = True, peak_legend_pos = 'upper right', peak_legend_bbox = (1.3, 1),
              start_search_time: float = 0, end_search_time = None, labels_eps = 0.1, min_height = 0, min_peak_width = 0.1,
              marker_offset = (0, 0.05), marker_size = 80,
              show_tag_text = True, tag_offset = (0.05, 0.05), tag_fontsize = 15,
              dpi = 600, line_width = 2, legend_fontsize = 15, **kwargs) -> Tuple[plt.Axes, List[plt.Artist], Dict[str, np.ndarray]]:
    """
    Parameters:
        - hplc_data: Union[HPLC_Data, List[HPLC_Data]], HPLC_Data or list of HPLC_Data
        - ax: matplotlib.axes.Axes, default is None, if None, create a new figure and axes
        - fig_size: Tuple[float, float], figure size
        - dfs_refinement_x: X-data offset dictionary, key is data tag, value is offset value
        - dfs_refinement_y: Y-data offset dictionary, key is data tag, value is offset value
        - file_labels: str, file labels string, such as label,color;label,color;...
        - file_label_fn: Callable, accepts a string of labels and a string of color mode for mbapy.plot.get_palette, and returns a dictionary of file labels
        - show_file_legend: FLAG, whether to show file labels
        - file_legend_pos: str, file_legend_pos of file legend
        - file_legend_bbox: Tuple[float, float], bbox_to_anchor of file legend
        - peak_labels: str, peak labels string, such as time,label,color;time,label,color;...
        - peak_label_fn: Callable, accepts a string of label and a string of color mode for mbapy.plot.get_palette, and returns a dictionary of peak labels
        - plot_peaks_underline: FLAG, whether to plot peaks underline, underline is calcu by HplcData.calcu_peaks_area
        - plot_peaks_line: FLAG, whether to plot peaks line in the same color of peak markers
        - plot_peaks_area: FLAG, whether to plot peaks area
        - show_tag_legend: FLAG, whether to show peak labels
        - peak_legend_pos: str, peak_legend_pos of peak legend
        - peak_legend_bbox: Tuple[float, float], bbox_to_anchor of peak legend
        - start_search_time: start of search time, in minutes
        - end_search_time: end of search time, in minutes
        - labels_eps: eps for matching peak labels
        - min_height: min height for peak detection
        - min_peak_width: min peak width for peak detection, in minutes, default is 0.1
        - marker_offset: Tuple[float, float], offset for peak markers
        - marker_size: peak marker size
        - show_tag_text: FLAG, whether to show tag text
        - tag_offset: Tuple[float, float], offset for tag text
        - tag_fontsize: tag text font size
        - dpi: DPI, default is 600
        - line_width: line width for plot
        - legend_fontsize: legend font size

    Returns:
        - ax: plt.Axes: ax
        - _bbox_extra_artists: List[plt.Artist]: legend artists, for saving
        - files_peaks_idx: Dict[str, np.ndarray]: peaks index array of each data, key is data tag, value is index array filtered by start_search_time and end_search_time
        - file_labels: Dict[str, Tuple[str, str]]: file labels, key is data tag, value is (label_string, color)
    
    Notes:
        - if one's data has no peaks, it will not exist in the `files_peaks_idx` return value
    """
    names, data_dfs = [], []
    hplc_data = [hplc_data] if isinstance(hplc_data, HplcData) else hplc_data
    # apply x, y offset
    for data in hplc_data:
        names.append(data.get_tag())
        data.refined_abs_data = data.get_abs_data(origin_data=True).copy()
        data.refined_abs_data[data.X_HEADER] += dfs_refinment_x.get(data.get_tag(), 0)
        data.refined_abs_data[data.Y_HEADER] += dfs_refinment_y.get(data.get_tag(), 0)
        data_dfs.append(data.get_abs_data(origin_data=False))
    # return if no data
    if len(data_dfs) == 0:
        return put_err('no data to plot, return None')
    # process peaks labels
    if isinstance(file_labels, str):
        file_labels = file_label_fn(file_labels)
    if not file_labels or len(file_labels) != len(names):
        put_err(f'only {len(file_labels)} labels found, should be {len(names)} labels, use name instead')
        file_labels = file_label_fn(';'.join(names))
        if len(file_labels) == 1:
            file_labels[0][1] = 'black' # 避免使用调色板颜色的单个标签
    # process peak labels
    if isinstance(peak_labels, str):
        peak_labels = peak_label_fn(peak_labels)
    peak_labels_v = np.array(list(peak_labels.keys()))
    # plot each data
    if ax is None:
        _, ax = plt.subplots(figsize = fig_size)
    ax.figure.set_dpi(dpi)
    lines, scatters, sc_labels, files_peaks_idx = [], [], [], {}
    for label, data_i, data_df_i in zip(file_labels, hplc_data, data_dfs):
        label_string, color = label
        line = ax.plot(data_df_i[data_i.X_HEADER], data_df_i[data_i.Y_HEADER],
                       color = color, label = label_string, linewidth = line_width)[0]
        lines.append(line)
        # search peaks
        peaks_idx = data_i.search_peaks(min_peak_width, min_height, start_search_time, end_search_time)
        if peaks_idx.size > 0:
            files_peaks_idx[data_i.get_tag()] = peaks_idx
        peak_df = data_df_i.iloc[peaks_idx, :]
        # if do not plot peak label, skip
        if not data_i.plot_params['peak_label']:
            continue
        # plot peaks and matched labels
        for t, peak_idx, a in zip(peak_df[data_i.X_HEADER], peaks_idx, peak_df[data_i.Y_HEADER]):                    
            matched = np.where(np.abs(peak_labels_v - t) < labels_eps)[0]
            if matched.size > 0:
                label, col = peak_labels[peak_labels_v[matched[0]]]
                sc = ax.scatter(t+marker_offset[0], a+marker_offset[1], marker='*', s = marker_size, color = col)
                scatters.append(sc)
                sc_labels.append(label)
            else:
                col = 'black'
                ax.scatter(t+marker_offset[0], a+marker_offset[1], marker=11, s = marker_size, color = col)
            if show_tag_text:
                ax.text(t+tag_offset[0], a+tag_offset[1], f'{t:.2f}', fontsize=tag_fontsize, color = col)
            # plot peak-line, underline and area
            if any([plot_peaks_line, plot_peaks_underline, plot_peaks_area]):
                areas_info = data_i.get_area(peaks_idx)
                if peak_idx in areas_info: # check if peak is not filtered by start_search_time and end_search_time
                    area_info = areas_info[peak_idx]
                    if plot_peaks_line:
                        ax.plot(area_info['underline-x'], area_info['peak-line-y'], color = col, linewidth = line_width)
                    if plot_peaks_underline:
                        ax.plot(area_info['underline-x'], area_info['underline-y'], color = col, linewidth = line_width)
                    if plot_peaks_area:
                        ax.fill_between(area_info['underline-x'], area_info['peak-line-y'], area_info['underline-y'],
                                color = col, alpha = peak_area_alpha)
                
    # set y scale
    if y_log_scale:
        ax.set_yscale('log')
    # make file labels legend
    _bbox_extra_artists = []
    if show_file_legend:
        file_legend = plt.legend(fontsize=legend_fontsize, loc = file_legend_pos,
                                bbox_to_anchor = file_legend_bbox, draggable = True)
        ax.add_artist(file_legend)
        _bbox_extra_artists.append(file_legend)
    # make peak labels legend
    if scatters and show_tag_legend:
        [line.set_label(None) for line in lines]
        [sc.set_label(l) for sc, l in zip(scatters, sc_labels)]
        peak_legend = plt.legend(fontsize=legend_fontsize, loc = peak_legend_pos,
                                 bbox_to_anchor = peak_legend_bbox, draggable = True)
        ax.add_artist(peak_legend)
        _bbox_extra_artists.append(peak_legend)
    return ax, _bbox_extra_artists, files_peaks_idx, file_labels


def plot_pda_heatmap(hplc_data: HplcData, ax = None, fig_size = (12, 7),
                     cmap: str = 'Reds', n_xticklabels: int = 60, n_yticklabels: int = 10, **kwargs):
    """
    plot a heatmap of PDA data, with x-axis as time and y-axis as wave length.

    Parameters:
        - hplc_data: HplcData, PDA data
        - ax: matplotlib.axes.Axes, default is None, if None, create a new figure and axes
        - fig_size: Tuple[float, float], figure size, used when ax is None
        - cmap: str, colormap name, passed to sns.heatmap, default is 'Reds'
        - n_xticklabels: int, make a tick every n_xticklabels, passed to sns.heatmap, default is 60
        - n_yticklabels: int, make a tick every n_yticklabels, passed to sns.heatmap, default is 10
        
    Returns:
        - ax: plt.Axes: axes object
        - ax_topx: plt.Axes: new axes object for top x axis, used for setting xticklabels
    """
    # tick helper func
    def make_tick(ax, axis: str, n_ticks: int):
        ticks = list(map(lambda x: (x[0], f'{float(x[1]._text):.2f}'),
                        zip(getattr(ax, f'get_{axis}ticks')(), getattr(ax, f'get_{axis}ticklabels')())))
        filtered_ticks = ticks[::n_ticks]
        if ticks[-1] not in filtered_ticks:
            if len(ticks) - ticks.index(filtered_ticks[-1]) < 0.75*n_ticks:
                filtered_ticks.pop(-1)
            filtered_ticks.append(ticks[-1])
        return filtered_ticks
    df = hplc_data.data_df.copy(True)
    df.set_index(hplc_data.X_HEADER, inplace=True, drop=True)
    if ax is None:
        _, ax = plt.subplots(figsize = fig_size)
    ax = sns.heatmap(df.T, cmap=cmap, ax = ax, cbar_kws={'label': 'Absorbance (AU)'},
                     xticklabels=1, yticklabels=1)
    # set bottom X axis' ticks to .2f and Y axis' ticks to .1f
    xticks, yticks = make_tick(ax, 'x', n_xticklabels), make_tick(ax, 'y', n_yticklabels)
    ax.set_xticks(list(map(lambda x: x[0], xticks)))
    ax.set_yticks(list(map(lambda x: x[0], yticks)))
    ax.set_xticklabels(list(map(lambda x: f'{float(x._text):.2f}', ax.get_xticklabels())))
    ax.set_yticklabels(list(map(lambda x: f'{float(x._text):.1f}', ax.get_yticklabels())))
    # make a new axis for top x axis
    ax_topx = ax.twiny() # call ax_topx.xaxis.tick_top() inner this func
    ax_topx.set_xticks(ax.get_xticks(), ax.get_xticklabels())
    # set axis' label
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Wave Length (nm)')
    return ax, ax_topx


__all__ = [
    'process_file_labels',
    'process_peak_labels',
    'plot_hplc',
    'plot_pda_heatmap',
    ]


if __name__ == '__main__':
    from mbapy.sci_instrument.hplc.waters import WatersPdaData
    data = WatersPdaData('data_tmp/scripts/hplc/WatersPDA.arw')
    data.set_opt_wave_length(228)
    plot_hplc(data, start_search_time=4, dfs_refinment_x={data.get_tag(): -3},
              plot_peaks_underline=True, plot_peaks_line=True, plot_peaks_area=True, dpi = 100)
    plt.show()
    plot_pda_heatmap(data)
    plt.show()