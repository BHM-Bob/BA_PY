
from typing import Callable, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy

if __name__ == '__main__':
    from mbapy.base import put_err
    from mbapy.plot import get_palette
    from mbapy.sci_instrument._utils import \
        process_label_col as process_file_labels
    from mbapy.sci_instrument._utils import \
        process_num_label_col as process_peak_labels
    from mbapy.sci_instrument.hplc._base import HplcData
else:
    from ...base import put_err
    from ...plot import get_palette
    from .._utils import process_label_col as process_file_labels
    from .._utils import process_num_label_col as process_peak_labels
    from ._base import HplcData
    

def plot_hplc(hplc_data: Union[HplcData, List[HplcData]],
              ax = None, fig_size = (10, 8),
              dfs_refinment_x: Dict[str, float] = {}, dfs_refinment_y: Dict[str, float] = {},
              file_labels: Union[str, Tuple[str, str, str]] = [], file_label_fn: Callable = process_file_labels,
              show_file_legend = True, file_legend_pos = 'upper right', file_legend_bbox = (1.3, 0.75),
              peak_labels: Union[str, Dict[float, Tuple[str, str]]] = {}, peak_label_fn: Callable = process_peak_labels,
              show_tag_legend = True, peak_legend_pos = 'upper right', peak_legend_bbox = (1.3, 1),
              start_search_time: float = 0, end_search_time = None, labels_eps = 0.1, min_height = 0, min_peak_width = 1,
              marker_offset = (0, 0.05), marker_size = 80,
              show_tag_text = True, tag_offset = (0.05, 0.05), tag_fontsize = 15,
              dpi = 600, line_width = 2, legend_fontsize = 15, **kwargs) -> Tuple[plt.Axes, List[plt.Artist]]:
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
        - show_tag_legend: FLAG, whether to show peak labels
        - peak_legend_pos: str, peak_legend_pos of peak legend
        - peak_legend_bbox: Tuple[float, float], bbox_to_anchor of peak legend
        - start_search_time: start of search time, in minutes
        - end_search_time: end of search time, in minutes
        - labels_eps: eps for matching peak labels
        - min_height: min height for peak detection
        - min_peak_width: min peak width for peak detection, in minutes
        - marker_offset: Tuple[float, float], offset for peak markers
        - marker_size: peak marker size
        - show_tag_text: FLAG, whether to show tag text
        - tag_offset: Tuple[float, float], offset for tag text
        - tag_fontsize: tag text font size
        - dpi: DPI, default is 600
        - line_width: line width for plot
        - legend_fontsize: legend font size

    Returns:
        - plt.Axes: ax
        - List[plt.Artist]: legend artists, for saving
    """
    names, data_dfs = [], []
    hplc_data = [hplc_data] if isinstance(hplc_data, HplcData) else hplc_data
    for data in hplc_data:
        names.append(data.get_tag())
        data_dfs.append(data.get_abs_data())
        if names[-1] in dfs_refinment_x or names[-1] in dfs_refinment_y:
            data_dfs[-1] = data_dfs[-1].copy(True)
            data_dfs[-1][data.X_HEADER] += dfs_refinment_x.get(names[-1], 0)
            data_dfs[-1][data.Y_HEADER] += dfs_refinment_y.get(names[-1], 0)
    # 若无数据则返回错误信息
    if len(data_dfs) == 0:
        return put_err('no data to plot, return None')
    # 处理文件标签
    if isinstance(file_labels, str):
        file_labels = file_label_fn(file_labels)
    if not file_labels or len(file_labels) != len(names):
        put_err(f'only {len(file_labels)} labels found, should be {len(names)} labels, use name instead')
        file_labels = file_label_fn(';'.join(names))
        if len(file_labels) == 1:
            file_labels[0][1] = 'black' # 避免使用调色板颜色的单个标签
    # 处理峰值标签
    if isinstance(peak_labels, str):
        peak_labels = peak_label_fn(peak_labels)
    peak_labels_v = np.array(list(peak_labels.keys()))
    # 绘制每个数据
    if ax is None:
        _, ax = plt.subplots(figsize = fig_size)
    ax.figure.set_dpi(dpi)
    lines, scatters, sc_labels = [], [], []
    for label, data_i, data_df_i in zip(file_labels, hplc_data, data_dfs):
        label_string, color = label
        line = ax.plot(data_df_i[data_i.X_HEADER], data_df_i[data_i.Y_HEADER],
                       color = color, label = label_string, linewidth = line_width)[0]
        lines.append(line)
        # 搜索峰值
        st = data_i.get_tick_by_minute(start_search_time) # start_search_time单位为分钟, st's unit is data tick
        ed = data_i.get_tick_by_minute(end_search_time) if end_search_time is not None else None
        peaks_idx, peak_props = scipy.signal.find_peaks(data_df_i[data_i.Y_HEADER], rel_height = 1,
                                                        prominence = min_height, width = data_i.get_tick_by_minute(min_peak_width))
        peaks_idx = peaks_idx[peaks_idx >= (st - data_i.get_tick_by_minute(dfs_refinment_x.get(data_i.get_tag(), 0)))] # addjust for offset
        if ed is not None:
            peaks_idx = peaks_idx[peaks_idx <= ed]
        peak_df = data_df_i.iloc[peaks_idx, :]
        for t, a in zip(peak_df[data_i.X_HEADER], peak_df[data_i.Y_HEADER]):                    
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
    # 设置文件标签图例
    _bbox_extra_artists = []
    if show_file_legend:
        file_legend = plt.legend(fontsize=legend_fontsize, loc = file_legend_pos,
                                bbox_to_anchor = file_legend_bbox, draggable = True)
        ax.add_artist(file_legend)
        _bbox_extra_artists.append(file_legend)
    # 设置峰值标签图例
    if scatters and show_tag_legend:
        [line.set_label(None) for line in lines]
        [sc.set_label(l) for sc, l in zip(scatters, sc_labels)]
        peak_legend = plt.legend(fontsize=legend_fontsize, loc = peak_legend_pos,
                                 bbox_to_anchor = peak_legend_bbox, draggable = True)
        ax.add_artist(peak_legend)
        _bbox_extra_artists.append(peak_legend)
    return ax, _bbox_extra_artists


__all__ = [
    'process_file_labels',
    'process_peak_labels',
    'plot_hplc',
    ]


if __name__ == '__main__':
    from mbapy.sci_instrument.hplc.waters import WatersData
    data = WatersData('data_tmp/scripts/hplc/ORI_DATA5184.arw')
    plot_hplc(data, dpi = 100)
    plt.show()