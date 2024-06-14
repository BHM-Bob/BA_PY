<!--
 * @Date: 2024-06-14 16:44:36
 * @LastEditors: BHM-Bob 2262029386@qq.com
 * @LastEditTime: 2024-06-14 16:46:31
 * @Description: 
-->
# Module Overview

This Python module is designed to support the plotting of high-performance liquid chromatography (HPLC) data. It includes functions for processing label columns, peak labels, and a comprehensive plotting function that allows for detailed visualization of HPLC data with various customization options.

# Functions

## process_file_labels(labels: str) -> List[Tuple[str, str]]
### Function Description
Converts a string of labels into a list of tuples, each containing a label and its corresponding color. This function is typically used for file labels associated with HPLC data.

### Parameters
- `labels` (str): A string representing labels, which can include color codes.

### Return Value
- Returns a list of tuples with labels and colors.

## process_peak_labels(labels: str) -> Dict[float, Tuple[str, str]]
### Function Description
Processes peak labels from a string and returns a dictionary with peak times as keys and tuples of labels and colors as values.

### Parameters
- `labels` (str): A string representing peak labels and associated colors.

### Return Value
- Returns a dictionary mapping peak times to their labels and colors.

## plot_hplc(hplc_data: Union[HplcData, List[HplcData]], **kwargs) -> Tuple[plt.Axes, List[plt.Artist], Dict[str, np.ndarray]]
### Function Description
Plots HPLC data on a matplotlib axes object, with various options for customization such as log scale, peak labeling, and legend display.

### Parameters
- `hplc_data` (Union[HplcData, List[HplcData]]): An instance or list of HPLC data instances to plot.
- `ax` (matplotlib.axes.Axes, optional): The matplotlib axes to plot on. If `None`, a new figure and axes are created.
- `fig_size` (Tuple[float, float], optional): The size of the figure in inches.
- `y_log_scale` (bool, optional): Whether to apply a logarithmic scale to the y-axis.
- `dfs_refinment_x` (Dict[str, float], optional): A dictionary for x-axis data refinement with data tags as keys.
- `dfs_refinment_y` (Dict[str, float], optional): A dictionary for y-axis data refinement with data tags as keys.
- `file_labels` (Union[str, Tuple[str, str, str]], optional): A string or tuple representing file labels.
- `file_label_fn` (Callable, optional): A function to process file labels.
- `show_file_legend` (bool, optional): Whether to show the file legend.
- `file_legend_pos` (str, optional): The position of the file legend.
- `file_legend_bbox` (Tuple[float, float], optional): The bounding box for the file legend.
- `peak_labels` (Union[str, Dict[float, Tuple[str, str]]], optional): A string or dictionary for peak labels.
- `peak_label_fn` (Callable, optional): A function to process peak labels.
- `plot_peaks_underline` (bool, optional): Whether to plot the underline of peaks.
- `plot_peaks_line` (bool, optional): Whether to plot a line at the peak.
- `plot_peaks_area` (bool, optional): Whether to plot the area under the peak.
- `peak_area_alpha` (float, optional): The transparency level for peak areas.
- `show_tag_legend` (bool, optional): Whether to show the peak label legend.
- `peak_legend_pos` (str, optional): The position of the peak legend.
- `peak_legend_bbox` (Tuple[float, float], optional): The bounding box for the peak legend.
- `start_search_time` (float, optional): The start time for peak searching.
- `end_search_time` (float, optional): The end time for peak searching.
- `labels_eps` (float, optional): The epsilon value for matching peak labels.
- `min_height` (float, optional): The minimum height for peak detection.
- `min_peak_width` (float, optional): The minimum width for peak detection.
- `marker_offset` (Tuple[float, float], optional): The offset for peak markers.
- `marker_size` (int, optional): The size of peak markers.
- `show_tag_text` (bool, optional): Whether to show tag text.
- `tag_offset` (Tuple[float, float], optional): The offset for tag text.
- `tag_fontsize` (int, optional): The font size for tag text.
- `dpi` (int, optional): The DPI for the figure.
- `line_width` (int, optional): The line width for plotting.
- `legend_fontsize` (int, optional): The font size for the legend.

### Return Value
- Returns a tuple containing the matplotlib axes object, a list of legend artists, a dictionary of peak indices for each data set, and the processed file labels.

### Notes
- If a data set has no peaks within the specified range, it will not appear in the `files_peaks_idx` return value.

# Exported Members

- `process_file_labels`
- `process_peak_labels`
- `plot_hplc`