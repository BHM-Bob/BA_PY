# Module Overview

The module `mbapy.sci_instrument.mass._base` provides a function `plot_mass` for visualizing mass spectrometry data using the `MassData` class. It allows for plotting mass-to-charge ratios against peak heights, with options to customize the plot's appearance and behavior.

# Function

## plot_mass
### Function Description
The `plot_mass` function creates a plot of mass spectrometry data, displaying peaks as vertical lines and optionally as scatter points with labels.

### Parameters

- `data`: A `MassData` object that contains mass spectrometry data in `data_df` and `peak_df`.
- `ax`: A matplotlib Axes object. If `None`, a new one will be created with the specified `fig_size`.
- `fig_size`: A tuple defining the size of the figure if a new Axes is created.
- `xlim`: A tuple specifying the x-axis limits. If `None`, it defaults to the minimum and maximum values of the x-axis data.
- `show_legend`: A boolean indicating whether to show a legend.
- `legend_fontsize`: Font size for the legend.
- `legend_pos`: Position for the legend.
- `legend_bbox`: Bounding box for the legend's position.
- `min_height`: The minimum height for peaks to be considered. If `None`, it defaults to the minimum value in `peak_df`.
- `min_height_percent`: A percentage of the maximum peak height to use as a minimum height filter.
- `verbose`: A flag to print x-axis data limits.
- `color`: Color for peak lines and scatter points.
- `labels_eps`: A tolerance value for matching labels to peak data.
- `labels`: A dictionary mapping mass values to tuples containing a label, text color, and marker.
- `tag_fontsize`: Font size for peak tags.
- `marker_size`: Size for peak markers.
- `normal_marker`: Marker style for peaks without a specified label.
- `is_y_log`: A boolean to determine if the y-axis should be in logarithmic scale.
- `**kwargs`: Additional keyword arguments for `matplotlib.pyplot.Axes.vlines()`.

### Returns
- `ax`: The matplotlib Axes object with the plot.
- `_bbox_extra_artists`: A list of additional artists used in the plot, such as legends.

## process_peak_labels
This function is imported from `mbapy.sci_instrument._utils` and is used for processing peak labels, but it's not detailed in the provided code snippet.

# Exported Members

- `plot_mass`
- `process_peak_labels`

# Example Usage

The function `plot_mass` can be used to plot mass spectrometry data as follows:

```python
# Assuming 'mass_data' is an instance of MassData with loaded data
ax, artists = plot_mass(mass_data)
plt.show()
```

This will create a plot with peaks represented as vertical lines and optionally with labels if a `labels` dictionary is provided.

# Notes

- The function `plot_mass` is designed to work with objects of the `MassData` class, which should have `data_df` and `peak_df` attributes populated with mass spectrometry data.
- The function provides a high degree of customization for plotting, including options for labeling peaks, setting axis scales, and adjusting the plot's visual style.
- The `plot_mass` function can handle both linear and logarithmic scales for the y-axis based on the `is_y_log` parameter.
- The `labels` parameter allows users to specify custom labels and markers for specific mass values, enhancing the plot's informational content.