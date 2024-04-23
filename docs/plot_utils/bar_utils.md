<!--
 * @Date: 2024-04-23 21:42:58
 * @LastEditors: BHM-Bob 2262029386@qq.com
 * @LastEditTime: 2024-04-23 22:04:46
 * @Description: 
-->
# mbapy.plot_utils.bar_utils
This module provides a set of functions for creating complex bar plots with multiple levels of grouping (hues) and positional factors. It includes utilities for managing axis labels, calculating bar positions, and plotting bar charts with options for jitter, error bars, and hatch patterns.

# Functions

## pro_hue_pos(factors: List[str], df: pd.DataFrame, width: float, hue_space: float, bar_space: float)
### Function Overview
Generates position and label information for a grouped bar plot with multiple factors.

### Parameters
- **factors (List[str])**: A list of strings representing the factors to group the bars by.
- **df (pd.DataFrame)**: A DataFrame containing the data for the bar plot.
- **width (float)**: The width of each individual bar.
- **hue_space (float)**: The space between each group of bars.
- **bar_space (float)**: The space between each bar in a group.

### Returns
A tuple containing two lists: the first with the labels for each factor and each bar, and the second with the x-positions for each bar.

### Usage
This function is used to prepare data for plotting functions like `plot_bar`.

## plot_bar(factors: List[str], tags: List[str], df: pd.DataFrame, **kwargs)
# Function: plot_bar
The `plot_bar` function is a comprehensive tool for creating stacked bar plots with multiple levels of grouping, often referred to as 'hues'. It is designed to handle complex data structures where each bar can represent a category, and within that category, there can be further细分 (subdivisions) based on other factors.

### Parameters:

#### Essential Parameters:
- **factors (List[str])**: A list of strings where each string is a column name from the DataFrame `df` that represents a factor by which the bars will be grouped. The order of these factors determines the level of grouping, with the first factor being the most detailed (lowest level on the x-axis).
- **tags (List[str])**: A list of strings corresponding to the column names in `df` that contain the values for the bars to be plotted. Each tag represents a different set of bars that will be stacked within each group defined by the factors.
- **df (pd.DataFrame)**: A pandas DataFrame that must contain the columns specified in `factors` and `tags`. The DataFrame should be preprocessed by functions like `pro_bar_data` or `sort_df_factors` to ensure the data is structured correctly for the plot.

#### Keyword Arguments (kwargs):
##### figsize
- **figsize (Tuple[float, float])**: A tuple specifying the size of the figure. Defaults to (8, 6).
- **Default**: (8, 6)
- **Behavior**: Sets the size of the entire figure. Larger values can provide more space for intricate plots or when the plot is expected to be printed or viewed at a higher resolution.

##### dpi
- **dpi (int)**: The resolution of the figure in dots per inch. Defaults to 100.
- **Default**: 100
- **Behavior**: Adjusts the dots per inch, which affects the sharpness and file size of the saved figure. Higher DPI values make the plot more detailed but also increase the file size.

##### jitter
- **jitter (bool)**: If True, the function will add jitter to the x positions of the bars. This can be useful to show distributions within categories. Defaults to False.
- **Default**: False
- **Behavior**: When set to True, adds randomness to the x position of each bar, which can help in visualizing the distribution within categories, especially when there are many bars that would otherwise overlap.

##### jitter_kwargs
- **jitter_kwargs (dict)**: A dictionary of arguments to customize the jitter plot. It can include 'size', 'alpha', and 'color'.
- **Default**: {'size': 10, 'alpha': 0.5, 'color': None}
- **Behavior**: Customizes the appearance of the jitter plot. The 'size' parameter controls the magnitude of the jitter, 'alpha' sets the transparency, and 'color' allows specifying the color of the jitter points.

##### width
- **width (float)**: The width of each individual bar. Defaults to 0.4.
- **Default**: 0.4
- **Behavior**: Determines the width of each bar. Wider bars can make the plot bolder but may also lead to overlapping if there are too many bars in a group.

##### hue_space
- **hue_space (float)**: The space between groups of bars that are differentiated by the factors. Defaults to 0.2.
- **Default**: 0.2
- **Behavior**: Determines the space between groups of bars that are differentiated by the factors. Increasing this value can make the plot more visually appealing and easier to read.

##### bar_space
- **bar_space (float)**: The space between each bar within a group. Defaults to 0.05.
- **Default**: 0.05
- **Behavior**: Determines the space between each bar within a group. Increasing this value can make the plot more visually appealing and easier to read.

##### xrotations
- **xrotations (List[int])**: A list of angles in degrees to rotate the x-axis tick labels. Each angle corresponds to a level of factor. The length of this list should match the number of factors plus one for the top-level x-axis label.
- **Default**: [0]*len(factors)
- **Behavior**: Each value in the list corresponds to the rotation angle for the x-axis labels at each factor level. Rotation can help when x-axis labels are long or when they overlap.

##### colors
- **colors (List[str])**: A list of colors to use for the bars. Each color corresponds to a tag.
- **Default**: plt.rcParams['axes.prop_cycle'].by_key()['color']
- **Behavior**: A list of colors for the bars corresponding to each tag. Different colors can distinguish between the bars clearly.

##### hatchs
- **hatchs (Optional[List[str]])**: A list of hatch patterns to use for the bars. If provided, it should match the length of `tags`.
- **Default**: None
- **Behavior**: If provided, adds a hatch pattern to the bars for visual distinction. Each tag can have a different hatch pattern, enhancing the plot's information density.

##### font_size
- **font_size (Optional[List[int]])**: A list of font sizes for the x-axis labels of each level and the y-axis label.
- **Default**: None
- **Behavior**: Allows setting custom font sizes for the x-axis labels of each level and the y-axis label. This can be important for readability, especially in plots with many bars or small bars.

##### labels
- **labels (Optional[List[str]])**: A list of labels to use for the bars. Each label corresponds to a tag.
- **Default**: None
- **Behavior**: Customizes the legend labels for each set of bars. This can be useful when the column names are not descriptive enough or when a more simplified label is preferred.

##### ylabel
- **ylabel (Optional[str])**: The label for the y-axis. If not provided, the first tag is used.
- **Default**: The first tag name if not provided
- **Behavior**: Sets the label for the y-axis. This can be important for clarifying the unit or nature of the data being plotted.

##### offset
- **offset (List[float])**: A list of offsets for the x-axis labels of each level. The first value is ignored as it is automatically set.
- **Default**: [None] + [(i+1)*(plt.rcParams['font.size']+8) for i in range(len(factors))]
- **Behavior**: Offsets the x-axis labels of each factor level to prevent overlap and拥挤 (crowding). Adjusting these values can help in fine-tuning the plot's appearance.

##### xticks_pad
- **xticks_pad (List[float])**: A list of padding values for the x-axis tick labels of each level.
- **Default**: [5 for _ in range(len(factors)+1)]
- **Behavior**: Sets the padding between the x-axis and its tick labels. Increasing the padding can create more space between the labels and the axis.

##### edgecolor
- **edgecolor (List[str])**: A list of edge colors for the bars. Each color corresponds to a tag.
- **Default**: edgecolor=['white'] * len(tags)
- **Behavior**: `edgecolor` sets the color of the bar borders. Borders can make the bars stand out more, especially when plotted on a colored background.

##### linewidth
- **linewidth (float)**: The line width for the bar outlines. Defaults to 0.
- **Default**: 0
- **Behavior**: Sets the thickness of the bar outlines. Increasing this value can make the plot more visually appealing and easier to read.

##### err
- **err (Union[str, np.array])**: Either a string representing a column name in `df` that contains the error values, or a numpy array of error values. If provided, error bars will be added to the plot.
- **Default**: None
- **Behavior**: If provided, adds error bars to the plot. This parameter can be a string referencing a column in `df` that contains the error values or a numpy array of values directly.

##### err_kwargs
- **err_kwargs (dict)**: A dictionary of arguments to customize the error bars, such as 'capsize', 'capthick', 'elinewidth', 'fmt', and 'ecolor'.
- **Default**: {'capsize': 5, 'capthick': 2, 'elinewidth': 2, 'fmt': ' k', 'ecolor': 'black'}
- **Behavior**: Customizes the appearance of the error bars, including the size of the caps, thickness of the error lines, and the style of the error markers.

### Returns:
The function returns an array of positions for the bars, an axis object, and optionally the DataFrame with jitter data if the `jitter` parameter was set to True.

### Notes:
- The `plot_bar` function is versatile and allows for a high degree of customization to suit various data visualization needs.
- Proper use of `hue_space`, `bar_space`, and `offset` can greatly enhance the plot's clarity, especially when dealing with multiple bars and factors.
- The `jitter` parameter can be particularly useful for datasets where individual data points are of interest within each bar group.
- The `colors`, `hatchs`, and `labels` parameters offer flexibility in distinguishing between different sets of bars and can be used to represent different categories or subcategories effectively.
- The `err` parameter is a powerful tool for conveying the uncertainty or variability in the data, which is crucial for many analytical and scientific plots.
- Careful consideration of the `figsize`, `font_size`, and `xrotations` can make the plot more accessible and understandable to the audience, especially when the plot is included in a report or presentation with space constraints.

### Usage:
The `plot_bar` function is highly customizable and allows for the creation of complex bar plots that can effectively communicate multi-level grouped data. It is particularly useful in scenarios where data is nested within categories and there is a need to visualize both the category-level aggregates and the nested subdivisions.

### Example of `plot_bar` usage with `kwargs`:
```python
plot_bar(['Category', 'Subcategory'], ['Value1', 'Value2'], data_frame,
         figsize=(10, 8), colors=['blue', 'red'], hatchs=['--', '++++'],
         xrotations=[45, 0], font_size=[12, 10, 14], ylabel='Values',
         offset=[0, 10, 20], xticks_pad=[5, 3, 2], edgecolor=['black', 'none'],
         linewidth=1, err='Std_Error', err_kwargs={'capsize': 3, 'elinewidth': 1})
```

In this example, a bar plot is created with two levels of grouping ('Category' and 'Subcategory') and two sets of bars ('Value1' and 'Value2'). The bars will have different colors and hatch patterns, rotated x-axis labels with specified padding, and error bars with customized appearance.

## plot_positional_hue(factors: List[str], tags: List[str], df: pd.DataFrame, **kwargs)
### Function Overview
A wrapper function that supports creating a positional hue plot with a custom core plotting function.

### Parameters
- **factors (List[str])**: A list of factors for grouping the bars.
- **tags (List[str])**: A list of tags for the bars to be plotted.
- **df (pd.DataFrame)**: A DataFrame containing the data for plotting.
- **kwargs**: Additional keyword arguments for customization.

### Returns
A function that, when called with additional arguments, will create the plot.

### Usage
This function allows for customization of the plotting behavior through the `core_plot_func` parameter.

# Classes

## AxisLable
### Class Overview
The `AxisLable` class is used to represent labels on an axis, with the ability to hold space for bars and maintain a hierarchy among labels.
- **__init__(name: str, hold_space: int = 1)**: Initializes an `AxisLable` instance.
- **__eq__(other: AxisLable)**: Compares two `AxisLable` instances for equality.
- **__hash__**: Returns the hash value of the `AxisLable` instance.
- **add_space(space: int)**: Adds space to the label.
- **add_father(father: AxisLable)**: Adds a father to the label's hierarchy.

### Initialization Method
- **AxisLable(name: str, hold_space: int = 1)**: Initializes an `AxisLable` instance with a name and the amount of space it holds.

### Methods
- **add_space(space: int = 1)**: Increases the hold space by the given amount.
- **add_father(father: AxisLable)**: Sets a father label for the current instance and adds the current instance to the father's children set.

### Usage
This class is used internally to manage the hierarchy and positioning of labels in a grouped bar plot.
