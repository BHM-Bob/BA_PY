<!--
 * @Date: 2024-04-23 21:42:58
 * @LastEditors: BHM-Bob 2262029386@qq.com
 * @LastEditTime: 2024-04-23 21:48:04
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
### Function Overview
Creates a stacked bar plot with multiple levels of grouping (hues).

### Parameters
- **factors (List[str])**: A list of factors for grouping the bars.
- **tags (List[str])**: A list of tags for the bars to be plotted.
- **df (pd.DataFrame)**: A DataFrame containing the data for plotting.
- **kwargs**: Additional keyword arguments for customization.

### Returns
An array of positions, an axis object, and optionally the processed DataFrame if jitter is used.

### Usage
This function is a primary interface for plotting complex bar plots with the module.

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
