<!--
 * @Date: 2024-04-23 21:42:51
 * @LastEditors: BHM-Bob 2262029386@qq.com
 * @LastEditTime: 2024-04-23 21:45:21
 * @Description: 
-->
# mbapy.plot_utils.scatter_utils
This module provides functions for plotting regression lines and scatter plots, with options for customization and additional features like confidence intervals and color mapping. It also includes a function to add a detailed legend to the plots, which can handle both the main plot elements and regression lines.  

*kimi generated*

# Functions

## plot_reg(X: str, Y: str, df: pd.DataFrame, ax: plt.Axes = None, figsize: Tuple[float, float] = (11, 6), method: str = 'line', ci: int = 95, order: int = 1, color: str = 'black', linewidth: float = 2) -> dict
### Function Overview
Performs regression analysis and plots the regression line on the given DataFrame's data.

### Parameters
- **X (str)**: The column name in `df` for the X-axis.
- **Y (str)**: The column name in `df` for the Y-axis.
- **df (pd.DataFrame)**: The DataFrame containing the data for plotting.
- **ax (plt.Axes)**: The Axes object to plot on. If None, a new Axes will be created.
- *(figsize)**: The size of the figure as a tuple (width, height).
- **method (str)**: The regression method to use ('line', 'quad', or 'sns').
- **ci (int)**: The confidence interval for the regression line when using the 'sns' method.
- **order (int)**: The order of the polynomial for quadratic regression.
- **color (str)**: The color of the regression line.
- **linewidth (float)**: The width of the regression line.

### Returns
A dictionary containing the Axes object, regression result, and the Line2D object of the regression line.

### Examples
```python
plot_reg('x_column', 'y_column', data_frame, method='sns', order=2)
```

## plot_scatter(X: str, Y: str, df: pd.DataFrame, label: str = None, size: Union[str, np.ndarray] = None, color: Union[str, np.ndarray] = 'black', cmap: plt.Colormap = None, cmap_label: str = None, cmap_lim: Tuple[float, float] = (None, None), cmap_tick_fn: Callable = lambda x: f'{x:.1f}', cmap_anchor: Tuple[float, float] = (0.0, 0.5), alpha: float = 1, marker: str = 'o', ax: plt.Axes = None, figsize: Tuple[float, float] = (10, 6), ticks_fontsize: int = 14, label_fontsize: int = 15, axis_fontsize: int = 20, reg: bool = False, reg_kwargs: dict = None) -> dict
### Function Overview
Plots a scatter plot with options for color mapping and regression line plotting.

### Parameters
- **X (str)**: The column name in `df` for the X-axis.
- **Y (str)**: The column name in `df` for the Y-axis.
- **df (pd.DataFrame)**: The DataFrame containing the data for plotting.
- **label (str)**: The label for the scatter plot.
- **size (Union[str, np.ndarray])**: The size of the markers in the scatter plot. Can be a column name or a numpy array.
- **color (Union[str, np.ndarray])**: The color of the markers. Can be a column name or a numpy array.
- **cmap (plt.Colormap)**: The colormap to use for the scatter plot.
- **cmap_label (str)**: The label for the colormap.
- **cmap_lim (Tuple[float, float])**: The limits for the colormap.
- **cmap_tick_fn (Callable)**: A function to format the colormap's tick labels.
- **cmap_anchor (Tuple[float, float])**: The anchor point for the colorbar.
- **alpha (float)**: The transparency of the markers.
- **marker (str)**: The style of the markers.
- **ax (plt.Axes)**: The Axes object to plot on. If None, a new Axes will be created.
- *(figsize)**: The size of the figure as a tuple (width, height).
- **ticks_fontsize (int)**: The font size for the ticks.
- **label_fontsize (int)**: The font size for the labels.
- **axis_fontsize (int)**: The font size for the axis titles.
- **reg (bool)**: Whether to plot a regression line.
- **reg_kwargs (dict)**: Keyword arguments for the `plot_reg` function.

### Returns
A dictionary containing the Axes object, the scatter plot PathCollection, a Figure (if a new one was created), the regression Line2D (if plotted), the regression result (if calculated), and the Colorbar (if a colormap was used).

### Examples
```python
plot_scatter('x_column', 'y_column', data_frame, size='size_column', color='color_column')
```

## add_scatter_legend(scatters: Union[PathCollection, List[PathCollection]], reg_lines: Union[Line2D, List[Line2D]], reg_results: Union[dict, List[dict]], ax: plt.Axes, main_h_kwgs: dict = None, reg_h_kwgs: dict = None, main_kwgs: dict = None, reg_kwgs: dict = None)
### Function Overview
Adds a detailed legend to a scatter plot, handling both the scatter plot elements and the regression lines.

### Parameters
- **scatters (Union[PathCollection, List[PathCollection]])**: The scatter plot elements to add to the legend.
- **reg_lines (Union[Line2D, List[Line2D]])**: The regression lines to add to the legend.
- **reg_results (Union[dict, List[dict]])**: The regression results used to create labels for the regression lines.
- **ax (plt.Axes)**: The Axes object to add the legend to.
- **main_h_kwgs (dict)**: Keyword arguments for the main legend handler.
- **reg_h_kwgs (dict)**: Keyword arguments for the regression legend handler.
- **main_kwgs (dict)**: Keyword arguments for the main legend.
- **reg_kwgs (dict)**: Keyword arguments for the regression legend.

### Returns
The Axes object with the added legends, as well as the main legend and regression legend objects.

### Examples
```python
add_scatter_legend(scatter_plot_elements, regression_lines, regression_results, current_axes)
```
