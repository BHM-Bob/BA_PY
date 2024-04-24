# mbapy.plot
This Python module provides a collection of functions and classes for data visualization and statistical analysis. It includes utilities for plotting bar charts, scatter plots, regression lines, and swarm plots, as well as functions for converting color representations, calculating positions for swarm plots, generating QQ-plots, and applying Tukey's test for multiple comparisons.  

*kimi generated*

# Functions

## rgb2hex(r, g, b) -> str
### Function Overview
Converts RGB values to a hexadecimal color code.

### Parameters
- **r (int)**: Red component of the color (0-255).
- **g (int)**: Green component of the color (0-255).
- **b (int)**: Blue component of the color (0-255).

### Returns
- **str**: Hexadecimal color code prefixed with '#'.

### Examples
```python
print(rgb2hex(255, 165, 0))  # Output: '#FFA500'
```

## hex2rgb(hex: str) -> List[int]
### Function Overview
Converts a hexadecimal color code to RGB values.

### Parameters
- **hex (str)**: Hexadecimal color code as a string.

### Returns
- **List[int]**: List containing the red, green, and blue components of the color.

### Examples
```python
print(hex2rgb('#FFA500'))  # Output: [255, 165, 0]
```

## rgbs2hexs(rgbs: List[Tuple[float]]) -> List[str]
### Function Overview
Converts a list of RGB tuples to a list of hexadecimal color codes.

### Parameters
- **rgbs (List[Tuple[float]])**: List of RGB tuples where each tuple contains three floats between 0 and 1.

### Returns
- **List[str]**: List of hexadecimal color codes as strings.

### Examples
```python
print(rgbs2hexs([(255, 0, 0), (0, 255, 0)]))  # Output: ['#FF0000', '#00FF00']
```

## get_palette(n: int = 10, mode: Union[None, str] = None, return_n: bool = True) -> List[str]
### Function Overview
Generates a sequence of hexadecimal color codes based on the specified number and mode.

### Parameters
- **n (int)**: Number of colors required.
- **mode (Union[None, str])**: Kind of colors to generate ('hls', 'green', 'pair', or None).
- **return_n (bool)**: If True, returns only the first `n` colors from the generated palette.

### Returns
- **List[str]**: List of hexadecimal color codes.

### Examples
```python
print(get_palette(n=5, mode='green'))  # Output: ['#80ab1c', '#405535', '#99b69b', '#92e4ce', '#72cb87']
```

## calcu_swarm_pos(x: float, y: np.ndarray, width: float, d: Optional[float] = None) -> np.ndarray
### Function Overview
Calculates the x-coordinates for data points in a swarm plot.

### Parameters
- **x (float)**: The x-coordinate of the center of the swarm.
- **y (np.ndarray)**: The y-coordinates of the data points.
- **width (float)**: The width of the swarm.
- **d (Optional[float])**: The distance between the data points. If None, it will be calculated.

### Returns
- **np.ndarray**: An array of x-coordinates for the data points.

### Examples
```python
# Assuming y contains the y-coordinates of data points
# and we want to calculate the corresponding x-coordinates for a swarm plot
x_positions = calcu_swarm_pos(0, y, 10)
```

## qqplot(tags: List[str], df: pd.DataFrame, figsize: Tuple[int, int] = (12, 6), nrows: int = 1, ncols: int = 1, **kwargs)
### Function Overview
Generates a QQ-plot for each column in the given DataFrame.

### Parameters
- **tags (List[str])**: List of column names to generate QQ-plots for.
- **df (pd.DataFrame)**: DataFrame containing the data.
- **sizeof (Tuple[int, int])**: Size of the figure.
- **nrows (int)**: Number of rows in the figure grid.
- **ncols (int)**: Number of columns in the figure grid.
- **kwargs**: Additional keyword arguments for customization.

### Returns
- **List[matplotlib.axes._subplots.Axes]**: List of Axes objects for the QQ-plots.

### Examples
```python
qqplot(['column1', 'column2'], data_frame, figsize=(8, 4))
```

## save_show(path: str, dpi: int = 300, bbox_inches: Union[str, Bbox] = 'tight')
### Function Overview
Saves the current matplotlib figure to a file and displays it.

### Parameters
- **path (str)**: Path where the figure will be saved.
- **dpi (int)**: Resolution of the saved figure in dots per inch.
- **bbox_inches (Union[str, Bbox])**: Portion of the figure to save.

### Returns
- **None**

### Examples
```python
save_show('path/to/save/figure.png', dpi=300)
```

## plot_stats_star(x1: float, x2: float, h: float, endpoint: float, p_value: float, ax: matplotlib.axes._subplots.Axes = plt, p2star: Callable[[float], str] = p_value_to_stars)
### Function Overview
Plots a horizontal line with endpoints and a significance star on a given Axes.

### Parameters
- **x1 (float)**: X-coordinate of the left endpoint.
- **x2 (float)**: X-coordinate of the right endpoint.
- **h (float)**: Y-coordinate of the horizontal line.
- **endpoint (float)**: Height of the endpoint.
- **p_value (float)**: P-value of the difference.
- **ax (matplotlib.axes._subplots.Axes)**: Axes instance to plot on.
- **p2star (Callable[[float], str])**: Function to convert p-value to stars.

### Returns
- **None**

### Examples
```python
plot_stats_star(1, 2, 0.5, 0.1, 0.05)
```

## plot_turkey(means: List[float], std_errs: List[float], tukey_results: Any, min_star: int = 1)
### Function Overview
Plots a bar chart with means and standard errors, marking significant differences based on Tukey's test results.

### Parameters
- **means (List[float])**: List of mean values for each group.
- **std_errs (List[float])**: List of standard errors for each group.
- **tukey_results (Any)**: Tukey's test results object.
- **min_star (int)**: Minimum number of stars to display.

### Returns
- **matplotlib.axes._subplots.Axes**: The current Axes instance.

### Examples
```python
plot_turkey([0.5, 1.2, 0.8], [0.1, 0.2, 0.15], tukey_test_results)
```