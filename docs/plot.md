# mbapy.plot

This module provides functions for creating various types of plots, including bar plots, QQ-plots, and Turkey plots.  

### rgb2hex -> str
Converts RGB values to a hexadecimal color code.

#### Params
- r (int): The red component of the color.
- g (int): The green component of the color.
- b (int): The blue component of the color.

#### Returns
- str: The hexadecimal color code.

#### Notes
This function takes the red, green, and blue components of a color and returns the corresponding hexadecimal color code.

#### Example
```python
>>> rgb2hex(255, 0, 0)
'#FF0000'
```

### hex2rgb -> List[int]
Converts a hexadecimal color code to RGB values.

#### Params
- hex (str): The hexadecimal color code.

#### Returns
- List[int]: A list containing the red, green, and blue components of the color.

#### Notes
This function takes a hexadecimal color code and returns a list containing the red, green, and blue components of the color.

#### Example
```python
>>> hex2rgb('#FF0000')
[255, 0, 0]
```

### rgbs2hexs -> List[str]
Converts a list of RGB tuples to a list of hexadecimal color codes.

#### Params
- rgbs (List[Tuple[float]]): A list of RGB tuples, each containing three floats between 0 and 1 representing the red, green, and blue components of the color.

#### Returns
- List[str]: A list of hexadecimal color codes as strings.

#### Notes
This function takes a list of RGB tuples and converts each tuple to a corresponding hexadecimal color code.

#### Example
```python
>>> rgbs2hexs([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])
['#FF0000', '#00FF00']
```

### get_palette -> List[str]
Get a sequence of hex colors.

#### Params
- n (int): How many colors are required.
- mode (Union[None, str]): The kind of colors. Possible values are 'hls', 'green', 'pair', or None.
- return_n (bool): Whether to return exactly n colors.

#### Returns
- List[str]: A list of hexadecimal color codes.

#### Notes
This function returns a sequence of hexadecimal color codes based on the specified mode and number of colors.

#### Example
```python
>>> get_palette(5, 'hls')
['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF']
```

### AxisLable
Represents an axis label.

#### Attrs
- name (str): The name of the axis label.
- hold_space (int): The amount of space to hold.

#### Methods
- add_space(space:int): Adds space to the axis label.

#### Notes
This class represents an axis label and provides a method to add space to the label.

#### Example
```python
>>> label = AxisLable('x-axis', 1)
>>> label.add_space(2)
```

### pro_hue_pos
Generate the position and labels for a grouped bar plot with multiple factors.

#### Params
- factors (List[str]): A list of strings representing the factors to group the bars by.
- df (pd.DataFrame): A pandas DataFrame containing the data for the bar plot.
- width (float): The width of each individual bar.
- bar_space (float): The space between each group of bars.

#### Returns
- Tuple[List[List[AxisLable]], List[List[float]]]: A tuple containing two lists. The first list contains the labels for each factor and each bar. The second list contains the x-positions for each bar.

#### Notes
This function generates the position and labels for a grouped bar plot with multiple factors.

#### Example
```python
>>> labels, positions = pro_hue_pos(['factor1', 'factor2'], df, 0.4, 0.2)
```

### plot_bar
Stack bar plot with hue style.

#### Params
- factors (List[str]): A list of factors.
- tags (List[str]): A list of tags.
- df (pd.DataFrame): A pandas DataFrame.
- **kwargs: Additional keyword arguments.

#### Returns
- np.array: An array of positions.
- ax1: An axis object.

#### Notes
This function plots a stacked bar plot with hue style based on the given factors, tags, and DataFrame.

#### Example
```python
>>> positions, ax = plot_bar(['factor1', 'factor2'], ['tag1', 'tag2'], df, width=0.4, bar_space=0.2)
```

### plot_positional_hue
Wrapper function to support additional arguments for plotting positional hue.

#### Params
- factors (List[str]): A list of factors.
- tags (List[str]): A list of tags.
- df (pd.DataFrame): A pandas DataFrame.
- **kwargs: Additional keyword arguments.

#### Returns
- function: A function for plotting positional hue.

#### Notes
This function is a wrapper that supports additional arguments for plotting positional hue.

#### Example
```python
@plot_positional_hue(['factor1', 'factor2'], ['tag1', 'tag2'], df)
def plot_func(ax, x, y, label, label_idx, margs, **kwargs):
    # do something
```

### qqplot
Generate a QQ-plot for each column in the given DataFrame.

#### Params
- tags (List[str]): A list of column names to generate QQ-plots for.
- df (pd.DataFrame): The DataFrame containing the data.
- figsize (tuple, optional): The size of the figure. Defaults to (12, 6).
- nrows (int, optional): The number of rows in the figure grid. Defaults to 1.
- ncols (int, optional): The number of columns in the figure grid. Defaults to 1.
- **kwargs: Additional keyword arguments including xlim, ylim, title, tick_size, and label_size.

#### Returns
- None

#### Notes
This function generates a QQ-plot for each column in the given DataFrame and displays the plots in a grid.

#### Example
```python
>>> qqplot(['column1', 'column2'], df, figsize=(12, 6), nrows=2, ncols=1)
```

### save_show
Saves the current matplotlib figure to a file at the specified path and displays the figure.

#### Params
- path (str): The path where the figure will be saved.
- dpi (int, optional): The resolution of the saved figure in dots per inch. Default is 300.
- bbox_inches (str or Bbox, optional): The portion of the figure to save. Default is 'tight'.

#### Returns
- None

#### Notes
This function saves the current matplotlib figure to a file at the specified path and displays the figure.

#### Example
```python
>>> save_show('plot.png', dpi=300, bbox_inches='tight')
```

### plot_turkey
Plot a bar chart showing the means of different groups along with the standard errors.

#### Params
- means: A list of mean values for each group.
- std_errs: A list of standard errors for each group.
- tukey_results: The Tukey's test results object.
- min_star: The minimum number of stars to indicate significance.

#### Returns
- Axes: The current `Axes` instance.

#### Notes
This function plots a bar chart showing the means of different groups along with the standard errors. It also marks the groups with significant differences based on the Tukey's test results.

#### Example
```python
>>> plot_turkey([10, 20, 30], [1, 2, 3], tukey_results, min_star=2)
```