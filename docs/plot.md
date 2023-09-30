# mbapy.plot

This module provides functions for creating various types of plots, including bar plots, QQ-plots, and Turkey plots.  

## Functions

### rgb2hex(r, g, b) -> str

Converts an RGB color code to a hexadecimal color code.  

Parameters:  
- r (int): The red component of the RGB color code.  
- g (int): The green component of the RGB color code.  
- b (int): The blue component of the RGB color code.  

Returns:  
- str: The hexadecimal color code.  

Example:  
```python
rgb2hex(255, 0, 0)
```

### hex2rgb(hex:str) -> List[int]

Converts a hexadecimal color code to an RGB color code.  

Parameters:  
- hex (str): The hexadecimal color code.  

Returns:  
- List[int]: The RGB color code as a list of three integers representing the red, green, and blue components.  

Example:  
```python
hex2rgb('#FF0000')
```

### rgbs2hexs(rgbs:List[Tuple[float]]) -> List[str]

Converts a list of RGB tuples to a list of hexadecimal color codes.  

Parameters:  
- rgbs (List[Tuple[float]]): A list of RGB tuples. Each tuple must contain three floats between 0 and 1 representing the red, green, and blue components of the color.  

Returns:  
- List[str]: A list of hexadecimal color codes as strings.  

Example:  
```python
rgbs2hexs([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
```

### get_palette(n:int = 10, mode:Union[None, str] = None, return_n = True) -> List[str]

Get a sequence of hexadecimal color codes.  

Parameters:  
- n (int, optional): The number of colors required. Defaults to 10.  
- mode (Union[None, str], optional): The kind of colors. Defaults to None.  
    - 'hls': Uses `sns.color_palette('hls', n)` to generate colors.  
    - 'green': Uses a predefined list of green colors.  
    - 'pair': Uses `plt.get_cmap('tab20')` to generate colors.  
    - None: Uses `plt.get_cmap('Set1')` for n <= 9 or `plt.get_cmap('Set3')` for n <= 12.  
- return_n (bool, optional): Whether to return exactly n colors. Defaults to True.  

Returns:  
- List[str]: A list of hexadecimal color codes.  

Example:  
```python
get_palette(5, 'green')
```

### AxisLable

A class that represents a label for an axis.  

Methods:  
- __init__(self, name:str, hold_space:int = 1): Initializes the AxisLable object with the given name and hold_space.  
- add_space(self, space:int = 1): Adds space to the hold_space.  

Example:  
```python
label = AxisLable('x')
label.add_space(2)
```

### pro_hue_pos(factors:List[str], df:pd.DataFrame, width:float, bar_space:float) -> Tuple[List[List[AxisLable]], List[List[float]]]

Generate the position and labels for a grouped bar plot with multiple factors.  

Parameters:  
- factors (List[str]): A list of strings representing the factors to group the bars by.  
- df (pd.DataFrame): A pandas DataFrame containing the data for the bar plot.  
- width (float): The width of each individual bar.  
- bar_space (float): The space between each group of bars.  

Returns:  
- Tuple[List[List[AxisLable]], List[List[float]]]: A tuple containing two lists. The first list contains the labels for each factor and each bar. The second list contains the x-positions for each bar.  

Example:  
```python
xlabels, pos = pro_hue_pos(['factor1', 'factor2'], df, 0.4, 0.2)
```

### plot_bar(factors:List[str], tags:List[str], df:pd.DataFrame, **kwargs)

Stacked bar plot with hue style.  

Parameters:  
- factors (List[str]): A list of factors.  
- tags (List[str]): A list of tags.  
- df (pd.DataFrame): A pandas DataFrame.  
- **kwargs: Additional keyword arguments.  
    - width (float): The width of each individual bar. Defaults to 0.4.  
    - bar_space (float): The space between each group of bars. Defaults to 0.2.  
    - xrotations (List[int]): The rotation angle of the x-axis labels for each factor. Defaults to [0] * len(factors).  
    - colors (List[str]): The colors to use for each tag. Defaults to plt.rcParams['axes.prop_cycle'].by_key()['color'].  
    - hatchs (List[str]): The hatch patterns to use for each tag. Defaults to ['-', '+', 'x', '\\', '*', 'o', 'O', '.'].  
    - labels (None or List[str]): The labels to use for each tag. If None, the column names of the DataFrame will be used. Defaults to None.  
    - font_size (None or List[int]): The font size of the x-axis labels for each factor. If None, the default font size will be used. Defaults to None.  
    - offset (List[int]): The offset of the x-axis labels for each factor. Defaults to [(i+1)*(plt.rcParams['font.size']+8) for i in range(len(factors))].  
    - err (None or float): The error bar size. If None, no error bars will be plotted. Defaults to None.  
    - err_kwargs (dict): Additional keyword arguments for the error bars. Defaults to {'capsize':5, 'capthick':2, 'elinewidth':2, 'fmt':' k'}.  

Returns:  
- np.array: An array of positions.  
- ax1: An axis object.  

Example:  
```python
plot_bar(['factor1', 'factor2'], ['tag1', 'tag2'], df, width=0.4, bar_space=0.2)
```

### plot_positional_hue(factors:List[str], tags:List[str], df:pd.DataFrame, **kwargs)

Wrapper function for creating a plot with positional hue.  

Parameters:  
- factors (List[str]): A list of factors.  
- tags (List[str]): A list of tags.  
- df (pd.DataFrame): A pandas DataFrame.  
- **kwargs: Additional keyword arguments.  

Returns:  
- function: The core wrapper function.  

Example:  
```python
@plot_positional_hue(['factor1', 'factor2'], ['tag1', 'tag2'], df)
def plot_func(ax, x, y, label, label_idx, margs, **kwargs):  
    # do something
```

### qqplot(tags:List[str], df:pd.DataFrame, figsize = (12, 6), nrows = 1, ncols = 1, **kwargs)

Generate QQ-plots for each column in the given DataFrame.  

Parameters:  
- tags (List[str]): A list of column names to generate QQ-plots for.  
- df (pd.DataFrame): The DataFrame containing the data.  
- figsize (tuple, optional): The size of the figure. Defaults to (12, 6).  
- nrows (int, optional): The number of rows in the figure grid. Defaults to 1.  
- ncols (int, optional): The number of columns in the figure grid. Defaults to 1.  
- **kwargs: Additional keyword arguments including xlim, ylim, title, tick_size, and label_size.  

Returns:  
- None

Example:  
```python
qqplot(['column1', 'column2'], df, figsize=(12, 6), nrows=2, ncols=1)
```

### save_show(path:str, dpi = 300, bbox_inches = 'tight')

Saves the current matplotlib figure to a file at the specified path and displays the figure.  

Parameters:  
- path (str): The path where the figure will be saved.  
- dpi (int, optional): The resolution of the saved figure in dots per inch. Default is 300.  
- bbox_inches (str or Bbox, optional): The portion of the figure to save. Default is 'tight'.  

Returns:  
- None

Example:  
```python
save_show('figure.png', dpi=300, bbox_inches='tight')
```

### plot_turkey(means, std_errs, tukey_results)

Plot a bar chart showing the means of different groups along with the standard errors.  

Parameters:  
- means: A list of mean values for each group.  
- std_errs: A list of standard errors for each group.  
- tukey_results: The Tukey's test results object.  

Returns:  
- The current `Axes` instance.  

Example:  
```python
plot_turkey([1, 2, 3], [0.1, 0.2, 0.3], tukey_results)
```

## Constants

### colors

A list of default colors for plotting.  

Example:  
```python
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
```