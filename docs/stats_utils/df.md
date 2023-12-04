# mbapy.stats.df

This module provides utility functions for working with pandas DataFrames.  

## Functions

### get_value(df: pd.DataFrame, column: str, mask: np.array) -> list

Get the values of a specific column in a DataFrame based on a boolean mask.  

#### Params
- df (pd.DataFrame): The input DataFrame.  
- column (str): The name of the column.  
- mask (np.array): The boolean mask to filter the DataFrame.  

#### Returns
- list: The values of the specified column that satisfy the mask.  

#### Example
```python
df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [6, 7, 8, 9, 10]})
mask = np.array([True, False, True, False, True])
get_value(df, 'A', mask)  # Output: [1, 3, 5]
```

### pro_bar_data(factors: List[str], tags: List[str], df: pd.DataFrame, **kwargs) -> pd.DataFrame

Calculate the mean, standard error, and count for each combination of factors in a DataFrame.  

#### Params
- factors (List[str]): The names of the columns representing the factors.  
- tags (List[str]): The names of the columns to calculate the statistics for.  
- df (pd.DataFrame): The input DataFrame.  
- kwargs (optional): Additional keyword arguments.  
    - min_sample_N (int): The minimum number of samples required for a combination to be included in the output. Defaults to 1.  

#### Returns
- pd.DataFrame: A DataFrame containing the calculated statistics for each combination of factors.  

Notes:  
- The output DataFrame will have the same columns as the input DataFrame, with the addition of columns for the mean, standard error, and count of each tag.  

#### Example
```python
df = pd.DataFrame({'factor1': ['A', 'A', 'B', 'B'], 'factor2': ['X', 'Y', 'X', 'Y'], 'y1': [1, 2, 3, 4], 'y2': [5, 6, 7, 8]})
pro_bar_data(['factor1', 'factor2'], ['y1', 'y2'], df)
```

### pro_bar_data_R(factors: List[str], tags: List[str], df: pd.DataFrame, suffixs: List[str], **kwargs) -> Callable

A decorator that wraps a function to be applied to each combination of factors in a DataFrame.  

#### Params
- factors (List[str]): The names of the columns representing the factors.  
- tags (List[str]): The names of the columns to apply the function to.  
- df (pd.DataFrame): The input DataFrame.  
- suffixs (List[str]): The suffixes to append to the tags in the output DataFrame.  
- kwargs (optional): Additional keyword arguments.  

#### Returns
- Callable: The wrapped function.  

Notes:  
- The wrapped function should take a single argument, which is a numpy array of values for a specific combination of factors.  
- The wrapped function should return a list of values, with the length equal to the number of suffixes.  

#### Example
```python
@pro_bar_data_R(['factor1', 'factor2'], ['y1', 'y2'], df, ['_mean', '_SE'])
def calc_stats(values):  
    return [np.mean(values), np.std(values, ddof=1)/np.sqrt(len(values))]

calc_stats(df.loc[(df['factor1'] == 'A') & (df['factor2'] == 'X'), ['y1', 'y2']].values)
```

### get_df_data(factors: Dict[str, List[str]], tags: List[str], df: pd.DataFrame, include_factors: bool = True) -> pd.DataFrame

Return a subset of the input DataFrame, filtered by the given factors and tags.  

#### Params
- factors (Dict[str, List[str]]): A dictionary containing the factors to filter by. The keys are column names in the DataFrame and the values are lists of values to filter by in that column.  
- tags (List[str]): A list of column names to include in the output DataFrame.  
- df (pd.DataFrame): The input DataFrame to filter.  
- include_factors (bool, optional): Whether to include the factors in the output DataFrame. Defaults to True.  

#### Returns
- pd.DataFrame: A subset of the input DataFrame, filtered by the given factors and tags.  

#### Example
```python
df = pd.DataFrame({'factor1': ['A', 'A', 'B', 'B'], 'factor2': ['X', 'Y', 'X', 'Y'], 'y1': [1, 2, 3, 4], 'y2': [5, 6, 7, 8]})
get_df_data({'factor1': ['A'], 'factor2': ['X']}, ['y1', 'y2'], df)
```

### sort_df_factors(factors: List[str], tags: List[str], df: pd.DataFrame) -> pd.DataFrame

Sort each combination of factors in a DataFrame.  

#### Params
- factors (List[str]): The names of the columns representing the factors.  
- tags (List[str]): The names of the columns to include in the output DataFrame.  
- df (pd.DataFrame): The input DataFrame.  

#### Returns
- pd.DataFrame: The sorted DataFrame.  

#### Example
```python
df = pd.DataFrame({'factor1': ['A', 'A', 'B', 'B'], 'factor2': ['X', 'Y', 'X', 'Y'], 'y1': [1, 2, 3, 4], 'y2': [5, 6, 7, 8]})
sort_df_factors(['factor2', 'factor1'], ['y1', 'y2'], df)
```

### remove_simi(tag: str, df: pd.DataFrame, sh: float = 1., backend: str = 'numpy-array', tensor = None, device = 'cuda') -> Tuple[pd.DataFrame, List[int]]

Remove similar values from a column in a DataFrame.  

#### Params
- tag (str): The name of the column to remove similar values from.  
- df (pd.DataFrame): The input DataFrame.  
- sh (float, optional): The threshold for similarity. Values with a difference less than or equal to this threshold will be considered similar. Defaults to 1.  
- backend (str, optional): The backend to use for the computation. Supported backends are 'numpy-mat', 'numpy-array', 'torch-array', and 'ba-cpp'. Defaults to 'numpy-array'.  
- tensor (optional): The tensor to use for the computation if the backend is 'torch-array'. Defaults to None.  
- device (str, optional): The device to use for the computation if the backend is 'torch-array'. Defaults to 'cuda'.  

#### Returns
- Tuple[pd.DataFrame, List[int]]: A tuple containing the modified DataFrame and a list of indices of the removed values.  

#### Example
```python
df = pd.DataFrame({'d': [1, 2, 3, 3, 5, 6, 8, 13]})
remove_simi('d', df, 2.1, 'numpy-array')
```

### interp(long_one: pd.Series, short_one: pd.Series) -> np.ndarray

Interpolate a short pandas Series to have the same length as a long pandas Series.  

#### Params
- long_one (pd.Series): The long pandas Series.  
- short_one (pd.Series): The short pandas Series.  

#### Returns
- np.ndarray: The interpolated short pandas Series.  

#### Example
```python
long_one = pd.Series([1, 2, 3, 4, 5])
short_one = pd.Series([1, 3, 5])
interp(long_one, short_one)  # Output: array([1., 2., 3., 4., 5.])
```

### merge_col2row(df: pd.DataFrame, cols: List[str], new_cols_name: str, value_name: str) -> pd.DataFrame

Merge columns in a DataFrame to rows.  

#### Params
- df (pd.DataFrame): The input DataFrame.  
- cols (List[str]): The names of the columns to merge.  
- new_cols_name (str): The name of the new column that will contain the column names.  
- value_name (str): The name of the new column that will contain the values.  

#### Returns
- pd.DataFrame: The modified DataFrame.  

#### Example
```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
merge_col2row(df, ['A', 'B'], 'new_col', 'value_col')
```

### make_three_line_table -> pd.DataFrame
**This function creates a three-line table from the input data frame, with specified factors and tags.**

#### Params
- factors: List of strings representing the factors to be included in the table.
- tags: List of strings representing the tags to be included in the table.
- df: Input pandas DataFrame containing the data.
- float_fmt: String representing the format for floating point numbers (default is '.3f').
- t_samples: Integer representing the threshold for the number of samples (default is 30).

#### Returns
- ndf: Pandas DataFrame containing the three-line table.

#### Notes
- The function calculates the three-line table using the input factors and tags, and the provided data frame.
- It applies formatting to the floating point numbers based on the specified float format.
- It uses a threshold for the number of samples to determine the confidence interval.

#### Example
```python
import pandas as pd
from typing import List

# Create sample data
data = {
    'factor1': [1, 2, 3, 4],
    'factor2': [5, 6, 7, 8],
    'tag1': [0.1, 0.2, 0.3, 0.4],
    'tag2': [0.5, 0.6, 0.7, 0.8],
    'tag1_SE': [0.01, 0.02, 0.03, 0.04],
    'tag2_SE': [0.05, 0.06, 0.07, 0.08],
    'tag1_N': [20, 25, 30, 35],
    'tag2_N': [40, 45, 50, 55]
}
df = pd.DataFrame(data)

factors = ['factor1', 'factor2']
tags = ['tag1', 'tag2']

# Create three-line table
result = make_three_line_table(factors, tags, df)
print(result)
```


## Notes

- The functions in this module are designed to work with pandas DataFrames.  
- Some functions have optional parameters that allow for customization of the behavior.  
- The examples provided demonstrate the usage of each function.