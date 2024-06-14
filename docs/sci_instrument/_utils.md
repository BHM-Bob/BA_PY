<!--
 * @Date: 2024-06-14 16:44:08
 * @LastEditors: BHM-Bob 2262029386@qq.com
 * @LastEditTime: 2024-06-14 16:44:18
 * @Description: 
-->
# Module Overview
This Python module is designed for processing label columns, particularly for plotting purposes. It provides functions to handle string inputs representing labels and colors, converting them into structured data that can be used for plotting with specific colors and markers.

# Functions

## process_label_col(labels: str, file_col_mode: str = 'hls') -> List
### Function Description
Converts a string of labels separated by semicolons into a list of tuples, each containing a label and its corresponding color. If a color is not provided for a label, a color is automatically assigned.

### Parameters
- `labels` (str): A string where each label is separated by a semicolon, and optionally followed by a comma and its color.
- `file_col_mode` (str, optional): The color mode used for generating automatic colors. Default is 'hls'.

### Return Value
- Returns a list of tuples, each containing a label and its color.

### Example
```python
labels = "Label1,#FF5733;Label2"
processed_labels = process_label_col(labels)
```

## process_num_label_col(labels: str, peak_col_mode: str = 'hls') -> dict
### Function Description
Processes a string of numerical labels, associating each with a color. The input string should contain numerical identifiers followed by labels, separated by semicolons.

### Parameters
- `labels` (str): A string where each pair of a numerical identifier and a label is separated by a semicolon.
- `peak_col_mode` (str, optional): The color mode for generating colors. Default is 'hls'.

### Return Value
- Returns a dictionary with numerical identifiers as keys and a list containing the label and color as values.

### Example
```python
labels = "1.0,Label1;2.5,Label2"
processed_labels = process_num_label_col(labels)
```

## process_num_label_col_marker(labels: str, peak_col_mode: str = 'hls', markers: List[str] = PLT_MARKERS) -> dict
### Function Description
Similar to `process_num_label_col`, but also assigns a marker to each numerical label based on the input string. The string format allows for an optional marker to be specified.

### Parameters
- `labels` (str): A string with numerical identifiers followed by labels and optionally a color and a marker, separated by semicolons.
- `peak_col_mode` (str, optional): The color mode for color generation. Default is 'hls'.
- `markers` (List[str], optional): A list of markers to be assigned to the labels. Default is `PLT_MARKERS`.

### Return Value
- Returns a dictionary with numerical identifiers as keys and a list containing the label, color, and marker as values.

### Example
```python
labels = "1.0,Label1,#FF5733,o;2.5,Label2,*"
processed_labels = process_num_label_col_marker(labels)
```

# Exported Members
- `process_label_col`
- `process_num_label_col`
- `process_num_label_col_marker`