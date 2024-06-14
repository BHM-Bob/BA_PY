# Module Overview
This Python module provides a set of functions and classes designed to handle and process scientific instrument data. It includes utilities for validating file paths, loading and processing data from raw files, and managing data associated with scientific instruments.

# Functions
## path_param_checker(path: str, suffixs: List[str] = None) -> bool
### Function Description
The `path_param_checker` function checks if the given path is valid. It returns `True` if the path is `None` or if it's a valid path with one of the specified suffixes.

### Parameters
- `path` (str): The file path to check.
- `suffixs` (List[str], optional): A list of valid file suffixes. Defaults to `None`.

### Return Value
- Returns a boolean indicating the validity of the path.

### Notes
- If `suffixs` is not provided, the function will only check if the path is `None` or a valid path.

### Example
```python
is_valid = path_param_checker('/path/to/file.txt', ['.txt', '.csv'])
```

# Classes
## SciInstrumentData
### Class Initialization
`SciInstrumentData(data_file_path: Union[None, str, List[str]] = None)`

### Members
- `data_file_path`: The path to the data file, which can be `None` or an absolute path.
- `processed_data`: The processed data, initialized to `None`.
- `tag`: A tag associated with the data, initialized to `None`.
- `processed_data_path`: The path where processed data will be saved, initialized to `None`.
- `raw_data`: The raw data loaded from the file, initialized to `None`.
- `data_df`: A pandas DataFrame containing the processed data, initialized to `None`.
- `X_HEADER`: The header for the x-axis data, initialized to `None`.
- `Y_HEADER`: The header for the y-axis data, initialized to `None`.
- `TICKS_IN_MINUTE`: The number of ticks per minute, initialized to `None`.
- `SUCCEED_LOADED`: A flag indicating if data loading was successful, initialized to `False`.

### Methods

#### check_processed_data_empty(processed_data = None) -> bool
##### Method Description
Checks if the `processed_data` or `self.processed_data` is empty.

##### Parameters
- `processed_data`: The data to check for emptiness. If `None`, `self.processed_data` is used.

##### Return Value
- Returns `True` if the data is empty, `False` otherwise.

#### load_raw_data_file(raw_data_bytes: bytes = None)
##### Method Description
Loads raw data from a file or bytes.

##### Parameters
- `raw_data_bytes`: Raw data in bytes. If `None`, data is loaded from `self.data_file_path`.

##### Return Value
- Returns the decoded raw data as a string.

#### load_raw_data_from_bytes(raw_data_bytes: bytes)
##### Method Description
An alias for `load_raw_data_file` that loads raw data from bytes.

##### Parameters
- `raw_data_bytes`: Raw data in bytes.

##### Return Value
- Returns the decoded raw data as a string.

#### load_processed_data_file(path: str = None, data_bytes: bytes = None)
##### Method Description
Loads processed data from a file or bytes.

##### Parameters
- `path`: The file path to load processed data from.
- `data_bytes`: Processed data in bytes.

##### Return Value
- Returns a pandas DataFrame containing the loaded data.

#### make_tag(tag: str = None, **kwargs)
##### Method Description
Generates a tag for the data, using the file path's stem if no tag is provided.

##### Parameters
- `tag`: The tag to set for the data.

##### Return Value
- Returns the generated tag.

#### process_raw_data(*args, **kwargs)
##### Method Description
Processes the raw data and converts it into a pandas DataFrame.

##### Return Value
- Returns the processed data as a DataFrame or an error message.

#### save_processed_data(path: str = None, *args, **kwargs)
##### Method Description
Saves the processed data to a file.

##### Parameters
- `path`: The file path to save the processed data to.

##### Return Value
- Returns the path where the data was saved.

#### get_tick_by_minute(minute: float)
##### Method Description
Returns the nearest tick to the given minute based on the `TICKS_IN_MINUTE`.

##### Parameters
- `minute`: The minute value to find the nearest tick for.

##### Return Value
- Returns the tick index.

#### get_processed_data(*args, **kwargs)
##### Method Description
Returns the processed data if available, otherwise processes the raw data.

##### Return Value
- Returns the processed data.

#### get_tag(*args, **kwargs)
##### Method Description
Returns the tag associated with the data.

##### Return Value
- Returns the tag.

# Exported Members
- `path_param_checker`
- `SciInstrumentData`