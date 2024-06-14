<!--
 * @Date: 2024-06-14 16:50:03
 * @LastEditors: BHM-Bob 2262029386@qq.com
 * @LastEditTime: 2024-06-14 16:51:00
 * @Description: 
-->
# Module Overview

This module provides a class `WatersData` that extends the functionality of the `HplcData` class to handle Waters chromatography data files with `.arw` extension. It includes methods for loading, processing, and saving the data, as well as generating tags for the data files.

# Class

## WatersData
### Class Initialization
`WatersData(data_file_path: str = None)`
- Inherits from `HplcData`.
- Initializes with a data file path and sets specific headers for the x-axis ('Time') and y-axis ('Absorbance').
- Sets `TICKS_IN_MINUTE` to 60, indicating the data's time resolution.
- Loads and processes the raw data upon instantiation if the file path is valid.

### Members
- Inherits all members from `HplcData`.
- Adds `info_df` and `data_df` as pandas DataFrames to store the information and data parts of the file, respectively.

### Methods

#### load_processed_data_file(path: str = None, data_bytes: bytes = None)
Loads processed data from a file or bytes into `info_df` and `data_df`.

#### make_tag(tag: str = None, tags: List[str] = ['"样品名称"', '"采集日期"', '"通道"'], join_str: str = '_')
Generates a tag for the data file using specified information fields, joined by a string.

#### process_raw_data(*args, **kwargs)
Processes the raw data by splitting lines and creating DataFrames for information and data.

#### get_abs_data(*args, **kwargs)
Returns the processed data, either from existing data or by processing the raw data.

#### save_processed_data(path: str = None, *args, **kwargs)
Saves the processed data to an Excel file with 'Info' and 'Data' sheets.

# Exported Members

- `WatersData`

# Example Usage

The module includes an example usage in the `__main__` block, which demonstrates the following:
- Instantiating `WatersData` with and without a file path.
- Saving processed data to a file.
- Retrieving and printing the data tag.
- Searching for peaks in the data with specified width and height thresholds.
- Calculating and printing the area under the peaks.
- Retrieving the area data for the peaks.

# Notes

- The `parameter_checker` decorator ensures the data file path is valid and ends with the `.arw` extension.
- The `make_tag` method constructs a tag using specific information fields from the data file, which helps in identifying the data set.
- The `process_raw_data` method is crucial for converting the raw data into structured DataFrames for further analysis.
- The example usage provides a practical demonstration of how to work with WatersData for HPLC data processing and analysis.