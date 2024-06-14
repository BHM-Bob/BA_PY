<!--
 * @Date: 2024-06-14 16:48:16
 * @LastEditors: BHM-Bob 2262029386@qq.com
 * @LastEditTime: 2024-06-14 16:49:52
 * @Description: 
-->
# Module Overview

This module extends the functionality provided by the `HplcData` class for handling Sciex (scientific data) files, which are typically chromatography data files with a `.txt` extension. It introduces two classes, `SciexData` and `SciexTicData`, both of which inherit from `HplcData` and are tailored for processing Sciex data and total ion current (TIC) data, respectively.

# Classes

## SciexData
### Class Initialization
`SciexData(data_file_path: Optional[str] = None)`
- Inherits from `HplcData` and is initialized with a data file path.
- Sets the x-axis header to 'Time' and the y-axis header to 'Absorbance'.
- Loads and processes raw data upon instantiation if the file path is provided and valid.
- Generates a tag for the data based on the file path.

### Members
- Inherits all members from `HplcData`.
- `TICKS_IN_MINUTE` is set to `None`, indicating the number of ticks per minute is not defined for Sciex data.

## SciexTicData
### Class Initialization
`SciexTicData(data_file_path: Optional[str] = None)`
- Inherits from `HplcData` and is initialized similarly to `SciexData`.
- Sets the x-axis header to 'Time' and the y-axis header to 'Intensity', which is typical for TIC data.

### Members
- Inherits all members from `HplcData`.
- `TICKS_IN_MINUTE` is set to `None` for TIC data, similar to `SciexData`.

### Methods
Both `SciexData` and `SciexTicData` inherit the following methods from `HplcData`:
- `load_raw_data_file()`: Loads raw data from the specified file path.
- `process_raw_data()`: Processes the loaded raw data into a usable format.
- `make_tag()`: Generates a tag for the data, which can be used for identification.
- `get_tick_by_minute()`: Converts a minute value to the corresponding tick value based on the data's time scale.
- `save_processed_data()`: Saves the processed data to a file.
- `load_processed_data_file()`: Loads processed data from a file.

# Exported Members

- `SciexData`
- `SciexTicData`

# Notes

- The `parameter_checker` decorator is used to ensure that the provided data file path is valid and ends with the `.txt` extension.
- Both classes are designed to be flexible and can be easily extended or modified to accommodate specific processing requirements for Sciex data or TIC data.