<!--
 * @Date: 2024-06-14 16:51:41
 * @LastEditors: BHM-Bob 2262029386@qq.com
 * @LastEditTime: 2024-06-14 16:56:05
 * @Description: 
-->
# Module Overview

The `mbapy.sci_instrument.mass.SCIEX` module extends the functionality of the `MassData` class to handle specific data formats associated with SCIEX mass spectrometry instruments. It introduces two classes, `SciexPeakListData` and `SciexOriData`, tailored for processing peak list data and original data files, respectively.

# Classes

## SciexPeakListData
### Class Initialization
`SciexPeakListData(data_file_path: Optional[str] = None)`
- Inherits from `MassData`.
- Sets specific headers and processing logic for peak list data files.
- Defines the expected file suffixes and the recommended suffix.
- Processes raw text data or loads processed Excel data.

### Members
- `X_HEADER`: 'Mass/charge (charge)'
- `Y_HEADER`: 'Height'
- `CHARGE_HEADER`: 'Charge'
- `X_MZ_HEADER`: 'Mass/charge (charge)'
- `X_M_HEADER`: 'Mass (charge)'
- `MULTI_HEADERS`: A list of headers relevant to peak list data.
- `HEADERS_TYPE`: A dictionary defining data types for the headers.

### Methods
- `process_raw_data`: Processes raw data by extracting numerical values and saving to `data_df` and `peak_df`.

## SciexOriData
### Class Initialization
`SciexOriData(data_file_path: Optional[str] = None)`
- Inherits from `MassData`.
- Sets specific headers and processing logic for original data files.
- Defines the expected file suffixes and the recommended suffix.
- Processes raw text data or loads processed Excel data.

### Members
- `X_HEADER`: 'Mass/Charge'
- `Y_HEADER`: 'Intensity'
- `X_MZ_HEADER`: 'Mass/Charge'
- `X_M_HEADER`: None
- `MULTI_HEADERS`: A list containing `X_HEADER` and `Y_HEADER`.
- `HEADERS_TYPE`: A dictionary defining data types for the headers.

### Methods
- Inherits `process_raw_data` from `MassData`, with specific logic for handling text data.

# Exported Members

- `SciexPeakListData`
- `SciexOriData`

# Example Usage

The module's `__main__` block demonstrates how to instantiate and use `SciexPeakListData` and `SciexOriData` classes:

```python
# Instantiate SciexPeakListData with an Excel file
pl = SciexPeakListData('path_to_peak_list_data.xlsx')
print(pl.processed_data.head())

# Instantiate SciexOriData with a text file
ori = SciexOriData('path_to_original_data.txt')
print(ori.processed_data.head())
print(ori.tag)
print(ori.get_tick_by_minute(0.6))
print(ori.search_peaks())
ori.save_processed_data()
ori.load_processed_data_file('path_to_processed_data.xlsx')
print(ori.processed_data.head())
```

# Notes

- Both classes are designed to handle different file formats associated with SCIEX mass spectrometry data.
- The `parameter_checker` decorator ensures the provided data file path is valid and matches the expected file suffixes.
- The `process_raw_data` method in `SciexPeakListData` includes additional logic to extract numerical values from the raw data.