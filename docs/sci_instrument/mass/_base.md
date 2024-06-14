<!--
 * @Date: 2024-06-14 16:51:27
 * @LastEditors: BHM-Bob 2262029386@qq.com
 * @LastEditTime: 2024-06-14 16:53:01
 * @Description: 
-->
# Module Overview

The `mbapy.sci_instrument.mass._base` module introduces a class `MassData` designed to handle mass spectrometry data. It inherits from `SciInstrumentData` and includes methods for loading, processing, and analyzing mass spectrometry data files.

# Class

## MassData
### Class Initialization
`MassData(data_file_path: Union[None, str, List[str]] = None)`
- Inherits from `SciInstrumentData`.
- Initializes with a data file path and sets specific headers for mass spectrometry data.
- Defines constants for common ESI (Electrospray Ionization) adducts and their respective iron (adduct) masses.

### Members
- `peak_df`: A pandas DataFrame to store peak data.
- `X_HEADER`: The header for the x-axis data, set to 'Mass/charge (charge)'.
- `Y_HEADER`: The header for the y-axis data, set to 'Height'.
- `CHARGE_HEADER`: Not defined explicitly in the class.
- `X_MZ_HEADER`: Not defined explicitly in the class.
- `X_M_HEADER`: Not defined explicitly in the class.
- `MULTI_HEADERS`: A list containing `X_HEADER` and `Y_HEADER`.
- `HEADERS_TYPE`: A dictionary defining data types for headers, with both `X_HEADER` and `Y_HEADER` set to float.

### Methods

#### load_processed_data_file(path: str = None, data_bytes: bytes = None)
Loads processed data from a file or bytes into `data_df` and optionally `peak_df`.

#### process_raw_data(*args, **kwargs)
Processes raw data by splitting lines and creating a DataFrame for mass spectrometry data.

#### save_processed_data(path: str = None, *args, **kwargs)
Saves the processed data to an Excel file with 'Data' and optionally 'Peak' sheets.

#### get_tick_by_minute(x)
Not supported for `MassData`, always returns an error.

#### search_peaks(xlim: Tuple[float, float] = None, min_width: float = 4, parallel: TaskPool = None, n_parallel: int = 4)
Searches for peaks within the data, with options for filtering by xlim and using parallel processing.

#### filter_peaks(xlim: Tuple[float, float] = None, min_height: float = None, min_height_percent: float = 1)
Filters peaks based on xlim and minimum height criteria.

# Constants

- `ESI_IRON_MODE`: A dictionary containing common ESI adducts and their respective properties.

# Exported Members

- `MassData`

# Notes

- The `parameter_checker` decorator is used to validate the data file path and ensure it is appropriate for mass spectrometry data files.
- The `search_peaks` method utilizes `scipy.signal.find_peaks_cwt` for peak detection and can leverage parallel processing to improve performance.
- The `filter_peaks` method allows for filtering of the detected peaks based on specified criteria, such as xlim and minimum height.
- The class is designed to be flexible and can be extended or modified for specific mass spectrometry data analysis needs.
