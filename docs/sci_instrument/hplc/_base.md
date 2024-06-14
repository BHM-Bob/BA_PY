# Module Overview

This module, `mbapy.sci_instrument.hplc._base`, is part of the `mbapy` package and is specifically designed to handle High-Performance Liquid Chromatography (HPLC) data. It includes a class `HplcData` that extends `SciInstrumentData` and provides methods for processing, analyzing, and visualizing HPLC data.

# Class

## HplcData
### Class Initialization
`HplcData(data_file_path: Union[None, str, List[str]] = None)`

### Members
- `X_HEADER`: The header name for the x-axis data, set to 'Time'.
- `Y_HEADER`: The header name for the y-axis data, set to 'Absorbance'.
- `TICKS_IN_MINUTE`: The number of data points per minute, set to 60.
- `area`: A dictionary to store the area and underline data for each peak.
- `peak_idx`: A NumPy array to store the indices of the peaks.

### Methods

#### get_abs_data()
Retrieves the absolute data for the HPLC, either from processed data or by processing the raw data.

#### search_peaks(peak_width_threshold: float, peak_height_threshold: float, start_search_time: float = 0, end_search_time: float = None, peak_height_rel: float = 1) -> np.ndarray
Searches for peaks within the HPLC data based on specified criteria.

##### Parameters
- `peak_width_threshold`: The minimum width of peaks in minutes.
- `peak_height_threshold`: The minimum height or prominence of peaks.
- `start_search_time`: The start time for peak searching in minutes.
- `end_search_time`: The end time for peak searching in minutes.
- `peak_height_rel`: The relative height used for peak detection.

##### Return Value
- Returns a NumPy array of indices representing the peaks in the data.

#### calcu_peaks_area(peaks_idx: np.ndarray, rel_height: float = 1, allow_overlap: bool = False) -> Dict[int, Dict]
Calculates the area under the peaks and other related properties.

##### Parameters
- `peaks_idx`: A NumPy array of indices representing the peaks.
- `rel_height`: The relative height for calculating peak widths.
- `allow_overlap`: Whether to allow overlap when calculating peak areas.

##### Return Value
- Returns a dictionary with detailed information about each peak.

#### get_area(peaks_idx: np.ndarray = None, rel_height: float = 1, allow_overlap: bool = False) -> Dict[int, Dict]
Retrieves the calculated areas for the peaks, either from existing calculations or by performing new calculations.

##### Parameters
- `peaks_idx`: Optional array of peak indices for which to calculate areas.
- `rel_height`: The relative height for peak width calculations.
- `allow_overlap`: Whether to allow overlap in peak area calculations.

##### Return Value
- Returns a dictionary containing the area and underline data for each peak.

# Exported Members

- `HplcData`

# Notes

- The `search_peaks` and `calcu_peaks_area` methods are crucial for analyzing HPLC data, allowing users to identify and quantify peaks based on various parameters.
- The class is designed to be easily extendable for more specific analysis or visualization needs related to HPLC data.