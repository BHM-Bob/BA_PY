'''
Date: 2024-05-20 16:53:21
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2025-01-12 13:54:02
Description: mbapy.sci_instrument.hplc._base
'''
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import scipy
import scipy.integrate
try:
    from scipy.integrate import simps as simpson
except :
    from scipy.integrate import simpson
import scipy.signal

if __name__ == '__main__':
    from mbapy.base import get_default_call_for_None, put_err
    from mbapy.sci_instrument._base import SciInstrumentData
else:
    from ...base import get_default_call_for_None, put_err
    from .._base import SciInstrumentData
    
    
class HplcData(SciInstrumentData):
    def __init__(self, data_file_path: Union[None, str, List[str]] = None) -> None:
        super().__init__(data_file_path)
        self.X_HEADER = 'Time'
        self.Y_HEADER = 'Absorbance'
        self.TICKS_IN_MINUTE = 60 # how many ticks in one minute
        self.IS_PDA = False
        self.plot_params: Dict[str, Any] = {'peak_label': True}
        self.refined_abs_data: pd.DataFrame = None
        self.area: Dict[int, Dict[str, Union[float, int, np.ndarray]]] = {} # the area and underline of peaks
        self.peaks_idx: np.ndarray = None # the index of peaks in the data, in tick.
        
    def get_abs_data(self, origin_data: bool = False, *args, **kwargs) -> pd.DataFrame:
        if self.refined_abs_data is not None and not origin_data:
            return self.refined_abs_data
        return self.processed_data if not self.check_processed_data_empty() else self.process_raw_data()
    
    def search_peaks(self, peak_width_threshold: float, peak_height_threshold: float,
                     start_search_time: float = 0, end_search_time: float = None,
                     peak_height_rel: float = 1) -> np.ndarray:
        """
        Parameters
            - start_search_time: float, start time to search peaks, in minutes, will transfer by get_tick_by_minute
            - end_search_time: float, end time to search peaks, in minutes, will transfer to tick by get_tick_by_minute
            - peak_width_threshold: float, the minimum width of peaks, in minutes, will transfer to tick by get_tick_by_minute
            - peak_height_threshold: float, the minimum height of peaks, scipy.signal.find_peaks parameter(prominence).
            - peak_height_rel: float, scipy.signal.find_peaks parameter(rel_height), default is 1.
            
        Returns:
            - peaks_idx: np.ndarray, the index of peaks in the data. first return value of scipy.signal.find_peaks.
        """
        # prepare prams and search peaks
        df = self.get_abs_data()
        t0 = self.get_tick_by_minute(self.get_abs_data()[self.X_HEADER][0]) # for refine x-axis data
        st = self.get_tick_by_minute(start_search_time) - t0 # start_search_time单位为分钟, st's unit is data tick
        ed = self.get_tick_by_minute(end_search_time) - t0 if end_search_time is not None else None
        width = self.get_tick_by_minute(peak_width_threshold)
        peaks_idx, peak_props = scipy.signal.find_peaks(df[self.Y_HEADER], rel_height = peak_height_rel,
                                                        prominence = peak_height_threshold, width = width)
        # filter peaks by start_search_time and end_search_time
        self.peaks_idx = peaks_idx[peaks_idx >= st]
        if ed is not None:
            self.peaks_idx = self.peaks_idx[self.peaks_idx <= ed]
        return self.peaks_idx
    
    def calcu_single_peak_area(self, st_tick: int, ed_tick: int, abs_data: pd.DataFrame = None)\
        -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Parameters:
            - st_tick: int, start tick of the peak.
            - ed_tick: int, end tick of the peak.
            - abs_data: pd.DataFrame, the absorbance data, default is None. will use self.get_abs_data().copy() if None.
            
        Returns:
            - st_minute: float, the start time of the peak, in minutes.
            - ed_minute: float, the end time of the peak, in minutes.
            - area: float, the area of the peak.
            - y: np.ndarray, the y-axis data of the peak.
            - underline_x: np.ndarray, the x-axis data of the underline.
            - underline_y: np.ndarray, the y-axis data of the underline.
            - refined_y: np.ndarray, the refined y-axis data of the peak, it's underline is y=0, for calculating area.
        """
        abs_data = get_default_call_for_None(abs_data, self.get_abs_data)
        st_minute, ed_minute = abs_data[self.X_HEADER][st_tick], abs_data[self.X_HEADER][ed_tick]
        y = abs_data[self.Y_HEADER][(abs_data[self.X_HEADER] >= st_minute) & (abs_data[self.X_HEADER] <= ed_minute)]
        underline_x = abs_data[self.X_HEADER][(abs_data[self.X_HEADER] >= st_minute) & (abs_data[self.X_HEADER] <= ed_minute)]
        underline_y = np.linspace(abs_data[self.Y_HEADER][st_tick], abs_data[self.Y_HEADER][ed_tick], len(y))
        refined_y = y - underline_y
        return st_minute, ed_minute, simpson(refined_y, x=underline_x), y, underline_x, underline_y, refined_y
        
    
    def calcu_peaks_area(self, peaks_idx: np.ndarray, rel_height: float = 1,
                         allow_overlap: bool = False) -> Dict[int, Dict[str, Union[float, int, np.ndarray]]]:
        """
        Parameters:
            - peaks_idx: np.ndarray, the index of peaks in the data, in tick.
            - rel_height: float, scipy.signal.peak_widths parameter(rel_height), default is 1.
            - overlap: bool, whether to avoid the overlap of peaks, default is False.
            
        Returns:
            - area: Dict[int, Dict[str, Union[float, np.ndarray]]], the area and underline of peaks.
                - key: the index of peaks, in tick.
                - 'peak_idx': int, the index of peaks, in tick.
                - 'time': float, the time of peaks, in minutes.
                - 'area': float, the area of peaks.
                - 'underline-x': np.ndarray, x posistion of the underline.
                - 'underline-y': np.ndarray, y posistion of the underline.
                - 'peak-line-y': np.ndarray, y posistion of the peak line.
                - 'left-tick': int, the tick of left side of the underline.
                - 'right-tick': int, the tick of right side of the underline.
                - 'left': int, the minute of left side of the underline.
                - 'right': int, the minute of right side of the underline.
                - 'width': float, the width of the peak(in ticks).
                - 'height': float, the height of the peak.
        """
        abs_data = self.get_abs_data()
        widths, width_heights, left, right = scipy.signal.peak_widths(abs_data[self.Y_HEADER], peaks_idx, rel_height = rel_height)
        self.area, self.peaks_idx = {}, peaks_idx
        for i, peak_tick in enumerate(peaks_idx):
            # adjust curruent peak's boundary if not allow overlap
            if not allow_overlap:
                # check left overlap
                if i > 0 and left[i-1] > left[i]:
                    left[i] = left[i] = right[i-1]
                # check right overlap
                if i < len(peaks_idx)-1 and right[i+1] < right[i]:
                    right[i] = right[i] = left[i+1]
            # calcu underline and peak area
            st_tick, ed_tick = round(left[i]), round(right[i])
            st_minute, ed_minute, area, y, underline_x, underline_y, refined_y = self.calcu_single_peak_area(st_tick, ed_tick, abs_data)
            time = abs_data[self.X_HEADER][peak_tick]
            self.area[peak_tick] = {'peak_idx': peak_tick, 'time': time,
                                    'area': area,
                                    'underline-x': np.array(underline_x), 'underline-y': underline_y,
                                    'peak-line-y': abs_data[self.Y_HEADER][(abs_data[self.X_HEADER] >= st_minute) & (abs_data[self.X_HEADER] <= ed_minute)],
                                    'left-tick': st_tick, 'right-tick': ed_tick,
                                    'left': st_minute, 'right': ed_minute,
                                    'width': widths[i], 'height': width_heights[i]}
        return self.area
    
    def get_area(self, peaks_idx: np.ndarray = None, rel_height: float = 1,
                 allow_overlap: bool = False) -> Dict[int, Dict[str, Union[float, int, np.ndarray]]]:
        """
        Returns:
            - area: Dict[int, Dict[str, Union[float, np.ndarray]]], the area and underline of peaks.
                - key: the index of peaks, in tick.
                - 'peak_idx': int, the index of peaks, in tick.
                - 'time': float, the time of peaks, in minutes.
                - 'area': float, the area of peaks.
                - 'underline-x': np.ndarray, x posistion of the underline.
                - 'underline-y': np.ndarray, y posistion of the underline.
                - 'peak-line-y': np.ndarray, y posistion of the peak line.
                - 'left-tick': int, the tick of left side of the underline.
                - 'right-tick': int, the tick of right side of the underline.
                - 'left': int, the minute of left side of the underline.
                - 'right': int, the minute of right side of the underline.
                - 'width': float, the width of the peak(in ticks).
                - 'height': float, the height of the peak.
        """
        if isinstance(peaks_idx, np.ndarray) and np.array_equal(peaks_idx, self.peaks_idx):
            return self.calcu_peaks_area(peaks_idx, rel_height, allow_overlap)
        elif self.area:
            return self.area
        else:
            return put_err('No area calculated and no peaks_idx provided, return None.')
    
    
__all__ = [
    'HplcData',
]
    
if __name__ == '__main__':
    pass