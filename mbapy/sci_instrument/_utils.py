'''
Date: 2024-05-25 08:43:43
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2024-07-10 11:19:39
Description: 
'''

from typing import List

if __name__ == '__main__':
    from mbapy.plot import PLT_MARKERS, get_palette
else:
    from ..plot import PLT_MARKERS, get_palette
    
    
def process_label_col(labels: str, file_col_mode: str = 'hls'):
    """
    This function is used to process label information, extract and format labels\n
    LABEL1,COLOR1;LABEL2,COLOR2;...

    Parameters:
        - labels (str): Label string, separated by semicolons
        - file_col_mode (str): File column mode, default is 'hls'

    Return:
        - file_labels (list): List containing the processed label information
        
    Example:
    >>> process_label_col('file1,red;file2,blue', 'hls')
    """
    labels = '' if labels is None else labels
    file_labels, colors = [], get_palette(len(labels.split(';')), mode = file_col_mode)
    for idx, i in enumerate(labels.split(';')):
        if i:
            pack = i.split(',')
            label, color = pack[0], pack[1] if len(pack) == 2 else colors[idx]
            file_labels.append([label, color])
    return file_labels

def process_num_label_col(labels: str, peak_col_mode: str = 'hls'):
    """
    This function is used to process label information, extract and format labels\n
    NUMBER1,LABEL1,COLOR1;NUMBER2,LABEL2,COLOR2;...

    Parameters:
        - labels (str): Label string, separated by semicolons
        - peak_col_mode (str): Peak column mode, default is 'hls'

    Return:
        - peak_labels (dict): Dictionary containing the processed label information
        
    Example:
    >>> process_num_label_col('10,peak1,red;20,peak2,blue', 'hls')
    """
    labels = '' if labels is None else labels
    peak_labels, cols = {}, get_palette(len(labels.split(';')), mode = peak_col_mode)
    for i, label in enumerate(labels.split(';')):
        if label:
            items = label.split(',')
            if len(items) == 2:
                (t, label), color = items, cols[i]
            elif len(items) == 3:
                t, label, color = items
            peak_labels[float(t)] = [label, color]
    return peak_labels

def process_num_label_col_marker(labels: str, peak_col_mode: str = 'hls',
                                 markers: List[str] = PLT_MARKERS):
    """
    This function is used to process label information, extract and format labels\n
    NUMBER1,LABEL1,COLOR1,MARKER1;NUMBER2,LABEL2,COLOR2,MARKER2;...

    Parameters:
        - labels (str): Label string, separated by semicolons
        - peak_col_mode (str): Peak column mode, default is 'hls'
        - markers (List[str]): List of marker types, default is PLT_MARKERS

    Return:
        - peak_labels (dict): Dictionary containing the processed label information
        
    Example:
    >>> process_num_label_col_marker('10,peak1,red,o;20,peak2,blue,v', 'hls', ['o', 'v', '^'])
    """
    labels = '' if labels is None else labels
    peak_labels, cols = {}, get_palette(len(labels.split(';')), mode = peak_col_mode)
    for i, label in enumerate(labels.split(';')):
        if label:
            items = label.split(',')
            if len(items) == 2:
                (t, label), color, maker = items, cols[i], markers[i]
            elif len(items) == 3:
                (t, label, color), maker = items, markers[i]
            elif len(items) == 4:
                t, label, color, maker = items
            peak_labels[float(t)] = [label, color, maker]
    return peak_labels


__all__ = [
    'process_label_col',
    'process_num_label_col',
    'process_num_label_col_marker',
]