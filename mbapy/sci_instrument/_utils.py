'''
Date: 2024-05-25 08:43:43
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2024-06-01 19:55:07
Description: 
'''

from typing import List

if __name__ == '__main__':
    from mbapy.plot import PLT_MARKERS, get_palette
else:
    from ..plot import PLT_MARKERS, get_palette
    
    
def process_label_col(labels: str, file_col_mode: str = 'hls'):
    labels = '' if labels is None else labels
    file_labels, colors = [], get_palette(len(labels.split(';')), mode = file_col_mode)
    for idx, i in enumerate(labels.split(';')):
        if i:
            pack = i.split(',')
            label, color = pack[0], pack[1] if len(pack) == 2 else colors[idx]
            file_labels.append([label, color])
    return file_labels

def process_num_label_col(labels: str, peak_col_mode: str = 'hls'):
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