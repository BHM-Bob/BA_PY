
if __name__ == '__main__':
    from mbapy.plot import get_palette
else:
    from ..plot import get_palette
    
    
def process_label_col(labels: str, file_col_mode = 'hls'):
    labels = '' if labels is None else labels
    file_labels, colors = [], get_palette(len(labels.split(';')), mode = file_col_mode)
    for idx, i in enumerate(labels.split(';')):
        if i:
            pack = i.split(',')
            label, color = pack[0], pack[1] if len(pack) == 2 else colors[idx]
            file_labels.append([label, color])
    return file_labels

def process_num_label_col(labels: str, peak_col_mode = 'hls'):
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