import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List

import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ['MBAPY_AUTO_IMPORT_TORCH'] = 'False'
os.environ['MBAPY_FAST_LOAD'] = 'True'
from mbapy.base import put_err
from mbapy.plot import get_palette, save_show

if __name__ == '__main__':
    from mbapy.file import get_paths_with_extension, get_valid_file_path
    from mbapy.scripts._script_utils_ import clean_path, show_args
else:
    from ..file import get_paths_with_extension, get_valid_file_path
    from ._script_utils_ import clean_path, show_args


def plot_hplc_load_arw_file(path: Path):
    lines = path.read_text().splitlines()
    info_df = pd.DataFrame([lines[1].split('\t')], columns = lines[0].split('\t'))
    data_df = pd.DataFrame([line.split('\t') for line in lines[2:]],
                           columns = ['Time', 'Absorbance'])
    return info_df, data_df.astype({'Time': float, 'Absorbance': float})
    
def plot_hplc_plot_waters(file_name:str, info_df: pd.DataFrame, data_df: pd.DataFrame, args):
    fig, ax = plt.subplots(figsize=(10, 6))
    xlim = args.xlim.split(',')
    if isinstance(info_df, pd.DataFrame):
        au_units = info_df['"检测单位"'][0].replace('"', '')
        xlim = [float(xlim[0]), data_df['Time'].max() if xlim[1] == 'None' else float(xlim[1])]
        ax.plot(data_df['Time'], data_df['Absorbance'], color = args.color, label = info_df[''])
    else:
        au_units = info_df[0]['"检测单位"'][0].replace('"', '')
        xlim_max = data_df[0]['Time'].max()
        for label, info_df_i, data_df_i in zip(args.file_labels, info_df, data_df):
            label_string, color = label
            ax.plot(data_df_i['Time'], data_df_i['Absorbance'], color = color, label = label_string)
            xlim_max = max(xlim_max, data_df_i['Time'].max())
        xlim = [float(xlim[0]), xlim_max if xlim[1] == 'None' else float(xlim[1])]
    plt.xticks(size = 20)
    plt.yticks(size = 20)
    ax.set_xlabel('Time (min)', fontsize=25)
    ax.set_ylabel(f'Absorbance ({au_units})', fontsize=25)
    ax.set_xlim(xlim[0], xlim[1])
    plt.legend(fontsize=15, loc = args.legend_pos, bbox_to_anchor = (args.legend_pos_bbox1, args.legend_pos_bbox2), draggable = True)
    save_show(os.path.join(args.output, f'{file_name} absorbance.png'), dpi = 600)
    
def plot_hplc(args):           
    # process args
    args.input = clean_path(args.input)
    args.output = clean_path(args.output) if args.output else args.input
    if not os.path.isdir(args.output):
        print(f'given output {args.output} is a file, change it to parent dir')
        args.output = args.output.parent
    # labels
    labels, colors = {}, get_palette(len(args.labels.split(';')), mode = 'hls')
    for idx, i in enumerate(args.labels.split(';')):
        if i:
            pack = i.split(',')
            mass, label, color = pack[0], pack[1], pack[2] if len(pack) == 3 else colors[idx]
            labels[float(mass)] = [label, color]
    args.labels = labels
    # file labels
    file_labels, colors = [], get_palette(len(args.file_labels.split(';')), mode = 'hls')
    for idx, i in enumerate(args.file_labels.split(';')):
        if i:
            pack = i.split(',')
            label, color = pack[0], pack[1] if len(pack) == 2 else colors[idx]
            file_labels.append([label, color])
    args.file_labels = file_labels
    # find base peak file and peak list file
    if args.system == 'waters':
        paths = get_paths_with_extension(args.input, ['arw'], recursive=args.recursive)
        load_data = plot_hplc_load_arw_file
    else:
        raise NotImplementedError(f'not support HPLC system: {args.system}')
    dfs = {path:load_data(Path(path)) for path in paths}
    dfs = {k:v for k,v in dfs.items() if v is not None}
    if not dfs:
        raise FileNotFoundError(f'can not find data files in {args.input}')
    # show args
    show_args(args, ['input', 'system', 'recursive', 'merge', 'output', 'min_peak_width',
                     'colors', 'labels', 'file_labels', 'labels_eps', 'xlim', 'expand',
                     'legend_pos', 'legend_pos_bbox1', 'legend_pos_bbox2'])
    # show data general info and output peak list DataFrame
    if args.merge:
        if args.system == 'waters':
            dfs = list(dfs.values())
            info_df, data_df = [d[0] for d in dfs], [d[1] for d in dfs]
            plot_hplc_plot_waters('merge', info_df, data_df, args)
    else:
        for path, df in dfs.items():
            path = Path(path).resolve()
            # plot each df
            if args.system == 'waters':
                info_df, data_df = df
                # output csv
                sample_name, sample_time, sample_channle = info_df['"样品名称"'][0], info_df['"采集日期"'][0], info_df['"通道"'][0]
                csv_name = f'{sample_name} - {sample_time} - {sample_channle}'.replace('"', '')
                csv_name = get_valid_file_path(csv_name.replace('/', '-'), valid_len=500).replace(':', '-')
                info_df.to_csv(str(path.parent / get_valid_file_path(csv_name + ' - info.csv')))
                data_df.to_csv(str(path.parent / get_valid_file_path(csv_name + ' - data.csv')))
                # plot
                print(f'plot {path.stem}: {csv_name}')
                print(info_df.T)
                plot_hplc_plot_waters(csv_name, info_df, data_df, args)


_str2func = {
    'plot-hplc': plot_hplc,
}


def main(sys_args: List[str] = None):
    args_paser = argparse.ArgumentParser()
    subparsers = args_paser.add_subparsers(title='subcommands', dest='sub_command')
    
    plot_hplc_args = subparsers.add_parser('plot-hplc', description='plot hplc spectrum')
    # set input file argument
    plot_hplc_args.add_argument("-i", "--input", type = str, default='.',
                                help="data file directory, default is %(default)s.")
    plot_hplc_args.add_argument("-s", "--system", type = str, default='waters',
                                help="HPLC system. Default is %(default)s, only accept arw file exported by Waters.")
    plot_hplc_args.add_argument('-r', '--recursive', action='store_true', default=False,
                                help='search input directory recursively, default is %(default)s.')
    plot_hplc_args.add_argument('-merge', action='store_true', default=False,
                                help='merge multi files into one plot, default is %(default)s.')
    # set output file argument
    plot_hplc_args.add_argument("-o", "--output", type = str, default=None,
                                help="output file dir or path. Default is %(default)s, means same as input dir")
    # set draw argument
    plot_hplc_args.add_argument('--min-peak-width', type = float, default=4,
                                help='filter peaks with min width in hplc/Charge plot, default is %(default)s.')
    plot_hplc_args.add_argument('-xlim', type = str, default='0,None',
                                help='set x-axis limit, input as "0,15", default is %(default)s.')
    plot_hplc_args.add_argument('-cols', '--colors', type = str, default='black',
                                help='draw color, default is %(default)s.')
    plot_hplc_args.add_argument('-labels', '--labels', type = str, default='',
                                help='labels, input as 1000,Pep1;1050,Pep2, default is %(default)s.')
    plot_hplc_args.add_argument('-flabels', '--file-labels', type = str, default='',
                                help='labels, input as 228,blue;304,red, default is %(default)s.')
    plot_hplc_args.add_argument('--labels-eps', type = float, default=0.5,
                                help='eps to recognize labels, default is %(default)s.')
    plot_hplc_args.add_argument('-expand', '--expand', type = float, default=0.2,
                                help='how much the x-axis and y-axisto be expanded, default is %(default)s.')
    plot_hplc_args.add_argument('-lpos', '--legend-pos', type = str, default='upper center',
                                help='legend position, can be string as "upper center", or be float as 0.1,0.2, default is %(default)s')
    plot_hplc_args.add_argument('-lposbbox1', '--legend-pos-bbox1', type = float, default=1.1,
                                help='legend position bbox 1 to anchor, default is %(default)s')
    plot_hplc_args.add_argument('-lposbbox2', '--legend-pos-bbox2', type = float, default=1,
                                help='legend position bbox 2 to anchor, default is %(default)s')

    
    args = args_paser.parse_args(sys_args)
    
    if args.sub_command in _str2func:
        print(f'excuting command: {args.sub_command}')
        _str2func[args.sub_command](args)
    else:
        put_err(f'no such sub commmand: {args.sub_command}')

if __name__ == "__main__":
    # dev code, MUST COMMENT OUT BEFORE RELEASE
    # pass
    
    main()