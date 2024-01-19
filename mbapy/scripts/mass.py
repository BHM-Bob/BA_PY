import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ['MBAPY_AUTO_IMPORT_TORCH'] = 'False'
os.environ['MBAPY_FAST_LOAD'] = 'True'
from mbapy import base, file
from mbapy.plot import get_palette, save_show

if __name__ == '__main__':
    from mbapy.scripts._script_utils_ import clean_path
else:
    from ._script_utils_ import clean_path


def plot_mass_load_file(path: Path):
    lines = path.read_text().splitlines()
    df = pd.DataFrame([line.split('\t') for line in lines[1:]],
                        columns = lines[0].split('\t'))
    if df.shape[1] == 2 and df.columns[0] == 'Time':
        return df.astype(float)
    elif df.shape[1] == 10:
        return df.astype({'Mass/Charge':float, 'Height':float, 'Charge':int,
                            'Monoisotopic':str, 'Mass (charge)':str,
                            'Mass/charge (charge)':str})

def plot_mass_plot_basepeak(name:str, base_peak: pd.DataFrame, args):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(base_peak['Time'], base_peak['Intensity'], color = args.color)
    plt.xticks(size = 20)
    plt.yticks(size = 20)
    plt.yscale('log')
    ax.set_title(f'{name} (TIC of TOF MS)', fontsize=25)
    ax.set_xlabel('Time (min)', fontsize=25)
    ax.set_ylabel('Intensity (cps)', fontsize=25)
    save_show(os.path.join(args.output, f'{name}.png'), dpi = 300)

def plot_mass_plot_peaklist(name:str, peak_list: pd.DataFrame, args):
    def plot_vlines(x, y, col, label = ''):
        plt.vlines(x, 0, y, colors = [col] * len(x), label = label)
        plt.scatter(x, y, c = col)
    fig, ax = plt.subplots(figsize=(8, 6))
    idx = peak_list['Monoisotopic'] == 'Yes'
    plot_vlines(peak_list['mass_data'], peak_list['Height'], args.color)
    labels_ms = np.array(list(args.labels.keys()))
    for ms, h, c in zip(peak_list['mass_data'][idx], peak_list['Height'][idx],
                        peak_list['Charge'][idx]):
        matched = np.where(np.abs(labels_ms - ms) < args.labels_eps)[0]
        if matched.size > 0:
            label, color = args.labels.get(labels_ms[matched[0]])
            plot_vlines([ms], [h], color, label)
            ax.text(ms, h, f'* {ms:.2f}({c:d})', fontsize=15, color = color)
        else:
            ax.text(ms, h, f'* {ms:.2f}({c:d})', fontsize=15, color = args.color)
    plt.xticks(size = 20)
    plt.yticks(size = 20)
    plt.yscale('log')
    axis_lim = (1-args.expand, 1+args.expand)
    plt.xlim(peak_list['mass_data'].min() * axis_lim[0], peak_list['mass_data'].max() * axis_lim[1])
    plt.ylim(peak_list['Height'].min() * axis_lim[0], peak_list['Height'].max() * axis_lim[1])
    ax.set_title(f'{name} (Peak List of TOF MS)', fontsize=25)
    ax.set_xlabel(f'Mass{"" if args.mass else "/charge"}', fontsize=25)
    ax.set_ylabel('Intensity (cps)', fontsize=25)
    plt.legend(fontsize=15)
    save_show(os.path.join(args.output, f'{name}.png'), dpi = 300)
    
def plot_mass(args):           
    # process args
    args.dir = clean_path(args.dir)
    print(f'get arg: dir: {args.dir}')
    # find base peak file and peak list file
    paths = glob.glob(os.path.join(args.dir, '*.txt'))
    dfs = {path:plot_mass_load_file(Path(path)) for path in paths}
    dfs = {k:v for k,v in dfs.items() if v is not None}
    if not dfs:
        raise FileNotFoundError(f'can not find two files in {args.dir}')
    if not os.path.isdir(args.output):
        print(f'given output {args.output} is a file, change it to parent dir')
        args.output = args.output.parent
    print(f'get arg: output: {args.output}')
    print(f'get arg: mass: {args.mass}')
    labels, colors = {}, get_palette(len(args.labels.split(';')), mode = 'hls')
    for idx, i in enumerate(args.labels.split(';')):
        pack = i.split(',')
        mass, label, color = pack[0], pack[1], pack[2] if len(pack) == 3 else colors[idx]
        labels[float(mass)] = [label, color]
    args.labels = labels
    print(f'get arg: labels: {labels}')
    # show data general info and output peak list DataFrame
    for n,df in dfs.items():
        name = Path(n).resolve().stem
        print(f'\n\n\n\n\n{name}: peak list: signal(s):\n', df)
        df.to_csv(os.path.join(args.output, f'{name}.csv'))
        # process Mass (charge) and identify mass
        if df.shape[1] == 10:
            df['Mass (charge)'] = df['Mass (charge)'].str.extract(r'(\d+\.\d+)', expand=False).astype(float)
            df['Mass/charge (charge)'] = df['Mass/charge (charge)'].str.extract(r'(\d+\.\d+)', expand=False).astype(float)
            df['mass_data'] = df['Mass (charge)'] if args.mass else df['Mass/charge (charge)']
            drop_idx = df[df['Height'] < args.min_height].index
            if not drop_idx.empty:
                print(f'drop data with min-height: {args.min_height} and only these data remained:\n',
                      df[df['Height'] >= args.min_height])
                df.drop(drop_idx, axis = 0, inplace = True)
    # plot each df
    for n,df in dfs.items():
        name = Path(n).resolve().stem
        if df.shape[1] == 2:
            # plot base peak
            print(f'ploting base peak: {name}')
            plot_mass_plot_basepeak(name, df, args)
        elif df.shape[0] > 0: # avoid drop all data but still draw
            # plot peak list
            print(f'ploting peak list: {name}')
            plot_mass_plot_peaklist(name, df, args)

_str2func = {
    'plot-mass': plot_mass,
}


def main(sys_args: List[str] = None):
    args_paser = argparse.ArgumentParser()
    subparsers = args_paser.add_subparsers(title='subcommands', dest='sub_command')
    
    plot_mass_args = subparsers.add_parser('plot-mass', description='plot mass spectrum')
    # set dir argument
    plot_mass_args.add_argument("-d", "--dir", type = str, help="txt file directory")
    # set output file argument
    plot_mass_args.add_argument("-o", "--output", type = str, default='.',
                                help="output file dir or path, default is %(default)s")
    # set draw argument
    plot_mass_args.add_argument('-m', '--mass', action='store_true', default=False,
                                help='draw Mass instead of Mass/charge which is Mass+z, default is %(default)s')
    plot_mass_args.add_argument('-min', '--min-height', type = int, default=0,
                                help='filter data with min height, default is %(default)s')
    plot_mass_args.add_argument('-col', '--color', type = str, default='black',
                                help='draw color, default is %(default)s')
    plot_mass_args.add_argument('-labels', '--labels', type = str, default='',
                                help='labels, input as 1000,Pep1;1050,Pep2, default is %(default)s')
    plot_mass_args.add_argument('--labels-eps', type = float, default=0.5,
                                help='eps to recognize labels, default is %(default)s')
    plot_mass_args.add_argument('-expand', '--expand', type = float, default=0.2,
                                help='how much the x-axis and y-axisto be expanded, default is %(default)s')

    
    args = args_paser.parse_args(sys_args)
    
    if args.sub_command in _str2func:
        print(f'excuting command: {args.sub_command}')
        _str2func[args.sub_command](args)
    else:
        base.put_err(f'no such sub commmand: {args.sub_command}')

if __name__ == "__main__":
    main()