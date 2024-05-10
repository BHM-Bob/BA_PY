import argparse
import glob
import os
from functools import partial
from pathlib import Path
from typing import Dict, List

import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ['MBAPY_AUTO_IMPORT_TORCH'] = 'False'
os.environ['MBAPY_FAST_LOAD'] = 'True'

# if __name__ == '__main__':
from mbapy.base import put_err
from mbapy.plot import get_palette, save_show
from mbapy.file import get_paths_with_extension, get_valid_file_path
from mbapy.scripts._script_utils_ import clean_path, Command, excute_command, show_args
# else:
#     from ..base import put_err
#     from ..plot import get_palette, save_show
#     from ..file import get_paths_with_extension, get_valid_file_path
#     from ._script_utils_ import clean_path, excute_command, show_args


def _plot_vlines(x, y, col, label = None):
    plt.vlines(x, 0, y, colors = [col] * len(x), label = label)
    plt.scatter(x, y, c = col)
    
class plot_mass(Command):
    def __init__(self, args: argparse.Namespace, printf=print) -> None:
        super().__init__(args, printf)
        self.use_recursive_output = False
        
    @staticmethod
    def process_labels(labels: str):
        result, colors = {}, get_palette(len(labels.split(';')), mode = 'hls')
        for idx, i in enumerate(labels.split(';')):
            if i:
                pack = i.split(',')
                mass, label, color = pack[0], pack[1], pack[2] if len(pack) == 3 else colors[idx]
                result[float(mass)] = [label, color]
        return result
    
    def process_args(self):          
        # process input and output args
        # after process, output whether be str or be None if recursive
        self.args.dir = clean_path(self.args.dir)
        if self.args.output is None:
            self.args.output = self.args.dir
            if not os.path.isdir(self.args.output):
                print(f'given output {self.args.output} is a file, change it to parent dir')
                self.args.output = self.args.output.parent
        if self.args.recursive and self.args.output:
            self.args.output = None
        self.use_recursive_output = self.args.recursive and self.args.output is None
        # process labels args
        self.args.labels = self.process_labels(self.args.labels)
        if ',' in self.args.legend_pos:
            self.args.legend_pos = self.args.legend_pos.split(',')
            self.args.legend_pos = (float(self.args.legend_pos[0]), float(self.args.legend_pos[1]))
            
    @staticmethod
    def load_file(path: Path):
        lines = path.read_text().splitlines()
        df = pd.DataFrame([line.split('\t') for line in lines[1:]],
                            columns = lines[0].split('\t'))
        if df.shape[1] == 2 and df.columns[0] == 'Time' and df.columns[1] == 'Intensity':
            # setattr(df, '_content_type', 'base peak') # NOTE: this does not work
            df._attrs['content_type'] = 'base peak'
            return df.astype(float)
        elif df.shape[1] == 2 and df.columns[0] == 'Time' and df.columns[1] == 'Absorbance':
            df._attrs['content_type'] = 'absorbance'
            return df.astype(float)
        elif df.shape[1] == 2 and df.columns[0] == 'Mass/Charge' and df.columns[1] == 'Intensity':
            df._attrs['content_type'] = 'mass-charge'
            return df.astype(float)
        elif df.shape[1] == 10:
            df._attrs['content_type'] = 'peak list'
            df =  df.astype({'Mass/Charge':float, 'Height':float, 'Charge':int,
                            'Monoisotopic':str, 'Mass (charge)':str,
                            'Mass/charge (charge)':str})
            df['Mass (charge)'] = df['Mass (charge)'].str.extract(r'(\d+\.\d+)', expand=False).astype(float)
            df['Mass/charge (charge)'] = df['Mass/charge (charge)'].str.extract(r'(\d+\.\d+)', expand=False).astype(float)
            return df
        else:
            return put_err(f'Can not recognizable txt file: {path}, skip.')
        
    @staticmethod
    def filter_peaklist_data(df: pd.DataFrame, args):
        df['mass_data'] = df['Mass (charge)'] if args.mass else df['Mass/charge (charge)']
        drop_idx = df[df['Height'] < args.min_height].index
        if not drop_idx.empty:
            print(f'drop data with min-height: {args.min_height} and only these data remained:\n',
                df[df['Height'] >= args.min_height])
            df.drop(drop_idx, axis = 0, inplace = True)
        return df
            
    @staticmethod
    def load_data(dir: str, recursive: bool):
        # find base peak file and peak list file
        paths = get_paths_with_extension(dir, ['txt'], recursive)
        dfs = {path:plot_mass.load_file(Path(path)) for path in paths}
        dfs = {k:v for k,v in dfs.items() if v is not None}
        if not dfs:
            raise FileNotFoundError(f'can not find txt files in {dir}')
        return dfs
    
    @staticmethod
    def plot_basepeak(name:str, base_peak: pd.DataFrame, args, ax):
        ax.plot(base_peak['Time'], base_peak['Intensity'], color = args.color)
        plt.yscale('log')
        ax.set_title(f'{name} (TIC of TOF MS)', fontsize=25)
        ax.set_xlabel('Time (min)', fontsize=25)
        ax.set_ylabel('Intensity (cps)', fontsize=25)
        
    @staticmethod
    def plot_absorbance(name:str, df: pd.DataFrame, args, ax):
        ax.plot(df['Time'], df['Absorbance'], color = args.color)
        ax.set_title(f'{name} (Absorbance)', fontsize=25)
        ax.set_xlabel('Time (min)', fontsize=25)
        au_units = ('m' if df['Absorbance'].max() > 10 else '') + 'AU'
        ax.set_ylabel(f'Absorbance ({au_units})', fontsize=25)
        ax.set_xlim(0, df['Time'].max())
        
    @staticmethod
    def plot_peaklist(name:str, df: pd.DataFrame, args, ax):
        if args.xlim:
            xlim = [float(i) for i in args.xlim.split(',')]
            df = df[(df['Mass/Charge'] >= xlim[0]) & (df['Mass/Charge'] <= xlim[1])]
            print(f'x-axis data limit set to {xlim}')
        idx = df['Monoisotopic'] == 'Yes'
        _plot_vlines(df['mass_data'], df['Height'], args.color)
        labels_ms = np.array(list(args.labels.keys()))
        text_col = args.color
        for ms, h, c in zip(df['mass_data'][idx], df['Height'][idx],
                            df['Charge'][idx]):
            matched = np.where(np.abs(labels_ms - ms) < args.labels_eps)[0]
            if matched.size > 0:
                label, text_col = args.labels.get(labels_ms[matched[0]])
                _plot_vlines([ms], [h], text_col, label)
            else:
                text_col = args.color
            ax.text(ms, h, f'* {ms:.2f}({c:d})',
                    fontsize=args.__dict__.get('tag_fontsize', 15), color = text_col)
        plt.yscale('log')
        axis_lim = (1-args.expand, 1+args.expand)
        plt.xlim(df['mass_data'].min() * axis_lim[0], df['mass_data'].max() * axis_lim[1])
        plt.ylim(df['Height'].min() * axis_lim[0], df['Height'].max() * axis_lim[1])
        ax.set_title(f'{name} (Peak List of TOF MS)', fontsize=25)
        ax.set_xlabel(f'Mass{"" if args.mass else "/charge"}', fontsize=25)
        ax.set_ylabel('Intensity (cps)', fontsize=25)
        
    @staticmethod
    def plot_masscharge(name: str, df: pd.DataFrame, args, ax):
        # find peaks
        peaks_cache_path = os.path.join(args.output, f'{name} peaks.cache.npy')
        if args.use_peaks_cache and os.path.exists(peaks_cache_path):
            peaks = np.load(peaks_cache_path)
            print(f'loaded peaks from cache: {peaks_cache_path}')
        else:
            print('searching peaks...')
            peaks = scipy.signal.find_peaks_cwt(df['Intensity'], args.min_peak_width)
            np.save(peaks_cache_path, peaks)
        # filter peaks
        if peaks.any():
            df = df.iloc[peaks, :]
        if args.xlim:
            xlim = [float(i) for i in args.xlim.split(',')]
            if xlim[0] > xlim[1]:
                xlim = xlim[::-1]
                put_err('x-axis limit error, xlim should be in ascending order, change it to [min, max]')
            df = df[(df['Mass/Charge'] >= xlim[0]) & (df['Mass/Charge'] <= xlim[1])]
            print(f'x-axis data limit set to {xlim}')
        min_height = df['Intensity'].max() * args.min_height_percent / 100
        df = df[(df['Intensity'] >= min_height) & (df['Intensity'] >= args.min_height)]
        print(f'min-height set to {min_height}')
        print(f'searching done. {len(df)} peaks found.')
        df.to_csv(os.path.join(args.output, f'{name} {df._attrs["content_type"]}.csv'))
        # plot
        _plot_vlines(df['Mass/Charge'], df['Intensity'], args.color)
        labels_ms = np.array(list(args.labels.keys()))
        text_col = args.color
        for ms, h in zip(df['Mass/Charge'], df['Intensity']):
            matched = np.where(np.abs(labels_ms - ms) < args.labels_eps)[0]
            if matched.size > 0:
                label, text_col = args.labels.get(labels_ms[matched[0]])
                _plot_vlines([ms], [h], text_col, label)
            else:
                text_col = args.color
            ax.text(ms, h, f'* {ms:.2f}',
                    fontsize=args.__dict__.get('tag_fontsize', 15), color = text_col)
        # fix style
        plt.yscale('log')
        axis_lim = (1-args.expand, 1+args.expand)
        plt.xlim(df['Mass/Charge'].min() * axis_lim[0], df['Mass/Charge'].max() * axis_lim[1])
        plt.ylim(df['Intensity'].min() * axis_lim[0], df['Intensity'].max() * axis_lim[1])
        ax.set_title(f'{name} (Mass/Charge of TOF MS)', fontsize=25)
        ax.set_xlabel(f'Mass/Charge', fontsize=25)
        ax.set_ylabel('Intensity (cps)', fontsize=25)
    
    def main_process(self):
        dfs = self.load_data(self.args.dir, self.args.recursive)
        # show data general info and output peak list DataFrame
        for n,df in dfs.items():
            path = Path(n).resolve()
            name = path.stem
            if self.use_recursive_output:
                self.args.output = str(path.parent)
            print(f'\n\n\n\n\n{name}: {df._attrs["content_type"]}:\n', df)
            df.to_csv(os.path.join(self.args.output, f'{name} {df._attrs["content_type"]}.csv'))
            # process Mass (charge) and identify mass
            if df._attrs['content_type'] == 'peak list':
                df = self.filter_peaklist_data(df, self.args)
            if not df.empty:
                # plot each df
                print(f'plotting {name}: {df._attrs["content_type"]}')
                fig, ax = plt.subplots(figsize=(10, 6))
                if df._attrs["content_type"] == 'base peak':
                    self.plot_basepeak(name, df, self.args, ax)
                elif df._attrs["content_type"] == 'absorbance':
                    self.plot_absorbance(name, df, self.args, ax)
                elif df._attrs["content_type"] == 'peak list': # avoid drop all data but still draw
                    self.plot_peaklist(name, df, self.args, ax)
                elif df._attrs["content_type"] =='mass-charge':
                    self.plot_masscharge(name, df, self.args, ax)
                # style adjust and save
                plt.xticks(size = 20);plt.yticks(size = 20)
                plt.legend(fontsize=15, loc = self.args.legend_pos,
                           bbox_to_anchor = (self.args.legend_pos_bbox1, self.args.legend_pos_bbox2),
                           draggable = True)
                save_show(os.path.join(self.args.output, f'{name} {df._attrs["content_type"]}.png'),
                          dpi = 600, show = self.args.show_fig)
            else:
                print(f'no data left after filtering, skip {name}: {df._attrs["content_type"]}')
            

class explore_mass(plot_mass):
    from nicegui import ui
    def __init__(self, args: argparse.Namespace, printf=print) -> None:
        super().__init__(args, printf)
        self.labels_string = args.labels
        self.fig = None
        
    @ui.refreshable
    def make_fig(self):
        from nicegui import ui
        plt.close(self.fig)
        with ui.pyplot(figsize=(self.args.fig_w, self.args.fig_h), close = False) as fig:
            # process labels
            self.args.labels = self.process_labels(self.args.labels_string)
            # process io path
            if self.use_recursive_output:
                self.args.output = os.path.dirname(self.args.now_name)
            df = self.args.dfs[self.args.now_name].copy() # avoid modify original data
            name = Path(self.args.now_name).resolve().stem # same as plot-mass
            # plot
            print(f'plotting {name}: {df._attrs["content_type"]}')
            ax = fig.fig.gca()
            if df._attrs["content_type"] == 'peak list': # avoid drop all data but still draw
                df = self.filter_peaklist_data(df, self.args)
                if not df.empty:
                    self.plot_peaklist(name, df, self.args, ax)
                else:
                    ui.notify(f'no data left after filtering, skip {name}: {df._attrs["content_type"]}')
            elif df._attrs["content_type"] =='mass-charge':
                self.plot_masscharge(name, df, self.args, ax)
            # fix style
            plt.xticks(size = self.args.xticks_fontsize)
            plt.yticks(size = self.args.yticks_fontsize)
            plt.title(self.args.title, fontsize=self.args.title_fontsize)
            plt.xlabel(self.args.xlabel, fontsize=self.args.axis_label_fontsizes)
            plt.ylabel(self.args.ylabel, fontsize=self.args.axis_label_fontsizes)
            plt.legend(fontsize=self.args.legend_fontsize, loc=self.args.legend_pos,
                       bbox_to_anchor=(self.args.legend_pos_bbox1, self.args.legend_pos_bbox2), draggable = True)
        self.fig = fig.fig
        
    def main_process(self):
        from nicegui import app, ui
        from mbapy.game import BaseInfo
        # process args and load data
        dfs = self.load_data(self.args.dir, self.args.recursive)
        dfs = {k:v for k,v in dfs.items() if v is not None and v._attrs['content_type'] in ['peak list','mass-charge']}
        if not dfs:
            raise FileNotFoundError(f'can not find peak-list or mass-charge txt files in {self.args.dir}')
        # make global settings
        self.args = BaseInfo(now_name = list(dfs.keys())[0], dfs = dfs,
                            #  instance_refresh = False, # instance refresh not implemented yet
                            labels_string = self.labels_string,
                            title = '', xlabel = 'Mass/Charge', ylabel = 'Intensity (cps)',
                            xticks_fontsize = 20, yticks_fontsize = 20, tag_fontsize = 15,
                            axis_label_fontsizes = 25, title_fontsize = 25, legend_fontsize = 15,
                            fig_w = 10, fig_h = 8, dpi = 600, file_name = '',
                            **self.args.__dict__)
        # GUI
        with ui.header(elevated=True).style('background-color: #3874c8'):
            ui.label('mbapy-cli mass | Mass Data Explorer').classes('text-h4')
            ui.space()
            # ui.checkbox('Instance refresh', value=self.args.instance_refresh).bind_value_to(self.args, 'instance_refresh')
            ui.button('Refresh', on_click=self.make_fig.refresh, icon='refresh').props('no-caps')
            ui.button('Save', on_click=partial(save_show, path = self.args.file_name, dpi = self.args.dpi, show = self.args.show_fig), icon='save').props('no-caps')
            ui.button('Show', on_click=plt.show, icon='open_in_new').props('no-caps')
            ui.button('Exit', on_click=app.shutdown, icon='power')
        with ui.splitter(value = 20).classes('w-full h-full h-56') as splitter:
            with splitter.before:
                df_short_names = [n.replace(os.path.join(self.args.dir, ''), '') for n in dfs]
                with ui.tabs(value = df_short_names[0]).props('vertical').classes('h-full') as tabs:
                    df_tabs = [ui.tab(n).props('no-caps') for n in df_short_names]
                tabs.bind_value_to(self.args, 'now_name', lambda short_name: os.path.join(self.args.dir, short_name))
                # tabs.on_value_change(self.make_fig) # instance refresh not implemented yet
            with splitter.after:
                with ui.row().classes('w-full h-full'):
                    with ui.card():
                        # data filtering configs
                        ui.label('Data Filtering').classes('text-h6')
                        ui.checkbox('use peaks cache', value=self.args.use_peaks_cache).bind_value_to(self.args, 'use_peaks_cache')
                        ui.checkbox('filter by mass', value=self.args.mass).bind_value_to(self.args,'mass')
                        ui.number('min peak width', value=self.args.min_peak_width, min = 1).bind_value_to(self.args,'min_peak_width')
                        ui.number('min height', value=self.args.min_height, min = 0).bind_value_to(self.args, 'min_height')
                        ui.number('min height percent', value=self.args.min_height_percent, min = 0, max = 100).bind_value_to(self.args,'min_height_percent').classes('w-full')
                        ui.input('xlim', value=self.args.xlim).bind_value_to(self.args, 'xlim')
                        # configs for fontsize
                        ui.label('Configs for Fontsize').classes('text-h6')
                        ui.number('xticks fontsize', value=self.args.xticks_fontsize, min=0, step=0.5, format='%.1f').bind_value_to(self.args, 'xticks_fontsize')
                        ui.number('yticks fontsize', value=self.args.yticks_fontsize, min=0, step=0.5, format='%.1f').bind_value_to(self.args, 'yticks_fontsize')
                        ui.number('title fontsize', value=self.args.title_fontsize, min=0, step=0.5, format='%.1f').bind_value_to(self.args, 'title_fontsize')
                        ui.number('axis label fontsize', value=self.args.axis_label_fontsizes, min=0, step=0.5, format='%.1f').bind_value_to(self.args, 'axis_label_fontsizes')
                        ui.number('tag fontsize', value=self.args.tag_fontsize, min=0, step=0.5, format='%.1f').bind_value_to(self.args, 'tag_fontsize')
                        ui.input('title', value=self.args.title).bind_value_to(self.args, 'title')
                        ui.input('xlabel', value=self.args.xlabel).bind_value_to(self.args, 'xlabel')
                        ui.input('ylabel', value=self.args.ylabel).bind_value_to(self.args, 'ylabel')
                    with ui.card():
                        # configs for legend
                        ui.label('Configs for Legend').classes('text-h6')
                        ui.textarea('labels', value=self.args.labels_string).bind_value_to(self.args, 'labels_string').props('clearable')
                        ui.number('labels eps', value=self.args.labels_eps, min=0, step=0.1, format='%.1f').bind_value_to(self.args, 'labels_eps')
                        ui.number('legend fontsize', value=self.args.legend_fontsize, min=0, step=0.5, format='%.1f').bind_value_to(self.args, 'legend_fontsize')
                        ui.input('legend loc', value=self.args.legend_pos).bind_value_to(self.args, 'legend_pos')
                        ui.number('bbox1', value=self.args.legend_pos_bbox1, min=0, step=0.1, format='%.1f').bind_value_to(self.args, 'legend_pos_bbox1')
                        ui.number('bbox2', value=self.args.legend_pos_bbox2, min=0, step=0.1, format='%.1f').bind_value_to(self.args, 'legend_pos_bbox2')
                        # configs for saving
                        ui.label('Configs for Saving').classes('text-h6')
                        ui.checkbox('show figure', value=self.args.show_fig).bind_value_to(self.args,'show_fig')
                        ui.number('axis expand', value=self.args.expand, min=0, max=1, step=0.01, format='%.3f').bind_value_to(self.args, 'expand')
                        ui.number('figure width', value=self.args.fig_w, min=1, step=0.5, format='%.1f').bind_value_to(self.args, 'fig_w')
                        ui.number('figure height', value=self.args.fig_h, min=1, step=0.5, format='%.1f').bind_value_to(self.args, 'fig_h')
                        ui.number('DPI', value=self.args.dpi, min=1, step=1, format='%d').bind_value_to(self.args, 'dpi')
                        ui.input('figure file name', value=self.args.file_name).bind_value_to(self.args, 'file_name')
                    with ui.card():
                        ui.label(f'{df_short_names[0]}').classes('text-h6').bind_text_from(tabs, 'value')
                        self.make_fig()
        ## run GUI
        ui.run(host = 'localhost', port = 8010, title = 'Mass Data Explorer', reload=False)
        

_str2func = {
    'plot-mass': plot_mass,
    'explore-mass': explore_mass,
}


def main(sys_args: List[str] = None):
    args_paser = argparse.ArgumentParser()
    subparsers = args_paser.add_subparsers(title='subcommands', dest='sub_command')
    
    plot_mass_args = subparsers.add_parser('plot-mass', description='plot mass spectrum')
    # set dir argument
    plot_mass_args.add_argument("-d", "--dir", type = str, default='.',
                                help="txt file directory, default is %(default)s")
    plot_mass_args.add_argument('-r', '--recursive', action='store_true', default=False,
                                help='search input directory recursively, default is %(default)s')
    # set output file argument
    plot_mass_args.add_argument("-o", "--output", type = str, default=None,
                                help="output file dir or path, default is %(default)s, means same as input dir")
    # set draw argument
    plot_mass_args.add_argument('--use-peaks-cache', action='store_true', default=False,
                                help='use peaks cache to speed up plot, default is %(default)s')
    plot_mass_args.add_argument('-m', '--mass', action='store_true', default=False,
                                help='draw Mass instead of Mass/charge which is Mass+z, default is %(default)s')
    plot_mass_args.add_argument('-min', '--min-height', type = int, default=0,
                                help='filter data with min height in peak list plot, default is %(default)s')
    plot_mass_args.add_argument('-minp', '--min-height-percent', type = float, default=10,
                                help='filter data with min height percent to hightest in mass/charge plot, default is %(default)s')
    plot_mass_args.add_argument('--min-peak-width', type = float, default=4,
                                help='filter peaks with min width in Mass/Charge plot, default is %(default)s')
    plot_mass_args.add_argument('-xlim', type = str, default=None,
                                help='set x-axis limit, input as "200,2000", default is %(default)s')
    plot_mass_args.add_argument('-col', '--color', type = str, default='black',
                                help='draw color, default is %(default)s')
    plot_mass_args.add_argument('-labels', '--labels', type = str, default='',
                                help='labels, input as 1000,Pep1;1050,Pep2, default is %(default)s')
    plot_mass_args.add_argument('--labels-eps', type = float, default=0.5,
                                help='eps to recognize labels, default is %(default)s')
    plot_mass_args.add_argument('-expand', '--expand', type = float, default=0.2,
                                help='how much the x-axis and y-axisto be expanded, default is %(default)s')
    plot_mass_args.add_argument('-lpos', '--legend-pos', type = str, default='upper center',
                                help='legend position, can be string as "upper center", or be float as 0.1,0.2, default is %(default)s')
    plot_mass_args.add_argument('-lposbbox1', '--legend-pos-bbox1', type = float, default=1.2,
                                help='legend position bbox 1 to anchor, default is %(default)s')
    plot_mass_args.add_argument('-lposbbox2', '--legend-pos-bbox2', type = float, default=1,
                                help='legend position bbox 2 to anchor, default is %(default)s')
    plot_mass_args.add_argument('-sf', '--show-fig', action='store_true', default=False,
                                help='automatically show figure, default is %(default)s')

    explore_mass_args = subparsers.add_parser('explore-mass', description='explore mass spectrum data')
    for action in plot_mass_args._actions:
        if action.dest == 'use_peaks_cache':
            action.default = True
        if action.dest not in ['help']:
            explore_mass_args._add_action(action)

    excute_command(args_paser, sys_args, _str2func)


if __name__ in {"__main__", "__mp_main__"}:
    # dev code, MUST COMMENT OUT BEFORE RELEASE
    # main('explore-mass -d data_tmp/scripts/mass'.split())
    
    main()