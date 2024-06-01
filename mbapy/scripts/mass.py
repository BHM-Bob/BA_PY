import argparse
import itertools
import os
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

os.environ['MBAPY_AUTO_IMPORT_TORCH'] = 'False'
os.environ['MBAPY_FAST_LOAD'] = 'True'

from mbapy.base import Configs, put_err
from mbapy.file import get_paths_with_extension, get_valid_file_path
from mbapy.plot import get_palette, save_show
from mbapy.sci_instrument.mass import MassData, SciexOriData, SciexPeakListData
from mbapy.sci_instrument.mass import plot_mass as _plot_mass
from mbapy.sci_instrument.mass import process_peak_labels
from mbapy.scripts._script_utils_ import Command, clean_path, excute_command
from mbapy.web import TaskPool


class plot_mass(Command):
    def __init__(self, args: argparse.Namespace, printf=print) -> None:
        super().__init__(args, printf)
        self.task_pool: TaskPool = None
        self.dfs: Dict[str, MassData] = {}
        self.SUPPORT_SYS: Dict[str, MassData] = {'SCIEX-PeakList': SciexPeakListData, 'SCIEX-Ori': SciexOriData}
    
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
        # process others
        self.args.labels = process_peak_labels(self.args.labels)
        self.args.xlim = eval(f'[{self.args.xlim}]') if self.args.xlim else None
        
    @staticmethod
    def filter_peaklist_data(df: pd.DataFrame, args):
        df['mass_data'] = df['Mass (charge)'] if args.mass else df['Mass/charge (charge)']
        drop_idx = df[df['Height'] < args.min_height].index
        if not drop_idx.empty:
            print(f'drop data with min-height: {args.min_height} and only these data remained:\n',
                df[df['Height'] >= args.min_height])
            df.drop(drop_idx, axis = 0, inplace = True)
        return df
    
    def load_suffix_data(self, dir: str, suffix: List[str], recursive):
        for path in get_paths_with_extension(dir, suffix, recursive):
            path_obj = Path(path)
            if path_obj.stem in self.dfs:
                print(f'{path} already loaded as {type(self.dfs[path_obj.stem])} data type, skip')
                continue
            for n, model in self.SUPPORT_SYS.items():
                if path_obj.suffix in model.DATA_FILE_SUFFIX:
                    tmp_err_warning_level = Configs.err_warning_level
                    Configs.err_warning_level = 1
                    data = model(path)
                    Configs.err_warning_level = tmp_err_warning_level
                    if data.SUCCEED_LOADED:
                        print(f'loaded {path} as {n} data type')
                        self.dfs[data.get_tag()] = data
                        break
            
    def load_data(self, dir: str, recursive: bool):
        suffixs_r = list(set([model.RECOMENDED_DATA_FILE_SUFFIX for model in self.SUPPORT_SYS.values()]))
        suffixs = list(set(itertools.chain(*[model.DATA_FILE_SUFFIX for model in self.SUPPORT_SYS.values()])) - set(suffixs_r))
        self.load_suffix_data(dir, suffixs_r, recursive)
        self.load_suffix_data(dir, suffixs, recursive)
        if not self.dfs:
            raise FileNotFoundError(f'can not find txt files in {dir}')
        return self.dfs
    
    def main_process(self):
        self.load_data(self.args.dir, self.args.recursive)
        if self.args.multi_process > 1:
            self.task_pool = TaskPool('process', self.args.multi_process).run()
            print(f'created task pool with {self.args.multi_process} processes')
        # show data general info and output peak list DataFrame
        for n, data in self.dfs.items():
            if data.peak_df is None or data.check_processed_data_empty(data.peak_df):
                print(f'{n}: search peaks...')
                data.search_peaks(self.args.xlim, self.args.min_peak_width,
                                  self.task_pool, self.args.multi_process)
                data.filter_peaks(self.args.xlim, self.args.min_height, self.args.min_height_percent)
            data.save_processed_data()
            print(f'{n}: processed data saved to {data.processed_data_path}')
            _plot_mass(data, xlim=self.args.xlim, labels=self.args.labels, labels_eps=self.args.labels_eps)
            # style adjust and save
            plt.xticks(size = 20);plt.yticks(size = 20)
            png_path = Path(data.data_file_path).with_suffix('.png')
            save_show(png_path, dpi = 600, show = self.args.show_fig)
            print(f'{n}: plot saved to {png_path}')
        self.task_pool.close()


class explore_mass(plot_mass):
    from nicegui import ui
    def __init__(self, args: argparse.Namespace, printf=print) -> None:
        super().__init__(args, printf)
        self.labels_string = args.labels
        self.fig = None
        self._expansion = []
        self._bbox_extra_artists = None
        
    def _ui_only_one_expansion(self, e):
        if e.value:
            for expansion in self._expansion:
                if expansion != e.sender:
                    expansion.value = False
        
    @ui.refreshable
    def make_fig(self):
        from nicegui import ui
        plt.close(self.fig)
        with ui.pyplot(figsize=(self.args.fig_w, self.args.fig_h), close = False) as fig:
            # process args
            self.args.labels = process_peak_labels(self.args.labels_string)
            self.args.xlim = eval(f'[{self.args.xlim}]') if isinstance(self.args.xlim, str) else self.args.xlim
            # plot
            ax = fig.fig.gca()
            data = self.args.dfs[self.args.now_name]
            ## choose mass data
            if self.args.mass and data.X_M_HEADER is not None:
                data.X_HEADER = data.X_M_HEADER
            elif not self.args.mass:
                data.X_HEADER = data.X_MZ_HEADER
            ## search peaks
            if data.peak_df is None or data.check_processed_data_empty(data.peak_df):
                ui.notify(f'{data.get_tag()}: search peaks...')
                data.search_peaks(self.args.xlim, self.args.min_peak_width,
                                  self.task_pool, self.args.multi_process)
            tmp_data = deepcopy(data) # filter from original data
            tmp_data.filter_peaks(self.args.xlim, self.args.min_height, self.args.min_height_percent)
            ## plot
            ax, self._bbox_extra_artists = _plot_mass(tmp_data, ax = ax, xlim=self.args.xlim,
                                                      labels=self.args.labels, labels_eps=self.args.labels_eps,
                                                      legend_bbox=(self.args.legend_pos_bbox1, self.args.legend_pos_bbox2),
                                                      legend_pos=self.args.legend_pos)
            x_axis_exp = (1-self.args.xaxis_expand, 1+self.args.xaxis_expand)
            y_axis_exp = (1-self.args.yaxis_expand, 1+self.args.yaxis_expand)
            plt.xlim(tmp_data.peak_df[tmp_data.X_HEADER].min() * x_axis_exp[0], tmp_data.peak_df[tmp_data.X_HEADER].max() * x_axis_exp[1])
            plt.ylim(tmp_data.peak_df[tmp_data.Y_HEADER].min() * y_axis_exp[0], tmp_data.peak_df[tmp_data.Y_HEADER].max() * y_axis_exp[1])
            plt.xticks(size = self.args.xticks_fontsize)
            plt.yticks(size = self.args.yticks_fontsize)
            plt.title(self.args.title, fontsize=self.args.title_fontsize)
            plt.xlabel(self.args.xlabel, fontsize=self.args.axis_label_fontsizes)
            plt.ylabel(self.args.ylabel, fontsize=self.args.axis_label_fontsizes)
            self.fig = fig.fig
        
    def save_fig(self):
        from nicegui import ui
        png_path = Path(self.args.dfs[self.args.now_name].data_file_path).with_suffix('.png')
        ui.notify(f'saving figure to {png_path}')
        save_show(png_path, dpi = self.args.dpi, show = self.args.show_fig)
        
    def main_process(self):
        from nicegui import app, ui

        from mbapy.game import BaseInfo

        # set task pool
        if self.args.multi_process > 1:
            self.task_pool = TaskPool('process', self.args.multi_process).run()
            print(f'created task pool with {self.args.multi_process} processes')
        # process args and load data
        self.load_data(self.args.dir, self.args.recursive)
        self.dfs = {data.data_file_path:data for data in self.dfs.values()}
        if not self.dfs:
            raise FileNotFoundError(f'can not find peak-list or mass-charge txt files in {self.args.dir}')
        # make global settings
        self.args = BaseInfo(now_name = list(self.dfs.keys())[0], dfs = self.dfs,
                            #  instance_refresh = False, # instance refresh not implemented yet
                            labels_string = self.labels_string,
                            title = '', xlabel = 'Mass/Charge', ylabel = 'Intensity (cps)',
                            xticks_fontsize = 20, yticks_fontsize = 20, tag_fontsize = 15,
                            axis_label_fontsizes = 25, title_fontsize = 25, legend_fontsize = 15,
                            fig_w = 10, fig_h = 8, dpi = 600, file_name = '',
                            xaxis_expand = 0.2, yaxis_expand = 0.1,
                            legend_pos_bbox1 = 1.0, legend_pos_bbox2 = 1.0, legend_pos = 'upper right',
                            **self.args.__dict__)
        # GUI
        with ui.header(elevated=True).style('background-color: #3874c8'):
            ui.label('mbapy-cli mass | Mass Data Explorer').classes('text-h4')
            ui.space()
            # ui.checkbox('Instance refresh', value=self.args.instance_refresh).bind_value_to(self.args, 'instance_refresh')
            ui.button('Refresh', on_click=self.make_fig.refresh, icon='refresh').props('no-caps')
            ui.button('Save', on_click=self.save_fig, icon='save').props('no-caps')
            ui.button('Show', on_click=plt.show, icon='open_in_new').props('no-caps')
            ui.button('Exit', on_click=app.shutdown, icon='power')
        with ui.splitter(value = 15).classes('w-full h-full h-56') as splitter:
            with splitter.before:
                df_short_names = [n.replace(os.path.join(self.args.dir, ''), '') for n in self.dfs]
                with ui.tabs(value = df_short_names[0]).props('vertical').classes('w-full h-full') as tabs:
                    df_tabs = [ui.tab(n).props('no-caps').classes('w-full') for n in df_short_names]
                tabs.bind_value_to(self.args, 'now_name', lambda short_name: os.path.join(self.args.dir, short_name))
                # tabs.on_value_change(self.make_fig) # instance refresh not implemented yet
            with splitter.after:
                with ui.row().classes('w-full h-full'):
                    with ui.column().classes('h-full'):
                        # data filtering configs
                        with ui.expansion('Data Filtering', icon='filter_alt', value=True, on_value_change=self._ui_only_one_expansion) as expansion1:
                            self._expansion.append(expansion1)
                            ui.checkbox('use peaks cache', value=self.args.use_peaks_cache).bind_value_to(self.args, 'use_peaks_cache')
                            ui.checkbox('filter by mass', value=self.args.mass).bind_value_to(self.args, 'mass')
                            ui.number('min height', value=self.args.min_height, min = 0).bind_value_to(self.args, 'min_height')
                            ui.number('min height percent', value=self.args.min_height_percent, min = 0, max = 100).bind_value_to(self.args,'min_height_percent').classes('w-full')
                            ui.input('xlim', value=self.args.xlim).bind_value_to(self.args, 'xlim')
                        # configs for fontsize
                        with ui.expansion('Configs for Fontsize', icon='format_size', on_value_change=self._ui_only_one_expansion) as expansion2:
                            self._expansion.append(expansion2)
                            ui.number('xticks fontsize', value=self.args.xticks_fontsize, min=0, step=0.5, format='%.1f').bind_value_to(self.args, 'xticks_fontsize')
                            ui.number('yticks fontsize', value=self.args.yticks_fontsize, min=0, step=0.5, format='%.1f').bind_value_to(self.args, 'yticks_fontsize')
                            ui.number('title fontsize', value=self.args.title_fontsize, min=0, step=0.5, format='%.1f').bind_value_to(self.args, 'title_fontsize')
                            ui.number('axis label fontsize', value=self.args.axis_label_fontsizes, min=0, step=0.5, format='%.1f').bind_value_to(self.args, 'axis_label_fontsizes')
                            ui.number('tag fontsize', value=self.args.tag_fontsize, min=0, step=0.5, format='%.1f').bind_value_to(self.args, 'tag_fontsize')
                            ui.input('title', value=self.args.title).bind_value_to(self.args, 'title')
                            ui.input('xlabel', value=self.args.xlabel).bind_value_to(self.args, 'xlabel')
                            ui.input('ylabel', value=self.args.ylabel).bind_value_to(self.args, 'ylabel')
                        # configs for legend
                        with ui.expansion('Configs for Legend', icon='more', on_value_change=self._ui_only_one_expansion) as expansion3:
                            self._expansion.append(expansion3)
                            ui.textarea('labels', value=self.args.labels_string).bind_value_to(self.args, 'labels_string').props('clearable')
                            ui.number('labels eps', value=self.args.labels_eps, min=0, step=0.1, format='%.6f').bind_value_to(self.args, 'labels_eps')
                            ui.number('legend fontsize', value=self.args.legend_fontsize, min=0, step=0.5, format='%.1f').bind_value_to(self.args, 'legend_fontsize')
                            ui.input('legend loc', value=self.args.legend_pos).bind_value_to(self.args, 'legend_pos')
                            with ui.row().classes('w-full'):
                                ui.number('bbox1', value=self.args.legend_pos_bbox1, min=0, step=0.1, format='%.3f').bind_value_to(self.args, 'legend_pos_bbox1')
                                ui.number('bbox2', value=self.args.legend_pos_bbox2, min=0, step=0.1, format='%.3f').bind_value_to(self.args, 'legend_pos_bbox2')
                        # configs for saving
                        with ui.expansion('Configs for Saving', icon='save', on_value_change=self._ui_only_one_expansion) as expansion4:
                            self._expansion.append(expansion4)
                            ui.checkbox('show figure', value=self.args.show_fig).bind_value_to(self.args,'show_fig')
                            with ui.row().classes('w-full'):
                                ui.number('x axis expand', value=self.args.xaxis_expand, min=0, max=1, step=0.01, format='%.3f').classes('w-2/5').bind_value_to(self.args, 'xaxis_expand')
                                ui.number('y axis expand', value=self.args.yaxis_expand, min=0, max=1, step=0.01, format='%.3f').classes('w-2/5').bind_value_to(self.args, 'yaxis_expand')
                            ui.number('figure width', value=self.args.fig_w, min=1, step=0.5, format='%.3f').bind_value_to(self.args, 'fig_w')
                            ui.number('figure height', value=self.args.fig_h, min=1, step=0.5, format='%.3f').bind_value_to(self.args, 'fig_h')
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
    plot_mass_args.add_argument('-minp', '--min-height-percent', type = float, default=1,
                                help='filter data with min height percent to hightest in mass/charge plot, default is %(default)s')
    plot_mass_args.add_argument('--min-peak-width', type = float, default=4,
                                help='filter peaks with min width in Mass/Charge plot, default is %(default)s')
    plot_mass_args.add_argument('-xlim', type = str, default=None,
                                help='set x-axis limit, input as "200,2000", default is %(default)s')
    plot_mass_args.add_argument('-col', '--color', type = str, default='black',
                                help='draw color, default is %(default)s')
    plot_mass_args.add_argument('-labels', '--labels', type = str, default='',
                                help='labels, input as 1000,Pep1,red;1050,Pep2, default is %(default)s')
    plot_mass_args.add_argument('--labels-eps', type = float, default=0.5,
                                help='eps to recognize labels, default is %(default)s')
    plot_mass_args.add_argument('-sf', '--show-fig', action='store_true', default=False,
                                help='automatically show figure, default is %(default)s')
    plot_mass_args.add_argument('-mp', '--multi-process', type = int, default=4,
                                help='multi-process to speed up plot, default is %(default)s')

    explore_mass_args = subparsers.add_parser('explore-mass', description='explore mass spectrum data')
    for action in plot_mass_args._actions:
        if action.dest == 'use_peaks_cache':
            action.default = True
        if action.dest not in ['help']:
            explore_mass_args._add_action(action)

    if __name__ in ['__main__', 'mbapy.scripts.mass']:
        # '__main__' is debug, 'mbapy.scripts.mass' is user running
        excute_command(args_paser, sys_args, _str2func)

if __name__ in {"__main__", "__mp_main__"}:
    # dev code, MUST COMMENT OUT BEFORE RELEASE
    # main('explore-mass -d data_tmp/scripts/mass'.split())
    
    main()