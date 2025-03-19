import argparse
import os
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import matplotlib.pyplot as plt
import pandas as pd
from nicegui import app, ui, run

os.environ['MBAPY_AUTO_IMPORT_TORCH'] = 'False'
os.environ['MBAPY_FAST_LOAD'] = 'True'

from mbapy.base import get_fmt_time, get_storage_path, put_err
from mbapy.file import (decode_bits_to_str, get_paths_with_extension,
                        get_valid_file_path, write_sheets)
from mbapy.game import BaseInfo
from mbapy.plot import get_palette, save_show
from mbapy.sci_instrument.hplc import (EasychromData, HplcData, SciexData,
                                       SciexTicData, WatersData, WatersPdaData)
from mbapy.sci_instrument.hplc._utils import plot_hplc as _plot_hplc
from mbapy.sci_instrument.hplc._utils import \
    plot_pda_heatmap as _plot_pda_heatmap
from mbapy.sci_instrument.hplc._utils import (process_file_labels,
                                              process_peak_labels)
from mbapy.scripts._script_utils_ import Command, clean_path, excute_command
from mbapy.web_utils.task import TaskPool


class plot_hplc(Command):
    SUPPORT_SYSTEMS = {'waters', 'SCIEX', 'SCIEX-TIC', 'EasyChrom', 'WatersPDA'}
    def __init__(self, args: argparse.Namespace, printf=print) -> None:
        super().__init__(args, printf)
        self.dfs = {}
        self.sys2suffix = {'waters': 'arw', 'SCIEX': 'txt', 'SCIEX-TIC': 'txt', 'EasyChrom': 'txt', 'WatersPDA': 'arw'}
        self.sys2model: Dict[str, HplcData] = {'waters': WatersData, 'WatersPDA': WatersPdaData,
                                               'SCIEX': SciexData, 'SCIEX-TIC': SciexTicData,
                                               'EasyChrom': EasychromData}
        
    @staticmethod
    def make_args(args: argparse.ArgumentParser):
        args.add_argument('-i', '--input', type = str, default='.',
                          help="data file directory, default is %(default)s.")
        args.add_argument('-s', '--system', type = str, default='waters',
                          help=f"HPLC system. Default is %(default)s, those systems are supported: {', '.join(list(plot_hplc.SUPPORT_SYSTEMS))}")
        args.add_argument('-r', '--recursive', action='store_true', default=False,
                          help='search input directory recursively, default is %(default)s.')
        args.add_argument('-merge', action='store_true', default=False,
                          help='merge multi files into one plot, default is %(default)s.')
        args.add_argument('-o', '--output', type = str, default=None,
                          help="output file dir or path. Default is %(default)s, means same as input dir")
        # set draw argument
        args.add_argument('--pda-wave-length', type=float, default=228,
                          help='set pda wave length, default is %(default)s.')
        args.add_argument('--min-peak-width', type = float, default=4,
                          help='filter peaks with min width in hplc/Charge plot, default is %(default)s.')
        args.add_argument('-xlim', type = str, default='0,None',
                          help='set x-axis limit, input as "0,15", default is %(default)s.')
        args.add_argument('-flabels', '--file-labels', type = str, default='',
                          help='labels, input as 228,blue;304,red, default is %(default)s.')
        args.add_argument('-lpos', '--file-legend-pos', type = str, default='upper right',
                          help='legend position, can be string as "upper center", default is %(default)s')
        args.add_argument('-lposbbox', '--file-legend-bbox', type = str, default='1,1',
                          help='legend position bbox 1 to anchor, default is %(default)s')
        args.add_argument('-dpi', type = int, default=600,
                          help='set dpi of output image, default is %(default)s.')
        args.add_argument('-show', action='store_true', default=False,
                          help='show plot window, default is %(default)s.')
        return args

    def load_dfs_from_data_file(self) -> Dict[str, HplcData]:
        if os.path.isdir(self.args.input):
            paths = get_paths_with_extension(self.args.input, [self.sys2suffix[self.args.system]], recursive=self.args.recursive)
        else:
            paths = [str(self.args.input)]
        dfs = [self.data_model(path) for path in paths]
        dfs = [(data.get_tag(), data) for data in dfs if data.SUCCEED_LOADED]
        return dfs

    def process_args(self):
        assert self.args.system in self.SUPPORT_SYSTEMS, f'not support HPLC system: {self.args.system}'
        # process self.args
        self.args.input = clean_path(self.args.input)
        self.args.output = clean_path(self.args.output) if self.args.output else self.args.input
        if not os.path.isdir(self.args.output):
            print(f'given output {self.args.output} is a file, change it to parent dir')
            self.args.output = self.args.output.parent
        self.args.file_legend_bbox = eval(f'({self.args.file_legend_bbox})') # NOTE: can be invoked
        self.data_model = self.sys2model[self.args.system]
        # file labels
        self.args.file_labels = process_file_labels(self.args.file_labels)

    def main_process(self):
        def _save_fig(root, name, dpi, show, bbox_extra_artists):
            path = get_valid_file_path(os.path.join(root, name))
            print(f'saving plot to {path}')
            save_show(path, dpi, show=show, bbox_extra_artists = bbox_extra_artists)
        # load origin dfs from data file
        self.dfs = self.load_dfs_from_data_file()
        if not self.dfs:
            raise FileNotFoundError(f'can not find data files in {self.args.input}')
        # show data general info and output peak list DataFrame
        if self.args.merge:
            dfs = list(map(lambda x: x[1], self.dfs))
            ax, extra_artists, _, _ = _plot_hplc(dfs, **self.args.__dict__)
            _save_fig(self.args.output, "merge.png", self.args.dpi, self.args.show, extra_artists)
        else:
            # make file labels again if no file labels given
            if not self.args.file_labels:
                self.args.file_labels = [[n_i, 'black'] for (n_i, data_i) in self.dfs]
            all_file_labels = self.args.file_labels
            delattr(self.args, 'file_labels')
            for i, (tag, data) in enumerate(self.dfs):
                print(f'plotting data for {tag}')
                data.save_processed_data()
                if data.IS_PDA:
                    data.set_opt_wave_length(self.args.pda_wave_length)
                ax, extra_artists, _, _ = _plot_hplc(data, file_labels = [all_file_labels[i]], **self.args.__dict__)
                data_dir = os.path.dirname(data.data_file_path)
                _save_fig(data_dir, f"{tag.replace('/', '-')}.png", self.args.dpi, self.args.show, extra_artists)
                plt.close(ax.figure)


class explore_hplc(plot_hplc):
    def __init__(self, args: argparse.Namespace, printf=print) -> None:
        super().__init__(args, printf)
        self.now_name = ''
        self.fig = None
        self.pda_fig = None
        self.dfs: Dict[str, HplcData] = OrderedDict()
        self.dfs_checkin = {}
        self.dfs_refinment_x = {}
        self.dfs_refinment_y = {}
        self.stored_dfs = {}
        self._expansion = []
        self._bbox_extra_artists = None
        self.is_bind_lim = False
        self.xlim_number_min = None
        self.xlim_number_max = None
        self.xlim_search_number_min = None
        self.xlim_search_number_max = None
        self.files_peaks_idx = {}
        self.tag2label = {}
        self.pda_heatmap_panel = None
        self.area_df_panel = None
        self.area_percent_df_panel = None
        self.all_area_df = None
        self.hc_names: List[str] = []
        self.manual_peak_fig: ui.highchart = None
        self.manual_peaks: Dict[str, List[str, Tuple[float, float], float]] = {} # Dict[uuid: List[tag, point_st(min, abs unit), point_middle, point_ed, area]]
        self.manual_peak_st: List[str, int] = None
        self._manual_peak_table_ui_ele = None
        self._pickle_except_list = ['fig', '_expansion', '_bbox_extra_artists', 'is_bind_lim',
                                    'xlim_number_min', 'xlim_number_max',
                                    'xlim_search_number_min', 'xlim_search_number_max',
                                    'area_df_panel', 'area_percent_df_panel',
                                    'hc_names', 'manual_peaks', 'manual_peak_st']
        
    @staticmethod
    def make_args(args: argparse.ArgumentParser):
        args.add_argument('-i', '--input', type = str, default='.',
                          help="data file directory, default is %(default)s.")
        args.add_argument('-s', '--system', type = str, default='waters',
                          help=f"HPLC system. Default is %(default)s, those systems are supported: {', '.join(list(plot_hplc.SUPPORT_SYSTEMS))}")
        args.add_argument('-url', '--url', type = str, default='localhost',
                          help="url to connect to, default is %(default)s.")
        args.add_argument('-port', '--port', type = int, default=8011,
                          help="port to connect to, default is %(default)s.")
        return args
    
    def process_args(self):
        assert self.args.system in self.SUPPORT_SYSTEMS, f'not support HPLC system: {self.args.system}'
        self.args.input = clean_path(self.args.input)
        self.data_model = self.sys2model[self.args.system]
        
    async def load_data(self, event):
        for name, content in zip(event.names, event.contents):
            if name.endswith(self.sys2suffix[self.args.system]):
                df = self.data_model()
                df.raw_data = df.load_raw_data_from_bytes(content.read())
                df.processed_data = df.process_raw_data()
                df.data_file_path = name
                if df.check_processed_data_empty():
                    ui.notify(f'{name} is not a valid {self.args.system} file, skip')
                    continue
                self.stored_dfs[df.make_tag()] = df
            else:
                ui.notify(f'{name} is not a arw file, skip')
                continue
            ui.notify(f'loaded {name}')
        self.make_tabs.refresh()
        
    def load_data_from_dir(self):
        for (name, dfs) in self.load_dfs_from_data_file():
            self.stored_dfs[name] = dfs
        self.make_tabs.refresh()
        
    def _push_df_from_tabs(self, event):
        if event.value:
            self.dfs[event.sender.text] = self.stored_dfs[event.sender.text]
        else:
            self.dfs.pop(event.sender.text, None)
        self._ui_refinment_numbers.refresh()
        self._ui_plot_params.refresh()
        
    @ui.refreshable
    def make_tabs(self):
        with ui.card().classes('h-full'):
            for name in sorted(self.stored_dfs):
                if name not in self.dfs_checkin:
                    self.dfs_checkin[name] = False
                ui.checkbox(text = name, value = self.dfs_checkin[name],
                            on_change=self._push_df_from_tabs).bind_value_to(self.dfs_checkin, name)
                
    @ui.refreshable
    def make_fig(self):
        plt.close(self.fig)
        if self.dfs:
            for data in self.dfs.values():
                if data.IS_PDA:
                    data.set_opt_wave_length(self.args.pda_wave_length)
            with ui.pyplot(figsize=self.args.fig_size, close=False) as fig:
                self.fig = fig.fig
                for label_name in ['file_labels', 'peak_labels']:
                    if getattr(self.args, label_name) is None:
                        setattr(self.args, label_name, '')
                ax, self._bbox_extra_artists, self.files_peaks_idx, file_labels = \
                    _plot_hplc(list(self.dfs.values()), ax = self.fig.gca(),
                            dfs_refinment_x=self.dfs_refinment_x, dfs_refinment_y=self.dfs_refinment_y,
                            file_label_fn=partial(process_file_labels, file_col_mode=self.args.file_col_mode),
                            peak_label_fn=partial(process_peak_labels, peak_col_mode=self.args.peak_col_mode),
                            **self.args.__dict__)
                self.tag2label = {d.get_tag():l[0] for d,l in zip(self.dfs.values(), file_labels)}
                ax.tick_params(axis='both', which='major', labelsize=self.args.axis_ticks_fontsize)
                ax.set_xlabel(self.args.xlabel, fontsize=self.args.axis_label_fontsize)
                ax.set_ylabel(self.args.ylabel, fontsize=self.args.axis_label_fontsize)
                ax.set_xlim(left=self.args.xlim[0], right=self.args.xlim[1])
                ax.set_ylim(bottom=self.args.ylim[0], top=self.args.ylim[1])
                ax.set_title(self.args.title, fontdict={'fontsize':self.args.title_fontsize})
                plt.tight_layout()
            # update tag2label related ui elements
            self._ui_refinment_numbers.refresh()
            self._ui_plot_params.refresh()
            # re-calcu area df and refreash GUI table
            with ui.tab_panel(self.area_df_panel):
                self.make_area_df.refresh()
            with ui.tab_panel(self.area_percent_df_panel):
                self.make_area_percent_df.refresh()
            # update PDA heatmap fig
            with ui.tab_panel(self.pda_heatmap_panel):
                self.make_pda_heatmap.refresh()
            
    def _ui_transfer_df_to_table(self, df: pd.DataFrame, classes: str = 'w-full'):
        """first column is the name(str), others are time(float), transfer df to nicegui table format"""
        collums = [{'name': 'Name', 'label': 'Name', 'field': 'Name', 'sortable': True}] +\
            [{'name': f'{n:.4f}', 'label': f'{n:.4f}', 'field': f'{n:.4f}', 'sortable': True} for n in df.columns[1:]]
        rows = [{k['field']:(v if isinstance(v, str) else f'{v:.4f}') for k,v in zip(collums, list(line))} for line in df.values]
        return ui.table(columns=collums, rows=rows).classes(classes)
        
    def _make_highcharts_data(self):
        """return series and all options for highcharts"""
        series = [{'name': n, 'marker': {'symbol': 'circle'}, 'data': list(zip(i.get_abs_data()[i.X_HEADER], i.get_abs_data()[i.Y_HEADER]))}\
                    for n, i in self.dfs.items()] +\
                    [{'name': n, 'marker': {'symbol': 'diamond'}, 'showInLegend': False, 'color': 'black',
                    'data': i[1:4]}\
                        for n, i in self.manual_peaks.items()]
        return series, {'title': False,
                'chart': {'zooming': {'type': 'xy'},},
                'tooltip': {'enabled': False},
                'series': series,
                'xAxis': {'title': {'text': self.args.xlabel},},
                'yAxis': {'title': {'text': self.args.ylabel},},}
        
    @ui.refreshable
    def make_manual_peak_table(self):
        all_area_df = pd.DataFrame()
        # gather
        for tag, st, mid, ed, area in self.manual_peaks.values():
            # Dict[uuid: List[tag, point_st(min, abs unit), point_middle, point_ed, area]]
            all_area_df.loc[self.tag2label[tag], mid[0]] = area
        all_area_df = all_area_df.reset_index(drop=False) # so the index column is the first column
        all_area_df.rename(columns={'index': 'Name'}, inplace=True)
        # sort by peak time
        if self.manual_peaks:
            name_col = all_area_df.pop('Name')
            peak_cols = all_area_df.sort_index(axis=1)
            all_area_df = pd.concat([name_col, peak_cols], axis=1)
        # area table
        ui.label(f'All peaks\' area for all files').classes('no-caps')
        self._ui_transfer_df_to_table(all_area_df, classes='w-full')
        # normalize
        ui.label(f'All peaks\' area for all files (normalized)').classes('no-caps')
        for i in range(all_area_df.shape[0]):
            all_area_df.iloc[i, 1:] /= (all_area_df.iloc[i, 1:].sum() / 100)
        return self._ui_transfer_df_to_table(all_area_df, classes='w-full')
            
    def _ui_handle_hc_click(self, e):
        """add or pop one manual peak for self.manual_peaks, call self.make_highcharts_fig to refresh"""
        self.hc_names = list(self.dfs.keys()) + list(self.manual_peaks.keys())
        name = self.hc_names[e.series_index]
        if self.manual_peak_st is None or self.manual_peak_st[0] != name:
            if name in self.manual_peaks:
                ui.notify(f'clear manual peaks for {name}')
                del self.manual_peaks[name]
            else:
                self.manual_peak_st = [name, e.point_index]
                ui.notify(f'add manual peak start at idx={e.point_index} for {name}\nclick middle point to remove', multi_line = True)
                return # wait for next click
        else:
            # check if st > ed, if ture, swap
            if e.point_index <= self.manual_peak_st[1]:
                e.point_index, self.manual_peak_st[1] = self.manual_peak_st[1], e.point_index
            data = self.dfs[name]
            st, ed, area, _, _, underline_y, _ = data.calcu_single_peak_area(self.manual_peak_st[1], e.point_index)
            ui.notify(f'peak area: {area:.4f} ({st:.2f} ~ {ed:.2f}) for {name}')
            st_tick, ed_tick = self.manual_peak_st[1], e.point_index
            self.manual_peak_st = None
            peak_data = [name, (st, data.get_abs_data()[data.Y_HEADER][st_tick]),
                         ((st+ed)/2, underline_y[underline_y.size//2]),
                         (ed, data.get_abs_data()[data.Y_HEADER][ed_tick]), area]
            self.manual_peaks[uuid4().hex] = peak_data
        self.make_highcharts_fig(update=True)
        
    @ui.refreshable
    def make_highcharts_fig(self, update: bool = False):
        if self.dfs:
            # reomve manual peaks whose df is not in self.dfs
            for name in list(self.manual_peaks.keys()):
                if self.manual_peaks[name][0] not in self.dfs:
                    del self.manual_peaks[name]
            # draw highcharts
            if update:
                self.manual_peak_fig.options['series'] = self._make_highcharts_data()[0]
                self.manual_peak_fig.update()
            else:
                self.manual_peak_fig = ui.highchart(self._make_highcharts_data()[-1],
                                                    on_point_click=self._ui_handle_hc_click).classes('flex flex-grow')
            # gather peaks' area and make table for all peaks' area for all files
            self.make_manual_peak_table.refresh()
                
    @ui.refreshable
    def make_area_df(self):
        if not self.dfs:
            return
        # calcu peaks' area into area_df and collect all peaks' idx into all_peaks_idx
        area_df: Dict[str, Dict[int, float]] = {}
        all_peaks_idx: Dict[int, bool] = {}
        for n, peaks_idx in self.files_peaks_idx.items():
            data = self.dfs[n]
            peaks_area = data.calcu_peaks_area(peaks_idx)
            area_df[self.tag2label[n]] = {peaks_area[peak_idx]['time']:peaks_area[peak_idx]['area'] for peak_idx in peaks_area}
            for idx in area_df[self.tag2label[n]].keys():
                all_peaks_idx[idx] = True
        # make big table for all peaks' area for all files
        self.all_area_df = pd.DataFrame(columns=['Name'] + sorted(list(all_peaks_idx.keys())))
        for n, peaks_area in area_df.items():
            self.all_area_df.loc[n] = [n] + [area_df[n][idx] if idx in area_df[n] else 0 for idx in all_peaks_idx]
        self._ui_transfer_df_to_table(self.all_area_df, classes='w-full h-full')
        
    @ui.refreshable
    def make_area_percent_df(self):
        if not isinstance(self.all_area_df, pd.DataFrame) or self.all_area_df.empty:
            return
        # make percent table for all peaks' area for all files
        self.all_area_percent_df = self.all_area_df.copy()
        for i in range(self.all_area_percent_df.shape[0]):
            self.all_area_percent_df.iloc[i, 1:] /= (self.all_area_percent_df.iloc[i, 1:].sum() / 100)
        # transfer to nicegui table format
        self._ui_transfer_df_to_table(self.all_area_percent_df, classes='w-full h-full')
        
    @ui.refreshable
    async def make_pda_heatmap(self):
        return None
        # heatmap is RAM and time consuming, so just skip it for now
        if len(self.dfs) != 1 or not list(self.dfs.values())[0].IS_PDA:
            return
        data = list(self.dfs.values())[0]
        plt.close(self.pda_fig)
        with ui.pyplot(figsize=self.args.fig_size, close=False) as fig:
            self.pda_fig = fig.fig
            ax = fig.fig.gca()
            ax, ax_top = await run.cpu_bound(_plot_pda_heatmap, data, ax=ax, n_xticklabels=5*60)
            ax.tick_params(axis='both', which='major', labelsize=self.args.axis_ticks_fontsize)
            ax_top.tick_params(axis='x', which='major', labelsize=self.args.axis_ticks_fontsize)
            ax.set_xlim(left=self.args.xlim[0], right=self.args.xlim[1])
            ax.set_ylim(bottom=self.args.ylim[0], top=self.args.ylim[1])
            plt.tight_layout()
            
    def _ui_only_one_expansion(self, e):
        if e.value:
            for expansion in self._expansion:
                if expansion != e.sender:
                    expansion.set_value(False)
                    
    @ui.refreshable
    def _ui_refinment_numbers(self):
        # update dfs_refinment
        self.dfs_refinment_x = {n: self.dfs_refinment_x.get(n, 0) for n in self.dfs}
        self.dfs_refinment_y = {n: self.dfs_refinment_y.get(n, 0) for n in self.dfs}
        # update refinment numbers GUI
        for (n, x), (_, y) in zip(self.dfs_refinment_x.items(), self.dfs_refinment_y.items()):
            ui.label(self.tag2label.get(n, n)).tooltip(n)
            with ui.row():
                ui.number(label='x', value=x, step=0.01, format='%.4f').bind_value_to(self.dfs_refinment_x, n).classes('w-2/5')
                ui.number(label='y', value=y, step=0.01, format='%.4f').bind_value_to(self.dfs_refinment_y, n).classes('w-2/5')
                    
    @ui.refreshable
    def _ui_plot_params(self):
        # update refinment numbers GUI
        for n in self.dfs:
            ui.label(self.tag2label.get(n, n)).tooltip(n)
            with ui.row():
                ui.checkbox('peak_label', value=self.dfs[n].plot_params['peak_label']).bind_value_to(self.dfs[n].plot_params, 'peak_label').classes('w-2/5')
            
    def _ui_bind_xlim_onchange(self, e):
        if self.is_bind_lim:
            if e.sender == self.xlim_number_min:
                self.xlim_search_number_min.set_value(e.value)
            elif e.sender == self.xlim_search_number_min:
                self.xlim_number_min.set_value(e.value)
            elif e.sender == self.xlim_number_max:
                self.xlim_search_number_max.set_value(e.value)
            elif e.sender == self.xlim_search_number_max:
                self.xlim_number_max.set_value(e.value)
                    
    def save_fig(self):
        path = os.path.join('./', self.args.file_name)
        ui.notify(f'saving figure to {path}')
        save_show(path, dpi = self.args.dpi, show = False, bbox_extra_artists = self._bbox_extra_artists)
        
    @staticmethod
    def _apply_v2list(v, lst, idx):
        lst[idx] = v
        
    def make_gui(self):
        with ui.header(elevated=True).style('background-color: #3874c8'):
            ui.label('mbapy-cli HPLC | HPLC Data Explorer').classes('text-h4')
            ui.space()
            ui.checkbox('bind lim', value=self.is_bind_lim).bind_value_to(self, 'is_bind_lim').tooltip('bind value of search-lim and plot-lim')
            ui.checkbox('merge', value=self.args.merge).bind_value_to(self.args,'merge').bind_value_from(self, 'dfs', lambda dfs: len(dfs) > 1)
            ui.button('Save Session', on_click=self.save_session, icon='save').props('no-caps')
            ui.button('Plot', on_click=self.make_fig.refresh, icon='refresh').on_click(self.make_highcharts_fig.refresh).props('no-caps')
            ui.button('Save', on_click=self.save_fig, icon='save').props('no-caps')
            ui.button('Show', on_click=plt.show, icon='open_in_new').props('no-caps')
            ui.button('Exit', on_click=app.shutdown, icon='power')
        with ui.splitter(value = 20).classes('w-full h-full h-56') as splitter:
            with splitter.before:
                ui.upload(label = 'Load File', multiple=True, auto_upload=True, on_multi_upload=self.load_data).props('no-caps')
                tabs = self.make_tabs()
            with splitter.after:
                with ui.row().classes('w-full h-full'):
                    with ui.column().classes('h-full'):
                        # data filtering configs
                        with ui.expansion('Data Filtering', icon='filter_alt', value=True, on_value_change=self._ui_only_one_expansion) as expansion1:
                            self._expansion.append(expansion1)
                            ui.select(sorted(list(self.SUPPORT_SYSTEMS)), label='HPLC System', value=self.args.system).bind_value_to(self.args,'system').bind_value_to(self, 'data_model', lambda s: self.sys2model[s]).on_value_change(self.load_data_from_dir).classes('w-full')
                            ui.number('PDA wave length', value=self.args.pda_wave_length, step=0.1, format='%.1f').bind_value_to(self.args, 'pda_wave_length').classes('w-full')
                            ui.number('min peak width', value=self.args.min_peak_width, min = 0, step = 0.10).bind_value_to(self.args,'min_peak_width').tooltip('in minutes')
                            ui.number('min height', value=self.args.min_height, min = 0, step=0.01).bind_value_to(self.args, 'min_height')
                            ui.number('labels eps', value=self.args.labels_eps, min=0, format='%.2f').bind_value_to(self.args, 'labels_eps')
                            self.xlim_search_number_min = ui.number('start search time', value=self.args.start_search_time, min = 0, on_change=self._ui_bind_xlim_onchange).bind_value_to(self.args,'start_search_time').tooltip('in minutes')
                            self.xlim_search_number_max = ui.number('end search time', value=self.args.end_search_time, min = 0, on_change=self._ui_bind_xlim_onchange).bind_value_to(self.args, 'end_search_time').tooltip('in minutes')
                        # data refinment configs
                        with ui.expansion('Data Refinment', icon='auto_fix_high', on_value_change=self._ui_only_one_expansion) as expansion2:
                            self._expansion.append(expansion2)
                            self._ui_refinment_numbers()
                        # data refinment configs
                        with ui.expansion('Single Params', icon='format_list_bulleted', on_value_change=self._ui_only_one_expansion) as expansion6:
                            self._expansion.append(expansion6)
                            self._ui_plot_params()
                        # configs for fontsize
                        with ui.expansion('Configs for Fontsize', icon='format_size', on_value_change=self._ui_only_one_expansion) as expansion3:
                            self._expansion.append(expansion3)
                            with ui.row().classes('w-full'):
                                ui.input('title', value=self.args.title).bind_value_to(self.args, 'title')
                                ui.number('title fontsize', value=self.args.title_fontsize, min=0, step=0.5, format='%.1f').bind_value_to(self.args, 'title_fontsize')
                            with ui.row().classes('w-full'):
                                ui.input('xlabel', value=self.args.xlabel).bind_value_to(self.args, 'xlabel')
                                ui.input('ylabel', value=self.args.ylabel).bind_value_to(self.args, 'ylabel')
                            with ui.row().classes('w-full'):
                                ui.number('axis label fontsize', value=self.args.axis_label_fontsize, min=0, step=0.5, format='%.1f').bind_value_to(self.args, 'axis_label_fontsize')
                                ui.number('axis ticks fontsize', value=self.args.axis_ticks_fontsize, min=0, step=0.5, format='%.1f').bind_value_to(self.args, 'axis_ticks_fontsize')
                            ui.checkbox('show tag text', value=self.args.show_tag_text).bind_value_to(self.args,'show_tag_text')
                            with ui.row().classes('w-full'):
                                ui.number('tag fontsize', value=self.args.tag_fontsize, min=0, step=0.5, format='%.1f').bind_value_to(self.args, 'tag_fontsize')
                                ui.number('marker size', value=self.args.marker_size, min=0, step=5, format='%.1f').bind_value_to(self.args,'marker_size')
                            with ui.row().classes('w-full'):
                                ui.number('tag offset x', value=self.args.tag_offset[0], step=0.01, format='%.2f').on_value_change(lambda e: self._apply_v2list(e.value, self.args.tag_offset, 0))
                                ui.number('marker offset x', value=self.args.marker_offset[0], step=0.01, format='%.2f').on_value_change(lambda e: self._apply_v2list(e.value, self.args.marker_offset, 0))
                            with ui.row().classes('w-full'):
                                ui.number('tag offset y', value=self.args.tag_offset[1], step=0.01, format='%.2f').on_value_change(lambda e: self._apply_v2list(e.value, self.args.tag_offset, 1))
                                ui.number('marker offset y', value=self.args.marker_offset[1], step=0.01, format='%.2f').on_value_change(lambda e: self._apply_v2list(e.value, self.args.marker_offset, 1))
                            ui.number('line width', value=self.args.line_width, min=0, step=0.5, format='%.1f').bind_value_to(self.args, 'line_width')
                        # configs for legend
                        with ui.expansion('Configs for Legend', icon='more', on_value_change=self._ui_only_one_expansion) as expansion4:
                            self._expansion.append(expansion4)
                            with ui.row().classes('w-full'):
                                ui.checkbox('show file legend', value=self.args.show_file_legend).bind_value_to(self.args,'show_file_legend')
                                ui.checkbox('show peak legend', value=self.args.show_tag_legend).bind_value_to(self.args,'show_tag_legend')
                            with ui.row().classes('w-full'):
                                ui.textarea('file labels').bind_value_to(self.args, 'file_labels').props('clearable').tooltip('input as label1,color1;label2,color2')
                                ui.textarea('peak labels').bind_value_to(self.args, 'peak_labels').props('clearable').tooltip('input as peaktime,label,color;...')
                            with ui.row().classes('w-full'):
                                col_mode_option = ['hls', 'Set1', 'Set2', 'Set3', 'Dark2', 'Paired', 'Pastel1', 'Pastel2', 'tab10', 'tab20', 'tab20b', 'tab20c']
                                ui.select(label='file col mode', options = col_mode_option, value=self.args.file_col_mode).bind_value_to(self.args, 'file_col_mode').classes('w-2/5')
                                ui.select(label='peak col mode', options = col_mode_option, value=self.args.peak_col_mode).bind_value_to(self.args, 'peak_col_mode').classes('w-2/5')
                            ui.number('legend fontsize', value=self.args.legend_fontsize, min=0, step=0.5, format='%.2f').bind_value_to(self.args, 'legend_fontsize')
                            with ui.row().classes('w-full'):
                                all_loc = ['best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center']
                                ui.select(label='file legend loc', options=all_loc, value=self.args.file_legend_pos).bind_value_to(self.args, 'file_legend_pos').classes('w-2/5')
                                ui.select(label='peak legend loc', options=all_loc, value=self.args.peak_legend_pos).bind_value_to(self.args, 'peak_legend_pos').classes('w-2/5')
                            with ui.row().classes('w-full'):
                                ui.number('file bbox1', value=self.args.file_legend_bbox[0], step=0.01, format='%.2f').on_value_change(lambda e: self._apply_v2list(e.value, self.args.file_legend_bbox, 0)).classes('w-2/5')
                                ui.number('peak bbox1', value=self.args.peak_legend_bbox[0], step=0.01, format='%.2f').on_value_change(lambda e: self._apply_v2list(e.value, self.args.peak_legend_bbox, 0)).classes('w-2/5')
                            with ui.row().classes('w-full'):
                                ui.number('file bbox2', value=self.args.file_legend_bbox[1], step=0.01, format='%.2f').on_value_change(lambda e: self._apply_v2list(e.value, self.args.file_legend_bbox, 1)).classes('w-2/5')
                                ui.number('peak bbox2', value=self.args.peak_legend_bbox[1], step=0.01, format='%.2f').on_value_change(lambda e: self._apply_v2list(e.value, self.args.peak_legend_bbox, 1)).classes('w-2/5')
                        # configs for saving
                        with ui.expansion('Configs for Saving', icon='save', on_value_change=self._ui_only_one_expansion) as expansion5:
                            self._expansion.append(expansion5)
                            with ui.row().classes('w-full'):
                                ui.checkbox('plot peaks line', value=self.args.plot_peaks_line).bind_value_to(self.args, 'plot_peaks_line').classes('w-2/5')
                                ui.checkbox('plot peaks underline', value=self.args.plot_peaks_underline).bind_value_to(self.args, 'plot_peaks_underline').classes('w-2/5')
                            with ui.row().classes('w-full'):
                                ui.checkbox('plot peaks area', value=self.args.plot_peaks_area).bind_value_to(self.args, 'plot_peaks_area').classes('w-2/5')
                                ui.number('area alpha', value=self.args.peak_area_alpha, min=0, max=1, step=0.1, format='%.2f').bind_value_to(self.args, 'peak_area_alpha').classes('w-2/5')
                            with ui.row().classes('w-full'):
                                self.xlim_number_min = ui.number('xlim-min', value=self.args.xlim[0], step=0.1, format='%.2f', on_change=self._ui_bind_xlim_onchange).on_value_change(lambda e: self._apply_v2list(e.value, self.args.xlim, 0))
                                self.xlim_number_max = ui.number('xlim-max', value=self.args.xlim[1], step=0.1, format='%.2f', on_change=self._ui_bind_xlim_onchange).on_value_change(lambda e: self._apply_v2list(e.value, self.args.xlim, 1))
                            with ui.row().classes('w-full'):
                                ui.number('ylim-min', value=self.args.ylim[0], step=0.01, format='%.2f').on_value_change(lambda e: self._apply_v2list(e.value, self.args.ylim, 0))
                                ui.number('ylim-max', value=self.args.ylim[1], step=0.01, format='%.2f').on_value_change(lambda e: self._apply_v2list(e.value, self.args.ylim, 1))
                            with ui.row().classes('w-full'):
                                ui.number('figure width', value=self.args.fig_size[0], min=1, step=0.5, format='%.2f').on_value_change(lambda e: self._apply_v2list(e.value, self.args.fig_size, 0)).classes('w-2/5')
                                ui.number('figure height', value=self.args.fig_size[1], min=1, step=0.5, format='%.2f').on_value_change(lambda e: self._apply_v2list(e.value, self.args.fig_size, 1)).classes('w-2/5')
                            with ui.row().classes('w-full'):
                                dpi_input = ui.number('DPI', value=self.args.dpi, min=100, step=100, format='%d').bind_value_to(self.args, 'dpi').classes('w-2/5')
                                ui.select(options=[100, 300, 600], value=dpi_input.value, label='Quick Set DPI').bind_value_to(dpi_input).classes('w-2/5')
                            ui.input('figure file name', value=self.args.file_name).bind_value_to(self.args, 'file_name').classes('w-4/5')
                    with ui.column().classes('h-full flex flex-grow'):
                        with ui.tabs().classes('flex flex-grow justify-center') as tabs:
                            fig_panel = ui.tab('HPLC Figure').props('no-caps').classes('flex flex-grow')
                            highcharts_fig_panel = ui.tab('HPLC Manual Peaking Figure').props('no-caps').classes('flex flex-grow')
                            self.area_df_panel = ui.tab('HPLC Peaks Area DataFrame').props('no-caps').classes('flex flex-grow')
                            self.area_percent_df_panel = ui.tab('HPLC Peaks Area Percentage DataFrame').props('no-caps').classes('flex flex-grow')
                            self.pda_heatmap_panel = ui.tab('PDA Heatmap').props('no-caps').classes('flex flex-grow')
                        with ui.tab_panels(tabs, value=fig_panel).classes('flex flex-grow'):
                            with ui.tab_panel(fig_panel).classes('flex flex-grow'):
                                self.make_fig()
                            with ui.tab_panel(highcharts_fig_panel).classes('flex flex-grow'):
                                self.make_highcharts_fig()
                                self.make_manual_peak_table()
                            with ui.tab_panel(self.area_df_panel).classes('flex flex-grow'):
                                self.make_area_df()
                            with ui.tab_panel(self.area_percent_df_panel).classes('flex flex-grow'):
                                self.make_area_percent_df()
                            with ui.tab_panel(self.pda_heatmap_panel).classes('flex flex-grow'):
                                self.make_pda_heatmap()
        ui.run(host = self.args.url, port = self.args.port, title = 'HPLC Data Explorer',
               favicon=get_storage_path('icons/scripts-hplc-peak.png'), reload=False)
    
    def main_process(self):
        # make global settings
        self.args = BaseInfo(file_labels = '', peak_labels = '', merge = False, recursive = False,
                             pda_wave_length = 228,
                             min_peak_width = 0.1, min_height = 0.01, start_search_time = 0, end_search_time = None,
                             show_tag_text = True, labels_eps = 0.1,
                             file_legend_pos = 'upper right', file_legend_bbox = [1., 1.],
                             peak_legend_pos = 'upper right', peak_legend_bbox = [1.3, 1],
                             title = '', xlabel = 'Time (min)', ylabel = 'Absorbance (AU)',
                             axis_ticks_fontsize = 20, axis_label_fontsize = 25, 
                             file_col_mode = 'hls', peak_col_mode = 'Set1',
                             show_tag_legend = True, show_file_legend = True,
                             tag_fontsize = 15, tag_offset = [0.05,0.05], marker_size = 80, marker_offset = [0,0.05],
                             title_fontsize = 25, legend_fontsize = 15, line_width = 2,
                             xlim = [0, None], ylim = [None, None],
                             fig_size = [12, 8], fig = None, dpi = 600, file_name = '',
                             plot_peaks_line = False, plot_peaks_underline = False, plot_peaks_area = False, peak_area_alpha = 0.3,
                             **self.args.__dict__)
        # load dfs from input dir to stored_dfs
        self.load_data_from_dir()
        # GUI
        self.make_gui()
        
    def save_session(self):
        path = os.path.join(self.args.input, f'{get_fmt_time()}.mpss')
        super().save_session('hplc', path = path)
        ui.notify(f'session saved to {path}')
        
    def exec_from_session(self, session: Command):
        self.args, self.all_area_df, self.dfs_checkin, self.stored_dfs, self.dfs_refinment_x, self.dfs_refinment_y = session
        self.make_gui()
        
        
class extract_pda(plot_hplc):
    SUPPORT_SYSTEMS = {'Waters-PDA'}
    def __init__(self, args: argparse.Namespace, printf=print) -> None:
        super().__init__(args, printf)
        self.sys2suffix = {'Waters-PDA': 'arw'}
        self.sys2model: Dict[str, HplcData] = {'Waters-PDA': WatersPdaData}
        
    @staticmethod
    def make_args(args: argparse.ArgumentParser):
        args.add_argument('-w', '--wave-length', type = float,
                          help="traget wave length to extract.")
        args.add_argument('-i', '--input', type = str, default='.',
                          help="data file directory, default is %(default)s.")
        args.add_argument('-n', '--use-tag-name', action='store_true', default=False,
                          help="use tag name as file name, default is %(default)s.")
        args.add_argument('-s', '--system', type = str, default='Waters-PDA',
                          help=f"HPLC system. Default is %(default)s, those systems are supported: {', '.join(list(plot_hplc.SUPPORT_SYSTEMS))}")
        args.add_argument('-r', '--recursive', action='store_true', default=False,
                          help='search input directory recursively, default is %(default)s.')
        return args

    def process_args(self):
        assert self.args.system in self.SUPPORT_SYSTEMS, f'not support HPLC system: {self.args.system}'
        # process self.args
        self.args.input = clean_path(self.args.input)
        self.data_model = self.sys2model[self.args.system]

    def main_process(self):
        # load origin dfs from data file
        self.dfs: List[Tuple[str, Union[WatersPdaData,]]] = self.load_dfs_from_data_file()
        if not self.dfs:
            raise FileNotFoundError(f'can not find data files in {self.args.input}')
        # show data general info and output peak list DataFrame
        for i, (tag, data) in enumerate(self.dfs):
            print(f'extracting data for {tag} @ {self.args.wave_length} nm')
            abs_data = data.get_abs_data(self.args.wave_length)
            path = Path(data.data_file_path)
            if self.args.use_tag_name:
                tag = get_valid_file_path(tag).replace('/', '-')
                path = path.parent / f'{tag} @{self.args.wave_length}nm.xlsx'
            else:
                path = path.with_suffix(f'@{self.args.wave_length}nm.xlsx')
            write_sheets(path, {'Info': data.info_df, 'Data': abs_data}, index = False)


def plot_single_PDA_data(data: HplcData):
    tag = data.make_tag(tags=['"样品名称"', '"采集日期"', '"通道"', '"仪器方法名"'], join_str=' _ ').replace('/', '-')
    ax, ax_top = _plot_pda_heatmap(data, n_xticklabels=5*60)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax_top.tick_params(axis='x', which='major', labelsize=15)
    ax.set_title(tag, fontsize=20)
    ax.set_xlabel('Time (min)', fontsize=17)
    ax.set_ylabel('Wavelength (nm)', fontsize=17)
    plt.tight_layout()
    path = Path(data.data_file_path)
    path = get_valid_file_path(str(path.parent / f'{tag}_heatmap.png'))
    save_show(path, 600, show=False)
    print(f'save PDA heatmap to {path}')


class plot_pda(extract_pda):
    def __init__(self, args: argparse.Namespace, printf=print) -> None:
        super().__init__(args, printf)
        self.task_pool: Optional[TaskPool] = None

    @staticmethod
    def make_args(args: argparse.ArgumentParser):
        args.add_argument('-i', '--input', type = str, default='.',
                          help="data file directory, default is %(default)s.")
        args.add_argument('-n', '--use-tag-name', action='store_true', default=False,
                          help="use tag name as file name, default is %(default)s.")
        args.add_argument('-s', '--system', type = str, default='Waters-PDA',
                          help=f"HPLC system. Default is %(default)s, those systems are supported: {', '.join(list(plot_hplc.SUPPORT_SYSTEMS))}")
        args.add_argument('-r', '--recursive', action='store_true', default=False,
                          help='search input directory recursively, default is %(default)s.')
        args.add_argument('-mp', '--multi-process', type = int, default=4,
                          help='multi-process to speed up plot, default is %(default)s')
        return args

    def main_process(self):
        if self.args.multi_process > 1:
            self.task_pool = TaskPool('process', self.args.multi_process).start()
            print(f'created task pool with {self.args.multi_process} processes')
        # load origin dfs from data file
        self.dfs: List[Tuple[str, Union[WatersPdaData,]]] = self.load_dfs_from_data_file()
        if not self.dfs:
            raise FileNotFoundError(f'can not find data files in {self.args.input}')
        print(f'loaded {len(self.dfs)} data files')
        # show data general info and output peak list DataFrame
        for i, (tag, data) in enumerate(self.dfs):
            # save processed data
            if self.task_pool is not None:
                self.task_pool.add_task(tag, plot_single_PDA_data, data)
            else:
                plot_single_PDA_data(data)
        if self.task_pool is not None:
            self.task_pool.wait_till_tasks_done(self.dfs.keys())
            self.task_pool.close()
        

_str2func = {
    'plot-hplc': plot_hplc,
    'explore-hplc': explore_hplc,
    'extract-pda': extract_pda,
    'plot-pda': plot_pda,
}


def main(sys_args: List[str] = None):
    args_paser = argparse.ArgumentParser()
    subparsers = args_paser.add_subparsers(title='subcommands', dest='sub_command')
    plot_hplc_args = plot_hplc.make_args(subparsers.add_parser('plot-hplc', description='plot hplc spectrum'))
    explore_hplc_args = explore_hplc.make_args(subparsers.add_parser('explore-hplc', description='explore hplc spectrum data'))
    extract_pda_args = extract_pda.make_args(subparsers.add_parser('extract-pda', description='extract PDA data'))
    plot_pda_args = plot_pda.make_args(subparsers.add_parser('plot-pda', description='plot PDA data as heatmap'))

    excute_command(args_paser, sys_args, _str2func)

if __name__ == "__main__":
    # dev code, MUST COMMENT OUT BEFORE RELEASE
    # main('explore-hplc -i data_tmp/scripts/hplc'.split())
    # main('extract-pda -w 228 -i data_tmp/scripts/hplc/WatersPDA.arw -n'.split())
    
    main()