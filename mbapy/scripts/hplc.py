import argparse
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
from mbapy.base import put_err
from mbapy.plot import get_palette, save_show
from mbapy.file import decode_bits_to_str, get_paths_with_extension, get_valid_file_path
from mbapy.scripts._script_utils_ import clean_path, Command, excute_command

                
class plot_hplc(Command):
    def __init__(self, args: argparse.Namespace, printf=print) -> None:
        super().__init__(args, printf)
        self.dfs = {}
        self.SUPPORT_SYSTEMS = {'waters'}
        
    @staticmethod
    def make_args(args: argparse.ArgumentParser):
        args.add_argument('-i', '--input', type = str, default='.',
                          help="data file directory, default is %(default)s.")
        args.add_argument('-s', '--system', type = str, default='waters',
                          help="HPLC system. Default is %(default)s, only accept arw file exported by Waters.")
        args.add_argument('-r', '--recursive', action='store_true', default=False,
                          help='search input directory recursively, default is %(default)s.')
        args.add_argument('-merge', action='store_true', default=False,
                          help='merge multi files into one plot, default is %(default)s.')
        args.add_argument('-o', '--output', type = str, default=None,
                          help="output file dir or path. Default is %(default)s, means same as input dir")
        # set draw argument
        args.add_argument('--min-peak-width', type = float, default=4,
                          help='filter peaks with min width in hplc/Charge plot, default is %(default)s.')
        args.add_argument('-xlim', type = str, default='0,None',
                          help='set x-axis limit, input as "0,15", default is %(default)s.')
        args.add_argument('-cols', '--colors', type = str, default='black',
                          help='draw color, default is %(default)s.')
        args.add_argument('-labels', '--labels', type = str, default='',
                          help='labels, input as 1000,Pep1;1050,Pep2, default is %(default)s.')
        args.add_argument('-flabels', '--file-labels', type = str, default='',
                          help='labels, input as 228,blue;304,red, default is %(default)s.')
        args.add_argument('--labels-eps', type = float, default=0.5,
                          help='eps to recognize labels, default is %(default)s.')
        args.add_argument('-expand', '--expand', type = float, default=0.2,
                          help='how much the x-axis and y-axisto be expanded, default is %(default)s.')
        args.add_argument('-lpos', '--legend-pos', type = str, default='upper center',
                          help='legend position, can be string as "upper center", or be float as 0.1,0.2, default is %(default)s')
        args.add_argument('-lposbbox1', '--legend-pos-bbox1', type = float, default=1.1,
                          help='legend position bbox 1 to anchor, default is %(default)s')
        args.add_argument('-lposbbox2', '--legend-pos-bbox2', type = float, default=1,
                          help='legend position bbox 2 to anchor, default is %(default)s')
        return args
    
    @staticmethod
    def load_arw_file(path: Path, content: str = None):
        content = content or path.read_text()
        lines = content.splitlines()
        info_df = pd.DataFrame([lines[1].split('\t')], columns = lines[0].split('\t'))
        data_df = pd.DataFrame([line.split('\t') for line in lines[2:]],
                            columns = ['Time', 'Absorbance'])
        return info_df, data_df.astype({'Time': float, 'Absorbance': float})
    
    def process_file_labels(self, labels: str):
        labels = '' if labels is None else labels
        col_mode = self.args.__dict__.get('file_col_mode', 'hls')
        file_labels, colors = [], get_palette(len(labels.split(';')), mode = 'hls')
        for idx, i in enumerate(labels.split(';')):
            if i:
                pack = i.split(',')
                label, color = pack[0], pack[1] if len(pack) == 2 else colors[idx]
                file_labels.append([label, color])
        return file_labels
    
    def load_dfs_from_data_file(self):
        if self.args.system == 'waters':
            paths = get_paths_with_extension(self.args.input, ['arw'], recursive=self.args.recursive)
            load_data = plot_hplc.load_arw_file
        dfs = {path:load_data(Path(path)) for path in paths}
        return {k:v for k,v in dfs.items() if v is not None}
    
    def process_args(self):
        assert self.args.system in {'waters'}, f'not support HPLC system: {self.args.system}'
        # process self.args
        self.args.input = clean_path(self.args.input)
        self.args.output = clean_path(self.args.output) if self.args.output else self.args.input
        if not os.path.isdir(self.args.output):
            print(f'given output {self.args.output} is a file, change it to parent dir')
            self.args.output = self.args.output.parent
        # labels
        labels, colors = {}, get_palette(len(self.args.labels.split(';')), mode = 'hls')
        for idx, i in enumerate(self.args.labels.split(';')):
            if i:
                pack = i.split(',')
                mass, label, color = pack[0], pack[1], pack[2] if len(pack) == 3 else colors[idx]
                labels[float(mass)] = [label, color]
        self.args.labels = labels
        # file labels
        self.args.file_labels = self.process_file_labels(self.args.file_labels)
        
    @staticmethod
    def process_waters_data(path: str, df, args: argparse.Namespace, save_df: bool = True, show_df: bool = True):
        info_df, data_df = df
        # output csv
        sample_name, sample_time, sample_channle = info_df['"样品名称"'][0], info_df['"采集日期"'][0], info_df['"通道"'][0]
        csv_name = f'{sample_name} - {sample_time} - {sample_channle}'.replace('"', '')
        if save_df:
            csv_name = get_valid_file_path(csv_name.replace('/', '-'), valid_len=500).replace(':', '-')
            info_df.to_csv(str(path.parent / get_valid_file_path(csv_name + ' - info.csv')))
            data_df.to_csv(str(path.parent / get_valid_file_path(csv_name + ' - data.csv')))
        if show_df:
            print(f'plot {path.stem}: {csv_name}')
            print(info_df.T)
        return csv_name, info_df, data_df, args
        
    @staticmethod
    def plot_waters(file_name:str, info_df: pd.DataFrame, data_df: pd.DataFrame, args):
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
    
    def main_process(self):
        # load origin dfs from data file
        self.dfs = self.load_dfs_from_data_file()
        if not self.dfs:
            raise FileNotFoundError(f'can not find data files in {self.args.input}')
        # show data general info and output peak list DataFrame
        if self.args.merge:
            if self.args.system == 'waters':
                dfs = list(self.dfs.values())
                info_df, data_df = [d[0] for d in dfs], [d[1] for d in dfs]
                plot_hplc.plot_waters('merge', info_df, data_df, self.args)
        else:
            for path, df in self.dfs.items():
                path = Path(path).resolve()
                # plot each df
                process_fn = getattr(self, f'process_{self.args.system}_data')
                plot_fn = getattr(self, f'plot_{self.args.system}')
                plot_fn(*process_fn(path, df, self.args))


class explore_hplc(plot_hplc):
    from nicegui import ui
    def __init__(self, args: argparse.Namespace, printf=print) -> None:
        super().__init__(args, printf)
        self.now_name = ''
        self.fig = None
        self.dfs_checkin = {}
        self.stored_dfs = {}
        self._expansion = []
        
    @staticmethod
    def make_args(args: argparse.ArgumentParser):
        args.add_argument('-i', '--input', type = str, default='.',
                          help="data file directory, default is %(default)s.")
        args.add_argument('-s', '--system', type = str, default='waters',
                          help="HPLC system. Default is %(default)s, only accept arw file exported by Waters.")
        return args
    
    def process_args(self):
        self.args.input = clean_path(self.args.input)
        assert self.args.system in {'waters'}, f'not support HPLC system: {self.args.system}'
        
    async def load_data(self, event):
        from nicegui import ui
        for name, content in zip(event.names, event.contents):
            if self.args.system == 'waters':
                if name.endswith('.arw'):
                    content = decode_bits_to_str(content.read())
                    info_df, data_df = plot_hplc.load_arw_file(None, content)
                    name, info_df, data_df, _ = plot_hplc.process_waters_data(name, (info_df, data_df), None, save_df=False, show_df=False)
                    self.stored_dfs[name] = (info_df, data_df)
                else:
                    ui.notify(f'{name} is not a arw file')
                    continue
            ui.notify(f'loaded {name}')
        self.make_tabs.refresh()
        
    def _push_df_from_tabs(self, event):
        if event.value:
            self.dfs[event.sender.text] = self.stored_dfs[event.sender.text]
        else:
            self.dfs.pop(event.sender.text, None)
        
    @ui.refreshable
    def make_tabs(self):
        from nicegui import ui
        with ui.card().classes('h-full'):
            for name in self.stored_dfs:
                if name not in self.dfs_checkin:
                    self.dfs_checkin[name] = False
                ui.checkbox(text = name, value = self.dfs_checkin[name],
                            on_change=self._push_df_from_tabs).bind_value_to(self.dfs_checkin, name)
                
    def process_peak_labels(self, peak_labels: str):
        peak_labels = '' if peak_labels is None else peak_labels
        labels, cols = {}, get_palette(len(peak_labels.split(';')), mode = self.args.peak_col_mode)
        for i, label in enumerate(peak_labels.split(';')):
            if label:
                items = label.split(',')
                if len(items) == 2:
                    (t, label), color = items, cols[i]
                elif len(items) == 3:
                    t, label, color = items
                labels[float(t)] = [label, color]
        return labels
                
    def plot_waters(self, ax: plt.Axes):
        from nicegui import ui
        names, info_df, data_df = [], [], []
        for name, df in self.dfs.items():
            names.append(name)
            info_df.append(df[0])
            data_df.append(df[1])
        # check if no data
        if len(info_df) == 0:
            return ui.notify('no data to plot')
        # process file labels
        file_labels = self.process_file_labels(self.args.file_labels)
        if not file_labels or len(file_labels) != len(names):
            ui.notify(f'only {len(file_labels)} labels found, should be {len(names)} labels, use name instead')
            file_labels = self.process_file_labels(';'.join(names))
            if len(file_labels) == 1:
                file_labels[0][1] = 'black'
        # process peak labels
        peak_labels = self.process_peak_labels(self.args.peak_labels)
        peak_labels_v = np.array(list(peak_labels.keys()))
        # plot each
        tag_offset = [float(i) for i in self.args.tag_offset.split(',')]
        marker_offset = [float(i) for i in self.args.marker_offset.split(',')]
        lines, scatters, sc_labels = [], [], []
        for label, info_df_i, data_df_i in zip(file_labels, info_df, data_df):
            label_string, color = label
            line = ax.plot(data_df_i['Time'], data_df_i['Absorbance'], color = color, label = label_string)[0]
            lines.append(line)
            # search peaks
            st = int(self.args.start_search_time)
            peaks_idx, peak_props = scipy.signal.find_peaks(data_df_i['Absorbance'], rel_height = 1,
                                                        prominence =self.args.min_height,
                                                        width = self.args.min_peak_width)
            peaks_idx = peaks_idx[peaks_idx > st]
            peak_df = data_df_i.iloc[peaks_idx, :]
            for t, a in zip(peak_df['Time'], peak_df['Absorbance']):                    
                matched = np.where(np.abs(peak_labels_v - t) < self.args.labels_eps)[0]
                if matched.size > 0:
                    label, col = peak_labels[peak_labels_v[matched[0]]]
                    sc = ax.scatter(t+marker_offset[0], a+marker_offset[1], marker='*', s = self.args.marker_size, color = col)
                    scatters.append(sc)
                    sc_labels.append(label)
                else:
                    ax.scatter(t+marker_offset[0], a+marker_offset[1], marker=11, s = self.args.marker_size, color = 'black')
                if self.args.show_tag_text:
                    ax.text(t+tag_offset[0], a+tag_offset[1], f'{t:.2f}', fontsize=self.args.tag_fontsize)
        # style fix
        ax.tick_params(axis='both', which='major', labelsize=self.args.axis_ticks_fontsize)
        ax.set_xlabel(self.args.xlabel, fontsize=self.args.axis_label_fontsize)
        ax.set_ylabel(self.args.ylabel, fontsize=self.args.axis_label_fontsize)
        # set file labels legend
        if self.args.show_file_legend:
            file_legend = plt.legend(fontsize=self.args.legend_fontsize, loc = self.args.legend_pos,
                                    bbox_to_anchor = (self.args.bbox1, self.args.bbox2), draggable = True)
            ax.add_artist(file_legend)
        # set peak labels legend
        if scatters and self.args.show_peak_legend:
            [line.set_label(None) for line in lines]
            [sc.set_label(l) for sc, l in zip(scatters, sc_labels)]
            peak_legend = plt.legend(fontsize=self.args.legend_fontsize, loc = self.args.legend_pos,
                                     bbox_to_anchor = (self.args.bbox1, self.args.bbox2), draggable = True)
            ax.add_artist(peak_legend)
        
                
    @ui.refreshable
    def make_fig(self):
        from nicegui import ui
        plt.close(self.fig)
        with ui.pyplot(figsize=(self.args.fig_w, self.args.fig_h), close=False) as fig:
            self.fig = fig.fig
            getattr(self, f'plot_{self.args.system}')(fig.fig.gca())
            
    def _ui_only_one_expansion(self, e):
        if e.value:
            for expansion in self._expansion:
                if expansion != e.sender:
                    expansion.value = False                
    
    def main_process(self):
        from nicegui import app, ui
        from mbapy.game import BaseInfo
        # make global settings
        # do not support xlim because it makes confusion with peak searching
        self.args = BaseInfo(file_labels = '', peak_labels = '', merge = False, recursive = False,
                             min_peak_width = 1, min_height = 0, start_search_time = 0,
                             show_tag_text = True, labels_eps = 0.1,
                             legend_pos = 'upper right', bbox1 = 1.2, bbox2 = 1,
                             title = '', xlabel = 'Time (min)', ylabel = 'Absorbance (AU)',
                             axis_ticks_fontsize = 20,axis_label_fontsize = 25, 
                             file_col_mode = 'hls', peak_col_mode = 'Set1',
                             show_tag_legend = True, show_file_legend = True,
                             tag_fontsize = 15, tag_offset = '0.05,0.05', marker_size = 80, marker_offset = '0,0.05',
                             title_fontsize = 25, legend_fontsize = 15,
                             fig_w = 10, fig_h = 8, fig = None, dpi = 600, file_name = '', show_fig = False,
                             **self.args.__dict__)
        # load dfs from input dir
        for name, dfs in self.load_dfs_from_data_file().items():
            process_fn = getattr(self, f'process_{self.args.system}_data')
            name, info_df, data_df, _ = process_fn(name, dfs, None, save_df=False, show_df=False)
            self.stored_dfs[name] = (info_df, data_df)
        # GUI
        with ui.header(elevated=True).style('background-color: #3874c8'):
            ui.label('mbapy-cli HPLC | HPLC Data Explorer').classes('text-h4')
            ui.space()
            ui.checkbox('merge', value=self.args.merge).bind_value_to(self.args,'merge').bind_value_from(self, 'dfs', lambda dfs: len(dfs) > 1)
            ui.button('Plot', on_click=self.make_fig.refresh, icon='refresh').props('no-caps')
            ui.button('Save', on_click=partial(save_show, path = self.args.file_name, dpi = self.args.dpi, show = self.args.show_fig), icon='save').props('no-caps')
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
                            ui.select(list(self.SUPPORT_SYSTEMS), label='HPLC System', value=self.args.system).bind_value_to(self.args,'system').classes('w-full')
                            ui.number('min peak width', value=self.args.min_peak_width, min = 0, step = 0.10).bind_value_to(self.args,'min_peak_width')
                            ui.number('min height', value=self.args.min_height, min = 0, step=0.01).bind_value_to(self.args, 'min_height')
                            ui.number('start search time', value=self.args.start_search_time, min = 0).bind_value_to(self.args,'start_search_time')
                        # configs for fontsize
                        with ui.expansion('Configs for Fontsize', icon='format_size', on_value_change=self._ui_only_one_expansion) as expansion2:
                            self._expansion.append(expansion2)
                            ui.number('title fontsize', value=self.args.title_fontsize, min=0, step=0.5, format='%.1f').bind_value_to(self.args, 'title_fontsize')
                            ui.number('axis ticks fontsize', value=self.args.axis_ticks_fontsize, min=0, step=0.5, format='%.1f').bind_value_to(self.args, 'axis_ticks_fontsize')
                            ui.number('axis label fontsize', value=self.args.axis_label_fontsize, min=0, step=0.5, format='%.1f').bind_value_to(self.args, 'axis_label_fontsize')
                            ui.checkbox('show tag text', value=self.args.show_tag_text).bind_value_to(self.args,'show_tag_text')
                            with ui.row().classes('w-full'):
                                ui.number('tag fontsize', value=self.args.tag_fontsize, min=0, step=0.5, format='%.1f').bind_value_to(self.args, 'tag_fontsize')
                                ui.input('tag offset', value=self.args.tag_offset).bind_value_to(self.args, 'tag_offset').tooltip('input as x_offset,y_offset')
                            with ui.row().classes('w-full'):
                                ui.number('marker size', value=self.args.marker_size, min=0, step=5, format='%.1f').bind_value_to(self.args,'marker_size')
                                ui.input('marker offset', value=self.args.marker_offset).bind_value_to(self.args,'marker_offset').tooltip('input as x_offset,y_offset')
                            ui.input('title', value=self.args.title).bind_value_to(self.args, 'title')
                            ui.input('xlabel', value=self.args.xlabel).bind_value_to(self.args, 'xlabel')
                            ui.input('ylabel', value=self.args.ylabel).bind_value_to(self.args, 'ylabel')
                        # configs for legend
                        with ui.expansion('Configs for Legend', icon='more', on_value_change=self._ui_only_one_expansion) as expansion3:
                            self._expansion.append(expansion3)
                            with ui.row().classes('w-full'):
                                ui.checkbox('show file legend', value=self.args.show_file_legend).bind_value_to(self.args,'show_file_legend')
                                ui.checkbox('show peak legend', value=self.args.show_tag_legend).bind_value_to(self.args,'show_tag_legend')
                            with ui.row().classes('w-full'):
                                ui.textarea('file labels').bind_value_to(self.args, 'file_labels').props('clearable').tooltip('input as label1,color1;label2,color2')
                                ui.textarea('peak labels').bind_value_to(self.args, 'peak_labels').props('clearable').tooltip('input as peaktime,label,color;...')
                            col_mode_option = ['hls', 'Set1', 'Set2', 'Set3', 'Dark2', 'Paired', 'Pastel1', 'Pastel2', 'tab10', 'tab20', 'tab20b', 'tab20c']
                            with ui.row().classes('w-full'):
                                ui.select(label='file col mode', options = col_mode_option, value=self.args.file_col_mode).bind_value_to(self.args, 'file_col_mode').classes('w-32')
                                ui.select(label='peak col mode', options = col_mode_option, value=self.args.peak_col_mode).bind_value_to(self.args, 'peak_col_mode').classes('w-32')
                            ui.number('labels eps', value=self.args.labels_eps, min=0, format='%.1f').bind_value_to(self.args, 'labels_eps')
                            ui.number('legend fontsize', value=self.args.legend_fontsize, min=0, step=0.5, format='%.1f').bind_value_to(self.args, 'legend_fontsize')
                            ui.input('legend loc', value=self.args.legend_pos).bind_value_to(self.args, 'legend_pos')
                            ui.number('bbox1', value=self.args.bbox1, min=0, step=0.1, format='%.1f').bind_value_to(self.args, 'bbox1')
                            ui.number('bbox2', value=self.args.bbox2, min=0, step=0.1, format='%.1f').bind_value_to(self.args, 'bbox2')
                        # configs for saving
                        with ui.expansion('Configs for Saving', icon='save', on_value_change=self._ui_only_one_expansion) as expansion4:
                            self._expansion.append(expansion4)
                            ui.checkbox('show figure', value=self.args.show_fig).bind_value_to(self.args,'show_fig')
                            ui.number('figure width', value=self.args.fig_w, min=1, step=0.5, format='%.1f').bind_value_to(self.args, 'fig_w')
                            ui.number('figure height', value=self.args.fig_h, min=1, step=0.5, format='%.1f').bind_value_to(self.args, 'fig_h')
                            ui.number('DPI', value=self.args.dpi, min=1, step=1, format='%d').bind_value_to(self.args, 'dpi')
                            ui.input('figure file name', value=self.args.file_name).bind_value_to(self.args, 'file_name')
                    with ui.card():
                        ui.label(f'selected {len(self.dfs)} data files').classes('text-h6').bind_text_from(self, 'dfs', lambda dfs: f'selected {len(dfs)} data files')
                        self.make_fig()
        ## run GUI
        ui.run(host = 'localhost', port = 8011, title = 'HPLC Data Explorer', reload=False)
        

_str2func = {
    'plot-hplc': plot_hplc,
    'explore-hplc': explore_hplc
}


def main(sys_args: List[str] = None):
    args_paser = argparse.ArgumentParser()
    subparsers = args_paser.add_subparsers(title='subcommands', dest='sub_command')
    plot_hplc_args = plot_hplc.make_args(subparsers.add_parser('plot-hplc', description='plot hplc spectrum'))
    explore_hplc_args = explore_hplc.make_args(subparsers.add_parser('explore-hplc', description='explore hplc spectrum data'))

    excute_command(args_paser, sys_args, _str2func)

if __name__ == "__main__":
    # dev code, MUST COMMENT OUT BEFORE RELEASE
    # main('explore-hplc -i data_tmp/scripts/hplc'.split())
    
    main()