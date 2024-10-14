import argparse
import os
import shutil
import tempfile
from copy import deepcopy
from functools import partial
from multiprocessing import Manager
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from nicegui import app, events, run, ui

from mbapy.game import BaseInfo
from mbapy.web_utils.task import TaskPool

os.environ['MBAPY_AUTO_IMPORT_TORCH'] = 'False'
os.environ['MBAPY_FAST_LOAD'] = 'True'

import cv2
from PIL import Image

from mbapy.file import get_paths_with_extension, opts_file, write_sheets
from mbapy.file_utils.video import get_cv2_video_attr
from mbapy.plot import get_palette, save_show
from mbapy.scripts._script_utils_ import Command, clean_path, excute_command


def _make_prop_size(w, h, fit=(2, 15), scale=100):
    while True:
        if fit[0] <= w/scale <= fit[1] and fit[0] <= h/scale <= fit[1]:
            return (int(w/scale), int(h/scale))
        scale -= 1

class ProcessTemplate(BaseInfo):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.th = 0
        self.boxes: Dict[str, Tuple[int, int, int, int]] = {}
        self.boxes_name: Dict[str, str] = {}
        self.boxes_color: Dict[str, str] = {}
        
    def add_box(self, x1, y1, x2, y2):
        uid = uuid4().hex[:8]
        self.boxes[uid] = (int(min(x1, x2)), int(min(y1, y2)), int(max(x1, x2)), int(max(y1, y2)))
        self.boxes_name[uid] = uid
        self.boxes_color[uid] = 'red'
        
    def get_box_web_repr(self, uid: str) -> str:
        x1, y1, x2, y2 = self.boxes[uid]
        return f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" fill="none" stroke="{self.boxes_color[uid]}" stroke-width="4" />'
    
    def get_web_repr(self) -> str:
        return ' '.join(self.get_box_web_repr(uid) for uid in self.boxes)


class VideoFile:
    def __init__(self, name: str, path: str) -> None:
        self.name = name
        self.path = path
        self.cv2_video = cv2.VideoCapture(self.path)
        self.fps = get_cv2_video_attr(self.cv2_video, 'FPS')
        self.frame_count = get_cv2_video_attr(self.cv2_video, 'FRAME_COUNT')
        self.width = get_cv2_video_attr(self.cv2_video, 'FRAME_WIDTH')
        self.height = get_cv2_video_attr(self.cv2_video, 'FRAME_HEIGHT')
        
    def read(self, frame_idx: int = None):
        if frame_idx is None:
            return self.cv2_video.read()
        else:
            self.cv2_video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            frame =  self.cv2_video.read()
            self.cv2_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return frame
        
    def close(self):
        self.cv2_video.release()
        
        
def process_video(video_name: str, video_path: str, template: ProcessTemplate, mp_queue):
    mp_queue.put((video_name, 'loading video'))
    video = VideoFile(video_name, video_path)
    result = {'name': video_name,
              'video': {'fps': video.fps, 'width': video.width, 'height': video.height},
              'template': template, 'results': {}}
    # process for each frame
    for idx in range(video.frame_count):
        # Read a frame from the video
        ret, frame = video.read()
        if not ret:
            break
        # Convert the frame to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, frame = cv2.threshold(frame, template.th, 255, cv2.THRESH_BINARY)
        # Find traget in the frame
        result['results'][idx] = {}
        for name, box in template.boxes.items():
            box_mat = frame[box[1]:box[3], box[0]:box[2]]
            # cv2.imshow('main', cv2.cvtColor(box_mat, cv2.COLOR_GRAY2BGR))
            if box_mat.shape[0] > 0 and box_mat.shape[1] > 0:
                # calcu center
                sum0, sum1 = box_mat.sum(axis=0), box_mat.sum(axis=1)
                if sum0.max() > 0:
                    center_x = np.argmax(box_mat.sum(axis=0))
                else:
                    center_x = -1
                if sum1.max() > 0:
                    center_y = np.argmax(box_mat.sum(axis=1))
                else:
                    center_y = -1
                    
                result['results'][idx][name] = {'x': center_x, 'y': center_y, 'sum': box_mat.sum()}
        # put a progress message into the queue
        if idx % 100 == 0:
            mp_queue.put((video.name, 100 * idx/video.frame_count))
    # perform post-analysis
    result['ana_datas'] = {}
    groups = list(set(map(lambda x:x.split(',')[0], template.boxes_name.values())))
    for group_idx, single_group in enumerate(groups):
        box_names = list(filter(lambda x:x.split(',')[0] == single_group, template.boxes_name.values()))
        name2uid = {name:uid for uid, name in template.boxes_name.items()}
        box_uids = [name2uid[name] for name in box_names]
        ana_datas = {'sums': {}, 'Time':[], 'x':{}, 'y':{}, 't':{}}
        ana_datas['Time'] = np.array(list(result['results'].keys())) / video.fps
        for uid in box_uids:
            ana_datas['sums'][template.boxes_name[uid]] = []
            ana_datas['x'][template.boxes_name[uid]] = []
            ana_datas['y'][template.boxes_name[uid]] = []
            ana_datas['t'][template.boxes_name[uid]] = []
            for t, v in result['results'].items():
                ana_datas['sums'][template.boxes_name[uid]].append(v[uid]['sum'])
                if v[uid]['x'] >= 0 and v[uid]['y'] >= 0:
                    ana_datas['x'][template.boxes_name[uid]].append(v[uid]['x']+template.boxes[uid][0])
                    ana_datas['y'][template.boxes_name[uid]].append(v[uid]['y']+template.boxes[uid][1])
                    ana_datas['t'][template.boxes_name[uid]].append(t/result['video']['fps'])
        result['ana_datas'][single_group] = ana_datas
        result['ana_datas']['box_uids'] = box_uids
        mp_queue.put((video.name, f'{group_idx+1}/{len(groups)} analyzed'))
    # release the video, put the result into the queue
    video.close()
    mp_queue.put((video.name, 'done'))
    mp_queue.put((video.name, result))


class auto_ccp(Command):
    def __init__(self, args: argparse.Namespace, printf=print) -> None:
        super().__init__(args, printf)
        self.SUPPORT_FMT = ['.avi', '.mov', '.mp4', '.mkv']
        self.task_pool = None
        self.mp_manager = Manager()
        self.mp_queue = self.mp_manager.Queue()
        self.video: Dict[str, VideoFile] = {}
        self.templates: Dict[str, ProcessTemplate] = {'tmp': ProcessTemplate()}
        self.batch: Dict[str, Dict[str, str]] = {}
        self.queue: Dict[str, Dict[str, Union[str, float]]] = {}
        self.results: Dict[str, Dict[str, Dict[str, Union[int, float]]]] = {}
        self.ana_figs: Dict[str, plt.Figure] = {'stack': None, 'bar': None, 'heatmap': None, 'traj': None}
        self.ana_dfs: Dict[str, pd.DataFrame] = {}
        self.ana_datas: Dict[str, Any] = {'x': {}, 'y': {}, 't': {}}
        self.ui_template = None
        self.ui_template_read_idx = 0
        self.ui_template_on_video = None
        self.ui_template_frame = None
        self.ui_template_click_first = None
        self.ui_ana_results = None
        self.ui_ana_group = None
        self.template_is_mouse_down = False
        self.new_template_name = None
        self.TMP_DIR = None
        
    @staticmethod
    def make_args(args: argparse.ArgumentParser):
        args.add_argument('-i', '--input', type = str, default='.',
                          help="input directory, default is %(default)s.")
        args.add_argument('-url', '--url', type = str, default='localhost',
                          help="url to connect to, default is %(default)s.")
        args.add_argument('-port', '--port', type = int, default=8012,
                          help="port to connect to, default is %(default)s.")
        args.add_argument('-mp', '--multi-process', type=int, default=4,
                          help="number of multi-process, default is %(default)s.")
        return args

    def process_args(self):
        self.input = clean_path(self.args.input)
        self.TMP_DIR = os.path.join(self.input, 'tmp')
        os.makedirs(self.TMP_DIR, exist_ok=True)

    def load_video(self, path):
        if any(path.endswith(ext) for ext in self.SUPPORT_FMT):
            name = Path(path).stem
            return VideoFile(name, path)
        
    def submit_task(self, name: str, e):
        if self.batch[name]['template'] is None:
            return ui.notify(f'Please select a template for {name}')
        self.task_pool.add_task(name, process_video, name, self.video[name].path,
                                deepcopy(self.templates[self.batch[name]['template']]), self.mp_queue)
        self.queue[name] = self.batch.pop(name)
        self.queue[name]['progress'] = 0.
        self.build_batch_tab_gui.refresh()
        self.build_queue_tab_gui.refresh()
        
    @ui.refreshable
    def build_batch_tab_gui(self):
        with ui.list().classes('w-full h-full').props('dense separator'):
            for name, video in self.video.items():
                if name in self.queue:
                    continue
                with ui.card().classes('w-full'):
                    with ui.row().classes('flex flex-grow'):
                        self.batch[name] = {'name': name, 'template': None, 'templates': None}
                        ui.label(name).classes('flex flex-grow text-h6')
                        ui.label(f'time: {video.frame_count/video.fps:.2f}s | {video.width}x{video.height}')
                        self.batch[name]['templates'] = ui.select(list(self.templates.keys())).bind_value_to(self.batch[name], 'template')
                        ui.button('submit', on_click=partial(self.submit_task, name))
                        
    def update_queue_ui(self):
        while self.mp_queue.qsize() > 0:
            name, result = self.mp_queue.get()
            if isinstance(result, dict):
                self.results[name] = result
                self.ui_ana_results.set_options(list(self.results.keys()))
            elif isinstance(result, float) or isinstance(result, str):
                self.queue[name]['progress'] = result
    @ui.refreshable
    def build_queue_tab_gui(self):
        with ui.list().classes('w-full h-full').props('dense separator'):
            for name, task in self.queue.items():
                with ui.card().classes('w-full'):
                    with ui.row().classes('flex flex-grow'):
                        ui.label(name).classes('flex flex-grow text-h6')
                        ui.label().bind_text_from(self.queue[name], 'progress', lambda x: f'progress: {x:.2f}%' if isinstance(x, float) else x)
    
    def _handle_template_frame_event(self, e: events.MouseEventArguments):
        if e.type == 'mousedown':
            self.ui_template_click_first = (e.image_x, e.image_y)
            self.template_is_mouse_down = True
        elif e.type == 'mouseup':
            self.template_is_mouse_down = False
            x1, y1 = self.ui_template_click_first
            x2, y2 = e.image_x, e.image_y
            self.templates['tmp'].add_box(x1, y1, x2, y2)
            self.ui_template_frame.set_content(self.templates['tmp'].get_web_repr())
        elif e.type =='mousemove':
            if self.template_is_mouse_down:
                x1, y1 = self.ui_template_click_first
                x2, y2 = e.image_x, e.image_y
                self.ui_template_frame.set_content(self.templates['tmp'].get_web_repr() + f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" fill="none" stroke="red" stroke-width="4" />')
        self.build_template_tmp_box_ui.refresh()
        
    def create_template(self, e):
        self.templates[self.new_template_name] = deepcopy(self.templates['tmp'])
        self.ui_template.set_options(list(self.templates.keys()), value=self.new_template_name)
        for name in self.batch:
            self.batch[name]['templates'].set_options(list(self.templates.keys()))
    
    @ui.refreshable
    def build_template_frame_gui(self):
        if self.ui_template_on_video.value is not None:
            _, frame = self.video[self.ui_template_on_video.value].read(self.ui_template_read_idx)
            if self.templates['tmp'].th > 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, frame = cv2.threshold(frame, self.templates['tmp'].th, 255, cv2.THRESH_BINARY)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            with tempfile.NamedTemporaryFile(dir=self.TMP_DIR, suffix='.png', delete=False) as f:
                Image.fromarray(frame).save(f)
                self.ui_template_frame = ui.interactive_image(f.name, on_mouse=self._handle_template_frame_event,
                                                              events=['mousedown', 'mouseup', 'mousemove'], cross=True).classes('flex flex-grow')
                self.ui_template_frame.set_content(self.templates['tmp'].get_web_repr())
                
    def _handle_template_tmp_box_delete(self, uid: str, e):
        self.templates['tmp'].boxes.pop(uid)
        self.build_template_tmp_box_ui.refresh()
        self.ui_template_frame.set_content(self.templates['tmp'].get_web_repr())
        
    def _handle_template_tmp_box_watch(self, uid: str, e):
        self.templates['tmp'].boxes_color[uid] = 'green' if self.templates['tmp'].boxes_color[uid] == 'red' else'red'
        self.ui_template_frame.set_content(self.templates['tmp'].get_web_repr())
        
    def _handle_template_select_change(self, e):
        self.templates['tmp'] = deepcopy(self.templates[e.value])
        self.build_template_frame_gui.refresh()
        self.tempalte_th.set_value(self.templates['tmp'].th)
                
    @ui.refreshable
    def build_template_tmp_box_ui(self):
        with ui.column().classes('flex flex-grow'):
            for uid, box in self.templates['tmp'].boxes.items():
                with ui.row().classes('flex flex-grow'):
                    ui.input(value=self.templates['tmp'].boxes_name[uid]).bind_value_to(self.templates['tmp'].boxes_name, uid)
                    ui.label(f'{box[0]}, {box[1]} -> {box[2]}, {box[3]}')
                    ui.button(icon='remove_red_eye').on_click(partial(self._handle_template_tmp_box_watch, uid))
                    ui.button(icon='delete').on_click(partial(self._handle_template_tmp_box_delete, uid))
                    
    def save_template(self):
        if self.ui_template.value is not None:
            self.templates[self.ui_template.value].to_json(os.path.join(self.input, f'{self.new_template_name}.cppPTemplate'))
        
    def build_template_tab_gui(self):
        with ui.row().classes('w-full'):
            self.ui_template = ui.select(list(self.templates.keys()), label='choose a template').classes('flex flex-grow').on_value_change(self._handle_template_select_change)
            ui.input('new template name', value='new_template_01').classes('flex flex-grow').bind_value_to(self, 'new_template_name')
            ui.button('Create Template', on_click=self.create_template).classes('flex flex-grow')
            ui.button('Save Template', on_click=self.save_template).classes('flex flex-grow')
        with ui.row().classes('w-full'):
            self.ui_template_on_video = ui.select(list(self.video.keys()), label='choose video').classes('flex flex-grow').on_value_change(self.build_template_frame_gui.refresh)
            ui.number(min=0, max=1000, value=100, label='read frame').classes('flex flex-grow').bind_value_to(self, 'ui_template_read_idx').on_value_change(self.build_template_frame_gui.refresh)
            self.tempalte_th = ui.number(min=0, max=255, value=0, label='frame threashold').classes('flex flex-grow').bind_value_to(self.templates['tmp'], 'th').on_value_change(self.build_template_frame_gui.refresh)
        with ui.splitter(value=80).classes('w-full h-full') as splitter:
            with splitter.before:
                self.build_template_frame_gui()
            with splitter.after:
                self.build_template_tmp_box_ui()
                
    @ui.refreshable
    def ana_make_stack_fig(self, result: Dict, uids: List[str]):
        if result is None or uids is None:
            return
        plt.close(self.ana_figs['stack'])
        with ui.pyplot(figsize=(12, 8), close=False) as fig:
            self.ana_figs['stack'] = fig.fig
            ax = self.ana_figs['stack'].gca()
            ax.stackplot(self.ana_datas['Time'], self.ana_datas['sums'].values(), labels=self.ana_datas['sums'].keys(), alpha=0.8)
            ax.legend()
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Box Sum')
            ax.set_title('Box Sum Stack Line')
    
    @ui.refreshable
    def ana_make_bar_fig(self, result: Dict, uids: List[str]):
        if result is None or uids is None:
            return 
        plt.close(self.ana_figs['bar'])
        sums = self.ana_datas['sums']
        df = pd.DataFrame(data=sums)
        zeros = np.zeros((len(df), len(sums.keys())))
        max_idx = np.argmax(df.loc[:, list(sums.keys())], axis=-1)
        for i in np.unique(max_idx):
            zeros[max_idx==i, i] = 1
        df.loc[:, list(sums.keys())] = zeros
        self.ana_dfs['bar'] = df
        with ui.pyplot(figsize=(12, 8), close=False) as fig:
            self.ana_figs['bar'] = fig.fig
            ax = self.ana_figs['bar'].gca()
            ax.bar(sums.keys(), df.loc[:, list(sums.keys())].sum() / result['video']['fps'], label=sums.keys())
            ax.legend()
            ax.set_ylabel('Time (s)')
            ax.set_title('Box Sum Bar')
    
    @ui.refreshable
    def ana_make_heatmap_fig(self, result: Dict, uids: List[str]):
        if result is None or uids is None:
            return 
        plt.close(self.ana_figs['heatmap'])
        x = self.ana_datas['x']
        y = self.ana_datas['y']
        with ui.pyplot(close=False) as fig:
            self.ana_figs['heatmap'] = fig.fig
            axs = fig.fig.subplots(1, len(uids))
            for axi, xi, yi, uid in zip(axs if len(uids)>1 else [axs], x.values(), y.values(), uids):
                hb = axi.hexbin(xi, yi, gridsize=50, cmap='coolwarm')
                axi.set(xlim=(result['template'].boxes[uid][0], result['template'].boxes[uid][2]), ylim=(result['template'].boxes[uid][1], result['template'].boxes[uid][3]))
                axi.set_title(f'{result["template"].boxes_name[uid]}')
                axi.set_title(f'{result["template"].boxes_name[uid]}')
                xlim = (result['template'].boxes[uid][0], result['template'].boxes[uid][2])
                ylim = (result['template'].boxes[uid][1], result['template'].boxes[uid][3])
                axi.set(xlim=xlim, ylim=ylim)
                prob_size = _make_prop_size(xlim[1]-xlim[0], ylim[1]-ylim[0])
                axi.figure.set_figwidth(prob_size[0])
                axi.figure.set_figheight(prob_size[1])
                cb = fig.fig.colorbar(hb, ax=axi)
    
    @ui.refreshable
    def ana_make_traj_fig(self, result: Dict, uids: List[str]):
        if result is None or uids is None:
            return 
        plt.close(self.ana_figs['traj'])
        x = self.ana_datas['x']
        y = self.ana_datas['y']
        t = self.ana_datas['t']
        with ui.pyplot(close=False) as fig:
            self.ana_figs['traj'] = fig.fig
            axs = fig.fig.subplots(1, len(uids))
            for axi, ti, xi, yi, uid in zip(axs if len(uids)>1 else [axs], t.values(), x.values(), y.values(), uids):
                cmap = colormaps['coolwarm']
                if all(len(item_v)>0 for item_v in [xi, yi, ti]):
                    norm = Normalize(vmin=min(ti), vmax=max(ti))
                    colors = cmap(ti)[:-1] # [N-1]
                    points_i = np.stack([xi, yi], axis=0).transpose(1, 0) # [N, 2]
                    segments_i = np.concatenate([points_i[:-1], points_i[1:]], axis=1).reshape(-1, 2, 2) # [N-1, 2, 2]
                    line = LineCollection(segments_i, alpha=0.8, colors=colors, linewidths=2, linestyles='solid')
                    axi.add_collection(line)
                axi.set_title(f'{result["template"].boxes_name[uid]}')
                xlim = (result['template'].boxes[uid][0], result['template'].boxes[uid][2])
                ylim = (result['template'].boxes[uid][1], result['template'].boxes[uid][3])
                axi.set(xlim=xlim, ylim=ylim)
                prob_size = _make_prop_size(xlim[1]-xlim[0], ylim[1]-ylim[0])
                axi.figure.set_figwidth(prob_size[0])
                axi.figure.set_figheight(prob_size[1])
                cb = fig.fig.colorbar(line, ax=axi, cmap=cmap, norm=norm)
           
    async def run_analysis(self):
        if self.ui_ana_results.value is None or self.ui_ana_group.value is None:
            result, box_uids = None, None
        else:
            result = self.results[self.ui_ana_results.value]
            self.ana_datas, box_uids = result['ana_datas'][self.ui_ana_group.value], result['ana_datas']['box_uids']
            self.ana_make_stack_fig.refresh(result, box_uids)
            self.ana_make_bar_fig.refresh(result, box_uids)
            self.ana_make_heatmap_fig.refresh(result, box_uids)
            self.ana_make_traj_fig.refresh(result, box_uids)
    
    def _handle_ui_ana_results_change(self, e):
        template = self.results[e.value]['template']
        self.ui_ana_group.set_options(list(set(map(lambda x:x.split(',')[0], template.boxes_name.values()))))
    
    def build_ana_tab_gui(self):
        with ui.column().classes('flex flex-grow'):
            with ui.row().classes('w-full'):
                self.ui_ana_results = ui.select(list(self.results.keys()), label='choose a result').classes('flex flex-grow').on_value_change(self._handle_ui_ana_results_change)
                self.ui_ana_group = ui.select(label='group by', options=[]).classes('flex flex-grow')
                ui.button('Analyze', on_click=self.run_analysis).classes('flex flex-grow')
            with ui.column().classes('w-full'):
                with ui.tabs().classes('flex flex-grow justify-center active-bg-color=blue') as tabs:
                    stack_panel = ui.tab('boxes sum stack line').props('no-caps').classes('flex flex-grow')
                    bar_panel = ui.tab('boxes sum bar').props('no-caps').classes('flex flex-grow')
                    heatmap_panel = ui.tab('box heatmap').props('no-caps').classes('flex flex-grow')
                    traj_panel = ui.tab('trajectory').props('no-caps').classes('flex flex-grow')
                with ui.tab_panels(tabs, value=stack_panel).classes('flex flex-grow'):
                    with ui.tab_panel(stack_panel).classes('flex flex-grow'):
                        self.ana_make_stack_fig(None, None)
                    with ui.tab_panel(bar_panel).classes('flex flex-grow'):
                        self.ana_make_bar_fig(None, None)
                    with ui.tab_panel(heatmap_panel).classes('flex flex-grow'):
                        self.ana_make_heatmap_fig(None, None)
                    with ui.tab_panel(traj_panel).classes('flex flex-grow'):
                        self.ana_make_traj_fig(None, None)
        
    def make_gui(self):
        with ui.header(elevated=True).style('background-color: #3874c8'):
            ui.label('mbapy-cli BioHelper | Auto CCP').classes('text-h4')
            ui.space()
            ui.button('Exit', on_click=app.shutdown, icon='power')
        with ui.splitter(value=10).classes('w-full h-full') as splitter:
            with splitter.before:
                with ui.tabs().props('vertical').classes('w-full') as tabs:
                    tab_batch = ui.tab('Batch').props('no-caps')
                    tab_queue = ui.tab('Queue').props('no-caps')
                    tab_template = ui.tab('Template').props('no-caps')
                    tab_ana = ui.tab('Analysis').props('no-caps')
            with splitter.after:
                with ui.tab_panels(tabs, value=tab_batch) \
                        .props('vertical').classes('w-full h-full'):
                    with ui.tab_panel(tab_batch).classes('w-full'):
                        self.build_batch_tab_gui()
                    with ui.tab_panel(tab_queue).classes('w-full'):
                        self.build_queue_tab_gui()
                        ui.timer(0.5, self.update_queue_ui)
                    with ui.tab_panel(tab_template).classes('w-full'):
                        self.build_template_tab_gui()
                    with ui.tab_panel(tab_ana).classes('w-full'):
                        self.build_ana_tab_gui()
        ui.run(host = self.args.url, port = self.args.port, title = 'Auto CCP', reload=False)
    
    def main_process(self):
        # set task pool
        self.task_pool = TaskPool('process', self.args.multi_process).start()
        print(f'task pool created with {self.args.multi_process} processes')
        # load video
        for path in get_paths_with_extension(self.input, self.SUPPORT_FMT):
            video = self.load_video(path)
            self.video[video.name] = video
        # load ProcessTemplate
        for path in get_paths_with_extension(self.input, '.cppPTemplate'):
            template = ProcessTemplate().from_json(path)
            self.templates[str(Path(path).stem)] = template
        # GUI
        self.make_gui()
        shutil.rmtree(self.TMP_DIR)

_str2func = {
    'auto-ccp': auto_ccp,
}


def main(sys_args: List[str] = None):
    args_paser = argparse.ArgumentParser()
    subparsers = args_paser.add_subparsers(title='subcommands', dest='sub_command')
    auto_ccp_args = auto_ccp.make_args(subparsers.add_parser('auto-ccp', description='CCP video analysis'))

    if __name__ in ['__main__', 'mbapy.scripts.mass']:
        # '__main__' is debug, 'mbapy.scripts.mass' is user running
        excute_command(args_paser, sys_args, _str2func)


if __name__ in {"__main__", "__mp_main__"}:
    # dev code, MUST COMMENT OUT BEFORE RELEASE
    main('auto-ccp -i data_tmp/scripts/ccp'.split(' '))
    
    main()