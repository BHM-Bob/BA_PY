'''
Date: 2024-09-06 20:52:53
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2024-09-12 23:00:53
Description: 
'''
import argparse
import os
import time
from threading import Thread
from queue import Queue
from typing import Callable, Dict, List, Tuple, Union
from uuid import uuid4

import psutil
from nicegui import app, ui

os.environ['MBAPY_AUTO_IMPORT_TORCH'] = 'False'
os.environ['MBAPY_FAST_LOAD'] = 'True'
from mbapy.scripts._script_utils_ import Command, clean_path, excute_command


class SysInfo:
    def __init__(self, update_interval: float = 1.0) -> None:
        self._update_interval = update_interval
        self._info = {'CPU': [0], 'RAM': -1., 'GPU': [0], 'Disk': {}}
        self._cache = self._info.copy()
        self._que = Queue()
        self.host = Thread(target=self._run, name='sys_info_host')
        self.host.start()
        
    def _run(self):
        tik = time.time()
        while True:
            # get info
            self._info['CPU'] = psutil.cpu_percent(interval=self._update_interval, percpu=True)
            self._info['RAM'] = psutil.virtual_memory().percent
            self._info['Disk'] = {}
            for disk in psutil.disk_partitions():
                if disk.opts == 'cdrom':
                    continue
                try:
                    self._info['Disk'][disk.mountpoint] = psutil.disk_usage(disk.mountpoint).percent
                except PermissionError as e:
                    pass
            self._que.put(self._info.copy())
            # wait till interval
            while time.time() - tik < self._update_interval:
                time.sleep(self._update_interval - (time.time() - tik))
            tik = time.time()
            
    def get_info(self) -> Dict[str, Union[List[float], float, Dict[str, float]]]:
        if self._que.qsize() > 0:
            while self._que.qsize() > 1:
                self._que.get() # clean till only one left
            self._cache = self._que.get()
        return self._cache.copy()


class cmd_runner(Command):
    def __init__(self, args: argparse.Namespace, printf=print) -> None:
        super().__init__(args, printf)
        self.sys_info = SysInfo(update_interval=0.5)
        while self.sys_info._que.qsize() == 0:
            time.sleep(0.1)
        tik = time.time()
        self.sys_info_record = {tik-i: self.sys_info.get_info() for i in range(3 * 60)}  # remain 3 min records
        self.ui_cpu_chart = None
        self.ui_ram_chart = None
        
    @staticmethod
    def make_args(args: argparse.ArgumentParser):
        args.add_argument('--url', type=str, default='localhost',
                          help="server url, default is %(default)s.")
        args.add_argument('-p', '--port', type = int, default=8082,
                          help="server port, default is %(default)s.")
        return args
    
    
    def _get_sys_info_chart_data(self, name: str):
        _get_item_info = lambda n: list(map(lambda item: item[n], self.sys_info_record.values()))
        if name == 'CPU':
            return [{'name': f'CPU {i}', 'data': list(zip(range(len(self.sys_info_record)), usage))}\
                for i, usage in enumerate(zip(*_get_item_info('CPU')))]
        elif name == 'RAM':
            return [{'name': f'RAM', 'data': list(zip(range(len(self.sys_info_record)), _get_item_info('RAM')))}]
    
    def _update_gui(self):
        # if 
        if time.time() - max(self.sys_info_record.keys()) >= 1:
            del self.sys_info_record[min(self.sys_info_record.keys())]
            self.sys_info_record[time.time()] = self.sys_info.get_info()
            for chart, item in zip([self.ui_cpu_chart, self.ui_ram_chart], ['CPU', 'RAM']):
                chart.options['series'] = self._get_sys_info_chart_data(item)
                chart.update()
    
    def _build_new_task_gui(self):
        pass
    
    def _build_task_manager_gui(self):
        pass
    
    def _build_sys_info_gui(self):
        with ui.column().classes('w-full h-full'):
            with ui.row().classes('w-full h-1/2'):
                # CPU
                with ui.card().classes('w-2/5 h-full'):
                    self.ui_cpu_chart = ui.highchart({'title': 'CPU Usage', 'animation': False, 'marker': {'enable': False},
                                                      'series': self._get_sys_info_chart_data('CPU')}).classes('flex flex-grow')
                # RAM
                with ui.card().classes('w-2/5 h-full'):
                    self.ui_ram_chart = ui.highchart({'title': 'RAM Usage', 'animation': False, 'marker': {'enable': False},
                                                      'series': self._get_sys_info_chart_data('RAM')}).classes('flex flex-grow')
            with ui.row().classes('w-full h-1/2'):
                # GPU
                with ui.card().classes('w-2/5 h-full'):
                    pass
                # Disk
                with ui.card().classes('w-2/5 h-full'):
                    pass
            
        
    def main_process(self):
        with ui.header(elevated=True).style('background-color: #3874c8'):
            ui.label('mbapy-cli | Command Runnner').classes('text-h4')
            ui.space()
            ui.button('Exit', on_click=app.shutdown, icon='power')
        with ui.splitter(value=10).classes('w-full h-full') as splitter:
            with splitter.before:
                with ui.tabs().props('vertical active-bg-color=blue').classes('w-full') as tabs:
                    new_task_tab = ui.tab('Add Task')
                    task_manager_tab = ui.tab('Manage Tasks')
                    sys_info_tab = ui.tab('Sys Info')
            with splitter.after:
                with ui.tab_panels(tabs, value=new_task_tab) \
                        .props('vertical').classes('w-full h-full'):
                    with ui.tab_panel(new_task_tab):
                        self._build_new_task_gui()
                    with ui.tab_panel(task_manager_tab):
                        self._build_task_manager_gui()
                    with ui.tab_panel(sys_info_tab):
                        self._build_sys_info_gui()
        ui.timer(0.1, self._update_gui)
        ui.run(host = self.args.url, port = self.args.port, title = 'Command Runnner', reload=False)

_str2func = {
    'cmd-runner': cmd_runner,
}


def main(sys_args: List[str] = None):
    args_paser = argparse.ArgumentParser()
    subparsers = args_paser.add_subparsers(title='subcommands', dest='sub_command')
    cmd_runner_args = cmd_runner.make_args(subparsers.add_parser('cmd-runner', description='build cmd running scheduler'))

    excute_command(args_paser, sys_args, _str2func)

if __name__ == "__main__":
    # dev code, MUST COMMENT OUT BEFORE RELEASE
    main(['cmd-runner'])
    
    main()