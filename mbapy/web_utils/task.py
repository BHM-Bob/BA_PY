import _thread
import asyncio
import multiprocessing
import os
import re
import threading
import time
from collections import namedtuple
from enum import Enum
from functools import partial
from queue import Queue
from typing import Any, Callable, Dict, List, Literal, Set, Tuple, Union
from uuid import uuid4

from tqdm import tqdm

if __name__ == '__main__':
    # dev mode
    from mbapy.base import parameter_checker, put_err, put_log
else:
    from ..base import parameter_checker, put_err, put_log

statuesQue = Queue()
Key2Action = namedtuple('Key2Action', ['statue', 'func', 'args', 'kwgs',
                                       'is_reg', 'lock'],
                        defaults=[False, None])

def _wait_for_quit(statuesQue, key2action: Dict[str, List[Key2Action]]):
    flag, reg_k2a = 'running', {}
    # find key using reg
    for key, actions in key2action.items():
        for action in actions:
            if action.is_reg and key in reg_k2a:
                reg_k2a[key].append(action)
            else:
                reg_k2a[key] = [action]
    # listenning keyboard
    while flag != 'exit':
        # auto wait for keyboard
        s = input()
        # try match without reg
        if s in key2action:
            for action in key2action[s]:
                if action.lock:
                    with action.lock:
                        action.func(*action.args, **action.kwgs)
                else:
                    action.func(*action.args, **action.kwgs)
                flag = action.statue
        # try match with reg ONLY IF get no match witout reg
        elif reg_k2a:
            for key, actions in reg_k2a.items():
                match = re.match(key, s)
                if match:
                    for action in actions:
                        if action.lock:
                            with action.lock:
                                action.func(*action.args, **action.kwgs)
                        else:
                            action.func(*action.args, **action.kwgs)
                        flag = action.statue
        else:
            statues_que_opts(statuesQue, "__inputs__", "setValue", s)
    return 0

def statues_que_opts(theQue, var_name, opts, *args):
    """opts contain:
    getValue: get varName value
    setValue: set varName value
    putValue: put varName value to theQue
    reduceBy: varName -= args[0]
    addBy: varName += args[0]
    """
    dataDict, ret = theQue.get(), None
    if var_name in dataDict.keys():
        if opts in ["getValue", "getVar"]:
            ret = dataDict[var_name]
        elif opts in ["setValue", "setVar"]:
            dataDict[var_name] = args[0]
        elif opts == "reduceBy":
            dataDict[var_name] -= args[0]
        elif opts == "addBy":
            dataDict[var_name] += args[0]
        else:
            print("do not support {" "} opts".format(opts))
    elif opts == "putValue":
        dataDict[var_name] = args[0]
    else:
        print("do not have {" "} var".format(var_name))
    theQue.put(dataDict)
    return ret

def get_input(promot:str = '', end = '\n'):
    if len(promot) > 0:
        print(promot, end = end)
    ret = statues_que_opts(statuesQue, "__inputs__", "getValue")
    while ret is None:
        time.sleep(0.1)
        ret = statues_que_opts(statuesQue, "__inputs__", "getValue")
    statues_que_opts(statuesQue, "__inputs__", "setValue", None)
    return ret
    
def launch_sub_thread(statuesQue = statuesQue,
                      key2action: List[Tuple[str, Key2Action]] = []):
    """
    Launches a sub-thread to run a separate task concurrently with the main thread.
    
    Note:
        - statuesQue has two keys for mbapy inner usage: __is_quit__ and __inputs__.
        - key2action will add a key 'e' first, and then add other key-to-action.
            The 'e' key will trigle the 'exit' signal to _wait_for_quit func.
        - NOLY IF get no match without reg, then try to match with reg.
    
    Parameters:
        - statuesQue: ...
        - key2action(List[Tuple[str, Key2Action]]): key-to-action
            - key(str): keyboard inputs to trigle this action, such as 'save'.
            - key2action: Key2Action
                - inner_signal(str): signal for control _wait_for_quit func, such as ‘running' and 'exit'.  
                - func(Callable): action func.  
                - *args(list): *args for action func.  
                - **kwargs(dict): **kwargs for action func.
                - is_reg(bool): use reg to match, defalut to False. NOLY IF get no match without reg, then try to match with reg.
                - lock(_thread.lock...): thread lock, defalut to None.

    This function creates a global `statuesQue` queue and puts a dictionary with the keys `quit` and `input` into the queue. The `quit` key is set to `False` and the `input` key is set to `None`. 
    The function then starts a new thread by calling the `_wait_for_quit` function with the `statuesQue` queue as an argument. 
    Finally, the function prints the message "web sub thread started".
    """
    statuesQue.put(
        {
            "__is_quit__": False,
            "__inputs__": None,
            "__signal__": None,
        }
    )
    k2a = {
        'e': [Key2Action('exit', statues_que_opts, [statuesQue, "__is_quit__", "setValue", True], {})],
    }
    for item in key2action:
        if item[0] in k2a:
            put_log(f'add a new action to exists key {item[0]}')
            k2a[item[0]].append(item[1])
        else:
            k2a[item[0]] = [item[1]]
    _thread.start_new_thread(_wait_for_quit, (statuesQue, k2a))
    put_log('web sub thread launched')

def show_prog_info(idx:int, sum:int = -1, freq:int = 10, otherInfo:str = ''):
    """
    Print the progress information at regular intervals.

    Parameters:
    - idx (int): The current index.
    - sum (int, default=-1): The total number of items.
    - freq (int, default=10): The frequency at which progress information is printed.
    - otherInfo (str, default=''): Additional information to display.

    Returns:
    None
    """
    if idx % freq == 0:
        print(f'\r {idx:d} / {sum:d} | {otherInfo:s}', end = '')

class Timer:
    def __init__(self, ):
        self.lastTime = time.time()

    def OnlyUsed(self, ):
        return time.time() - self.lastTime

    def __call__(self) -> float:
        uesd = time.time() - self.lastTime
        self.lastTime = time.time()
        return uesd

class ThreadsPool:
    """self_func first para is a que for getting data,
    second is a que for send done data to main thread,
    third is que to send quited sig when get wait2quitSignal,
    fourth is other data \n
    _thread.start_new_thread(func, (self.ques[idx], self.sig, ))
    """
    def __init__(self, sum_threads:int, self_func, other_data, name = 'ThreadsPool') -> None:
        self.sumThreads = sum_threads
        self.sumTasks = 0
        self.name = name
        self.timer = Timer()
        self.sig = Queue()
        self.putDataQues = [ Queue() for _ in range(sum_threads) ]
        self.getDataQues = [ Queue() for _ in range(sum_threads) ]
        self.quePtr = 0
        for idx in range(sum_threads):
            _thread.start_new_thread(self_func,
                                     (self.putDataQues[idx],
                                      self.getDataQues[idx],
                                      self.sig,
                                      other_data, ))

    def put_task(self, data) -> None:
        self.putDataQues[self.quePtr].put(data)
        self.quePtr = ((self.quePtr + 1) if ((self.quePtr + 1) < self.sumThreads) else 0)
        self.sumTasks += 1
        
    def loop2quit(self, wait2quitSignal) -> list:
        """ be sure that all tasks sended, this func will
        send 'wait to quit' signal to every que,
        and start to loop waiting"""
        retList = []
        for idx in range(self.sumThreads):
            self.putDataQues[idx].put(wait2quitSignal)
        while self.sig._qsize() < self.sumThreads:
            sumTasksTillNow = sum([self.putDataQues[idx]._qsize() for idx in range(self.sumThreads)])
            print(f'\r{self.name:s}: {sumTasksTillNow:d} / {self.sumTasks:d} -- {self.timer.OnlyUsed():8.1f}s')
            for que in self.getDataQues:
                while not que.empty():
                    retList.append(que.get())
            time.sleep(1)
            if statues_que_opts(statuesQue, "__is_quit__", "getValue"):
                print('get quit sig')
                return retList            
        for que in self.getDataQues:
            while not que.empty():
                retList.append(que.get())
        return retList
    
class TaskStatus(Enum):
    SUCCEED = 0
    NOT_FOUND = 1
    NOT_FINISHED = 2
    NOT_SUCCEEDED = 3
    NOT_RETURNED = 4
    TIME_OUT = 5
    ZERO_LEFT = 6

class TaskPool:
    """
    task pool, use asyncio to run co-routines in a separate thread OR just run normal tasks in a separate thread.
    
    Attributes:
        - mode (str, default='async'): 'async' or 'thread', use asyncio or threading to run a pool.
        - loop (asyncio.loop): asyncio event loop.
        - thread (threading.Thread): threading thread.
        - tasks (Dict[str, asyncio.Future]): task name to future.
        - TASK_NOT_FOUND (TaskStatus): signal FLAG, task not found.
        - TASK_NOT_FINISHED (TaskStatus): signal FLAG, task not finished.
        - TASK_NOT_SUCCEEDED (TaskStatus): signal FLAG, task not succeeded.
        
    Methods:
    """
    @parameter_checker(mode = lambda mode: mode in ['async', 'thread',
                                                    'threads', 'process'])
    def __init__(self, mode: str = 'async', n_worker: int = None,
                 sleep_while_empty: float = 0.1):
        """
        Parameters:
            - mode (str, default='async'): 'async' or 'thread', use asyncio or threading to run a pool.
                - If it's IO heavy and has suitable coroutine function, use 'async'
                - If it's IO heavy and only has normal function, and wants run ONE task at ONCE, use 'thread'. Use Queue to cache tasks and run ONE task at ONCE.
                - If it's IO heavy and only has normal function, and wants run MULTI tasks at ONCE, use 'threads'. Use Queue to cache tasks and run MULTI tasks at ONCE.
                - If it's CPU heavy and wants run MULTI tasks at ONCE, use 'process'. Use multiprocessing.Pool to run MULTI tasks at ONCE.
            - n_worker (int, default=None): number of worker threads or processes.
            - sleep_while_empty (float, default=0.1): sleep time in a loop while task queue is empty.
        """
        if mode in ['async', 'thread'] and n_worker is not None:
            put_err(f'n_worker should be None when mode is {mode}, skip')
        self.MODE = mode
        self.N_WORKER = n_worker
        self._async_loop: asyncio.AbstractEventLoop = None
        self._thread_task_queue: Queue = Queue()
        self._thread_result_queue: Queue = Queue()
        self._thread_quit_event: threading.Event = threading.Event()
        self.thread: threading.Thread = None
        self.tasks = {}
        self.TASK_NOT_FOUND = TaskStatus.NOT_FOUND        
        self.TASK_NOT_FINISHED = TaskStatus.NOT_FINISHED
        self.TASK_NOT_SUCCEEDED = TaskStatus.NOT_SUCCEEDED
        self.TIME_OUT = TaskStatus.TIME_OUT
        self.NO_TASK_LEFT = TaskStatus.ZERO_LEFT

    def _run_async_loop(self):
        asyncio.set_event_loop(self._async_loop)
        self._async_loop.run_forever()
        
    def _run_thread_loop(self):
        while not self._thread_quit_event.is_set():
            if not self._thread_task_queue.empty():
                task_name, task_func, task_args, task_kwargs = self._thread_task_queue.get()
                try:
                    result = task_func(*task_args, **task_kwargs)
                    self._thread_result_queue.put((task_name, result, TaskStatus.SUCCEED))
                except Exception as e:
                    self._thread_result_queue.put((task_name, e, TaskStatus.NOT_SUCCEEDED))
            time.sleep(0.1)
            
    def _run_process_loop(self):
        pool, tasks_cache = multiprocessing.Pool(self.N_WORKER), {}
        while not self._thread_quit_event.is_set():
            # put async result into cache
            if not self._thread_task_queue.empty():
                task_name, task_func, task_args, task_kwargs = self._thread_task_queue.get()
                tasks_cache[task_name] = pool.apply_async(task_func, task_args, task_kwargs)
            # get finished tasks from cache
            for task_name, task_result in list(tasks_cache.items()):
                if task_result.ready(): # 无论任务是否因异常结束，ready()都返回True
                    try:
                        self._thread_result_queue.put((task_name, task_result.get(),
                                                       TaskStatus.SUCCEED))
                    except Exception as e:
                        self._thread_result_queue.put((task_name, e, TaskStatus.NOT_SUCCEEDED))
                    del tasks_cache[task_name]
            time.sleep(0.1)
        pool.close()

    def _add_task_async(self, name: str, coro_func, *args, **kwargs):
        future = asyncio.run_coroutine_threadsafe(coro_func(*args, **kwargs), self._async_loop)
        future.add_done_callback(partial(self._query_async_task_callback, task_name=name))
        return name
    
    def _add_task_thread(self, name: str, func, *args, **kwargs):
        self._thread_task_queue.put((name, func, args, kwargs))
        return name
    
    def add_task(self, name: str, coro_func, *args, **kwargs):
        # check name
        if name == '' or name is None:
            name = f'{coro_func.__name__}-{uuid4()}'
        if name in self.tasks:
            put_err(f'Task {name} already exists, replace it with the new one')
        # map mode to function
        mode = 'async' if self.MODE == 'async' else 'thread'
        self.tasks[name] = TaskStatus.NOT_RETURNED
        return getattr(self, f'_add_task_{mode}')(name, coro_func, *args, **kwargs)

    def _query_async_task_callback(self, future: asyncio.Future, task_name: str):
        try:
            self._thread_result_queue.put((task_name, future.result(),
                                           TaskStatus.SUCCEED))
        except Exception as e:
            put_err(f'error {e} when get result from queue') 
            self._thread_result_queue.put((task_name, e, TaskStatus.NOT_SUCCEEDED))
            
    def _query_task_queue(self, block: bool = True, timeout: int = 3):
        while not self._thread_result_queue.empty():
            try:
                _name, result, statue = self._thread_result_queue.get(block, timeout)
                self.tasks[_name] = (_name, result, statue)
            except Exception as e:
                put_err(f'error {e} when get result from queue')
        
    def query_task(self, name: str, block: bool = False, timeout: int = 3):
        """
        Parameters:
            - name (str): task name.
            - block (bool, default=False): when is true, block until get result or timeout.
            - timeout (int, default=3): timeout in seconds.
            
        Returns:
            - Case 1: task not found, return TaskStatus.NOT_FOUND.
            - Case 2: task not finished and block is Flase, return TaskStatus.NOT_FINISHED.
            - Case 3: task not finished and block is True, wait for timeout seconds, return TaskStatus.TIME_OUT.
            - Case 4: task finished with succeed or failed, not block or block but return in time, return the result.
        """
        # short-cut for not found
        st_tick = time.time()
        if name not in self.tasks:
            return self.TASK_NOT_FOUND
        # retrive finished results
        self._query_task_queue(block=block, timeout=timeout)
        # check if not return, succeed, or not succeed
        if self.tasks[name] == TaskStatus.NOT_RETURNED:
            if block and time.time() - st_tick < timeout:
                is_retuened = False
                while not is_retuened:
                    self._query_task_queue(block=block, timeout=timeout)
                    is_retuened = self.tasks[name] != TaskStatus.NOT_RETURNED
                    if time.time() - st_tick > timeout:
                        return self.TIME_OUT
            else:
                return self.TASK_NOT_FINISHED
        _name, result, statue = self.tasks[name]
        del self.tasks[name]
        if statue == TaskStatus.NOT_SUCCEEDED:
            put_err(f'Task {name} failed with {result}, return {result}')
        return result
    
    def query_single_task_from_tasks(self, tasks_name: List[str], block: bool = False, timeout: int = 3):
        """
        Parameters:
            - tasks_name (List[str]): task names. If any task not found, ignore it.
            - block (bool, default=False): when is true, block until get result or timeout.
            - timeout (int, default=3): timeout in seconds.
            
        Returns:
            - Case 1: all tasks not found or finished, return TaskStatus.ZERO_LEFT.
            - Case 2: all tasks not returned and timeput with block option, return TaskStatus.TIME_OUT.
            - Case 3: all tasks not returned and block is False, return TaskStatus.NOT_FINISHED.
            - Case 4: one or more tasks finished with succeed or failed, return one result.
        """
        st_tick = time.time()
        # Case 1 return
        if all(name not in self.tasks for name in tasks_name):
            return self.NO_TASK_LEFT
        # Case 2 return
        self._query_task_queue(block=block, timeout=timeout)
        if time.time() - st_tick > timeout:
            return self.TIME_OUT
        # Case 4 return
        for name in tasks_name:
            if name in self.tasks and self.tasks[name] != TaskStatus.NOT_RETURNED:
                _name, result, statue = self.tasks[name]
                del self.tasks[name]
                if statue == TaskStatus.NOT_SUCCEEDED:
                    put_err(f'Task {name} failed with {result}, return {result}')
                return result
        # Case 3 return
        return self.TASK_NOT_FINISHED
    
    def count_waiting_tasks(self):
        """
        - for async mode, return the number of unfinished tasks.
        - for thread, threads, and process mode, return the number of the un-begined tasks.
        """
        if self.MODE == 'async':
            return len([None for task in self.tasks.values() if not task.done()])
        elif self.MODE in ['thread', 'threads', 'process']:
            return self._thread_task_queue.qsize()
        
    def count_done_tasks(self):
        """INCLUDE succeed and failed tasks."""
        self._query_task_queue(block=False)
        in_que = self._thread_result_queue.qsize()
        in_dict = len([None for task in self.tasks.values() if task != TaskStatus.NOT_RETURNED])
        return in_que + in_dict

    def run(self):
        """launch the thread and event loop"""
        if self.MODE == 'async':
            self._async_loop = asyncio.new_event_loop()
        if self.MODE == 'threads':
            self.thread = []
            for _ in range(self.N_WORKER):
                self.thread.append(threading.Thread(target=self._run_thread_loop, daemon=True))
                self.thread[-1].start()
        else: # async, thread and process only need one thread
            self.thread = threading.Thread(target=getattr(self, f'_run_{self.MODE}_loop'), daemon=True)
            self.thread.start()
        return self
    
    def check_empty(self):
        """check tasks (undo and done) is empty"""
        if self.MODE == 'async':
            for task in self.tasks.values():
                if not task.done():
                    return False
            return True
        elif self.MODE in ['thread', 'threads', 'process']:
            return self._thread_task_queue.empty()
        
    def wait_till(self, condition_func, wait_each_loop: float = 0.5,
                  timeout: float = None, verbose: bool = False, *args, **kwargs):
        """
        wait till condition_func return True, or timeout.
        
        Parameters:
            - condition_func (Callable): a function that return True or False.
            - wait_each_loop (float, default=0.1): sleep time in a loop while waiting.
            - timeout (float, default=None): timeout in seconds.
            - *args, **kwargs: other args for condition_func.
        
        Returns:
            - pool(TaskPool): retrun self for chaining.
            - False: means timeout.
        """
        st = time.time()
        if verbose:
            bar =tqdm(desc='waiting', total=len(self.tasks), initial=self.count_done_tasks())
        while not condition_func(*args, **kwargs):
            if timeout is not None and time.time() - st > timeout:
                return False
            done = self.count_done_tasks()
            if verbose:
                bar.set_description(f'done/sum: {done}/{len(self.tasks)}')
                bar.update(self.count_done_tasks() - bar.n)
            time.sleep(wait_each_loop)
        return self
    
    def wait_till_tasks_done(self, task_names: List[str],
                             wait_each_loop: float = 0.5) -> Dict[str, Union[Any, Literal[TaskStatus.NOT_FOUND], Literal[TaskStatus.NOT_FINISHED]]]:
        self.wait_till(lambda names: names.issubset(set([r[0] for r in self.tasks.values() if r != TaskStatus.NOT_RETURNED])),
                       wait_each_loop = wait_each_loop, verbose=False, names=set(task_names))
        return {name: self.query_task(name) for name in task_names}
    
    def close(self):
        """close the thread and event loop, join the thread"""
        # close async pool
        if self.MODE == 'async':
            self._async_loop.call_soon_threadsafe(self._async_loop.stop)
            self._async_loop.close()
        # close thread and join it
        self._thread_quit_event.set()
        if self.MODE in ['async', 'thread', 'process']:
            self.thread.join()
        elif self.MODE == 'threads':
            [t.join() for t in self.thread]


__all__ = [
    'Key2Action',
    'statuesQue',
    '_wait_for_quit',
    'statues_que_opts',
    'get_input',
    'launch_sub_thread',
    'show_prog_info',
    'Timer',
    'ThreadsPool',
    'TaskStatus',
    'TaskPool'
]


if __name__ == '__main__':
    # dev code
    async def example_coroutine(name, seconds):
        print(f"Coroutine {name} started")
        await asyncio.sleep(seconds)
        print(f"Coroutine {name} finished after {seconds} seconds")
        return f"Coroutine {name} result"

    pool = TaskPool().run()
    pool.add_task("task1", example_coroutine, "task1", 2)
    pool.add_task("task2", example_coroutine, "task2", 4)
