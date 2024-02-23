import _thread
import asyncio
import os
import re
import time
import threading
from collections import namedtuple
from enum import Enum
from queue import Queue
from typing import Any, Callable, Dict, List, Tuple, Union

from mbapy.base import put_err, put_log

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
        - key2action iwill add a ket 'e' first, and then add other key-to-action.
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
    NOT_FOUND = 1
    NOT_FINISHED = 2
    NOT_SUCCEEDED = 3
    NOT_RETURNED = 4

class CoroutinePool:
    """
    co-routine pool, use asyncio to run co-routines in a separate thread.
    """
    def __init__(self):
        self.loop = None
        self.thread = None
        self.tasks = {}
        self.TASK_NOT_FOUND = TaskStatus.NOT_FOUND        
        self.TASK_NOT_FINISHED = TaskStatus.NOT_FINISHED
        self.TASK_NOT_SUCCEEDED = TaskStatus.NOT_SUCCEEDED

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def add_task(self, name: str, coro_func, *args, **kwargs):
        if name == '' or name is None:
            name = f'{coro_func.__name__}-{time.time():.2f}'
        future = asyncio.run_coroutine_threadsafe(coro_func(*args, **kwargs), self.loop)
        self.tasks[name] = future
        return name

    def query_task(self, name):
        if name in self.tasks:
            future = self.tasks[name]
            if future.done():
                try:
                    result = future.result()
                    del self.tasks[name]
                    return result
                except Exception as e:
                    del self.tasks[name]
                    return self.TASK_NOT_SUCCEEDED, e
            else:
                return self.TASK_NOT_FINISHED
        else:
            return self.TASK_NOT_FOUND

    def run(self):
        self.loop = asyncio.new_event_loop()      
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        return self
    
    def check_empty(self):
        for task in self.tasks.values():
            if not task.done():
                return False
        return True
    
    def close(self):
        self.loop.close()


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
    'CoroutinePool'
]


if __name__ == '__main__':
    # dev code
    async def example_coroutine(name, seconds):
        print(f"Coroutine {name} started")
        await asyncio.sleep(seconds)
        print(f"Coroutine {name} finished after {seconds} seconds")
        return f"Coroutine {name} result"

    pool = CoroutinePool().run()
    pool.add_task("task1", example_coroutine, "task1", 3)
    pool.add_task("task2", example_coroutine, "task2", 5)

    print(pool.query_task("task1"))  # Output: TaskStatus.NOT_FINISHED
    print(pool.query_task("task2"))  # Output: TaskStatus.NOT_FINISHED

    # wait for tasks to finish
    import time
    time.sleep(6)

    print(pool.query_task("task1"))  # Output: Coroutine task1 finished after 3 seconds
    print(pool.query_task("task2"))  # Output: Coroutine task2 finished after 5 seconds
