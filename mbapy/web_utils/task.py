
import _thread
import os
import time
from queue import Queue

statuesQue = Queue()

def _wait_for_quit(statuesQue,):
    flag = 1
    while flag:
        s = input()
        if s == "e":
            statues_que_opts(statuesQue, "quit", "setValue", True)
            flag = 0
        else:
            statues_que_opts(statuesQue, "input", "setValue", s)
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
    ret = statues_que_opts(statuesQue, "input", "getValue")
    while ret is None:
        time.sleep(0.1)
        ret = statues_que_opts(statuesQue, "input", "getValue")
    statues_que_opts(statuesQue, "input", "setValue", None)
    return ret
    
def launch_sub_thread(statuesQue = statuesQue):
    """
    Launches a sub-thread to run a separate task concurrently with the main thread.

    This function creates a global `statuesQue` queue and puts a dictionary with the keys `quit` and `input` into the queue. The `quit` key is set to `False` and the `input` key is set to `None`. 
    The function then starts a new thread by calling the `_wait_for_quit` function with the `statuesQue` queue as an argument. 
    Finally, the function prints the message "web sub thread started".
    """
    statuesQue.put(
        {
            "quit": False,
            "input": None,
        }
    )
    _thread.start_new_thread(_wait_for_quit, (statuesQue,))
    print('mbapy::web: web sub thread started')

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
            if statues_que_opts(statuesQue, "quit", "getValue"):
                print('get quit sig')
                return retList            
        for que in self.getDataQues:
            while not que.empty():
                retList.append(que.get())
        return retList
