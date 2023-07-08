'''
Author: BHM-Bob 2262029386@qq.com
Date: 2022-10-19 22:46:30
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-07-08 22:07:41
Description: 
'''
import sys, os
import time
from functools import wraps
import ctypes

import numpy as np


__NO_ERR__ = False
_Params = {
    'LAUNCH_WEB_SUB_THREAD':False,
}

def TimeCosts(runTimes:int = 1):
    """
    inner is func(times, *args, **kwargs)
    @TimeCosts(9)
    def f(idx, s):
        return s+idx
    print(f(8))\n
    print(f(s = 8))\n
    """
    def ret_wrapper( func ):
        def core_wrapper(*args, **kwargs):
            t0, ret = time.time(), []
            for times in range(runTimes):
                t1 = time.time()
                ret.append(func(times, *args, **kwargs))
                print(f'{times:2d} | {func.__name__:s} used {time.time()-t1:10.3f}s')
            print(f'{func.__name__:s} used {time.time()-t0:10.3f}s in total, {(time.time()-t0)/runTimes:10.3f}s by mean')
            return ret
        return core_wrapper
    return ret_wrapper

def autoparse(init):
    """
    Automatically assign property for __ini__() func
    Example
    ---------
    @autoparse
        def __init__(self, x):
            do something
    fixed from https://codereview.stackexchange.com/questions/269579/decorating-init-for-automatic-attribute-assignment-safe-and-good-practice
    """
    parnames = list(init.__code__.co_varnames[1:])
    defaults = init.__defaults__
    @wraps(init)
    def wrapped_init(self, *args, **kwargs):
        # remove the param who has no default value 
        # but in the end of the parnames
        if 'kwargs' in parnames and parnames[-1] == 'kwargs':
            parnames.remove('kwargs')
        if 'args' in parnames and parnames[-1] == 'args':
            parnames.remove('args')
        # Turn args into kwargs
        kwargs.update(zip(parnames[:len(args)], args))
        # apply default parameter values
        if defaults is not None:
            default_start = len(parnames) - len(defaults)
            for i in range(len(defaults)):
                if parnames[default_start + i] not in kwargs:
                    kwargs[parnames[default_start + i]] = defaults[i]
        # attach attributes to instance
        for arg in kwargs:
            setattr(self, arg, kwargs[arg])
        init(self, **kwargs)
    return wrapped_init

def rand_choose_times(choices_range:list[int] = [0,10], times:int = 100):
    """
    Generates a random sequence of integers within a given range and 
    counts the frequency of each number. Returns the most frequent number.
    
    :param choices_range: A list of two integers representing the lower and upper bounds of the range. Default is [0,10].
    :type choices_range: list[int]
    :param times: An integer representing the number of times the random sequence will be generated. Default is 100.
    :type times: int
    :return: An integer representing the most frequent number in the generated sequence.
    :rtype: int
    """
    randSeq = np.random.randint(low = choices_range[0], high = choices_range[1]+1, size = [times])
    count = [ np.sum(np.equal(randSeq,i)) for i in range(choices_range[0],choices_range[1]+1) ]
    return np.argmax(np.array(count))

def rand_choose(lst:list, seed = None):
    """
    Selects a random element from the given list.

    Parameters:
        lst (list): The list from which to select a random element.
        seed (int, optional): The seed value to use for random number generation. Defaults to None.

    Returns:
        Any: The randomly selected element from the list.
    """
    if lst is None:
        return put_err('lst is None', None)
    if seed is not None:
        np.random.seed(seed)
    return np.random.choice(lst)

def put_err(info:str, ret = None):
    if not __NO_ERR__:
        print(f'\nERR : {sys._getframe().f_code.co_name:s} : {info:s}\n')
    return ret
def put_log(info:str, head = "log", ret = None):
    print(f'\n{head:s} : {sys._getframe().f_code.co_name:s} : {info:s}\n')
    return ret

def get_time(chr:str = ':')->str:
    """
    Returns the current time as a string with a given character replacing the standard colon separator.

    :param chr: The character to replace the ':' separator. Default value is ':'.
    :type chr: str
    :return: A string that represents the current time with a given separator replacing the standard colon separator.
    :rtype: str
    """
    return time.asctime(time.localtime()).replace(':', chr)

class MyArgs():
    def __init__(self, args:dict) -> None:
        self.args = dict()
        args = self.get_args(args)
    def get_args(self, args:dict, force_update = True, del_origin = False):
        for arg_name in list(args.keys()):
            if arg_name in self.args and not force_update:
                pass
            else:
                setattr(self, arg_name, args[arg_name])
            if del_origin:
                del args[arg_name]
        return self
    def add_arg(self, arg_name:str, arg_value, force_update = True):
        setattr(self, arg_name, arg_value)
    def toDict(self):
        dic = {}
        for attr in vars(self):
            dic[attr] = getattr(self,attr)
        return dic   

def get_default_for_None(x, deault):
    """
    Return the default value when the input value is None.

    Args:
        x (Any): The input value to be checked.
        deault (Any): The default value to return if x is None.

    Returns:
        The input value if it is not None, otherwise the default value.
    """
    return x if x is not None else deault

def get_wanted_args(defalut_args:dict, kwargs:dict, del_kwargs = True):
    """
    wanted_args:dict with default value
    localVar = locals()
    """
    return MyArgs(defalut_args).get_args(kwargs, True, del_kwargs)
            
def split_list(lst:list, n = 1, drop_last = False):
    """
    return split sub lists.\n
    when drop_last is True and last one is less than n, drop the last one
    """
    result = [lst[i:i+n] for i in range(0, len(lst), n)]
    if drop_last and len(result[-1]) < n:
        del result[-1]
    return result

def get_dll_path_for_sys(module_name:str, sys_name: str = 'win'):
    if sys_name == 'win':
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bin', f'{module_name}.dll')
    else:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bin', f'{module_name}.so')

class MyDLL:
    @autoparse
    def __init__(self, path: str) -> None:
        """load a DLL file from path"""
        if not os.path.isfile(path):
            put_err(f'{path:s} is not exist')
        else:
            self.dll = ctypes.cdll.LoadLibrary(path)
        self.PTR = ctypes.POINTER # pointer generator
        self.REF = ctypes.byref # reference generator
        self.INT = ctypes.c_int # int type
        self.LONG = ctypes.c_long # long type
        self.FLOAT = ctypes.c_float # float type
        self.BOOL = ctypes.c_bool # bool type
        self.CHAR = ctypes.c_char # char type
        self.STR = ctypes.c_char_p # char* type
    def convert_c_lst(self, lst:list, c_type = ctypes.c_int):
        return (c_type * len(lst))(*lst)
    def convert_py_lst(self, c_lst:ctypes.POINTER(ctypes.c_int), size: int):
        return [c_lst[i] for i in range(size)]
    def get_func(self, func_name:str, func_args:list = None, func_ret = None):
        """
        Get a function by its name from the DLL object.

        Args:
            func_name (str): The name of the function to get.
            func_args (list, optional): The arguments of the function. Defaults to None.
            func_ret (Any, optional): The return type of the function. Defaults to None.

        Returns:
            function: The requested function object.
        """
        func = getattr(self.dll, func_name)
        if func_args is not None:
            func.argtypes = func_args
        if func_ret is not None:
            func.restype = func_ret
        return func    
    def free_ptr(self, ptr:ctypes.c_void_p, free_func:str = 'freePtr'):
        """
        Free the memory pointed to by `ptr` using the specified `free_func`.

        Parameters:
            ptr (ctypes.c_void_p): A pointer to the memory to be freed.
            free_func (str, optional): The name of the function to be used for freeing the memory. Defaults to 'free'.
        """
        if hasattr(self.dll, free_func):
            free_func = getattr(self.dll, free_func)
            free_func(ptr)
        else:
            put_err(f'{free_func:s} is not exist')


if __name__ == '__main__':
    # dev code
    pass