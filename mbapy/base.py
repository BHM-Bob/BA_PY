'''
Author: BHM-Bob 2262029386@qq.com
Date: 2022-10-19 22:46:30
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2024-11-30 17:15:30
Description: 
'''
import ctypes
import inspect
import json
import math
import os
import pathlib
import pkgutil
import platform
import sys
import time
from functools import wraps
from typing import Dict, List, Tuple, Union

import numpy as np

__NO_ERR__ = False

def get_num_digits(num:int):
    """
    Calculates the number of digits in a given integer.

    Args:
        num (int): The integer for which to calculate the number of digits.

    Returns:
        int: The number of digits in the given integer.
    """
    if num == 0:
        return 1
    else:
        return int(math.log10(abs(num))) + 1

def get_time(chr:str = ':')->str:
    """
    Returns the current time as a string, with the option to replace the ':' separator with a custom character.

    Parameters:
        chr (str): The character to replace the ':' separator with. Defaults to ':'.

    Returns:
        str: The current time as a string, with the ':' separator replaced with the custom character.
    """
    return time.asctime(time.localtime()).replace(':', chr)

def get_fmt_time(fmt = "%Y-%m-%d %H-%M-%S.%f", timestamp = None):
    """
    Returns a formatted string representing the given timestamp in the specified format.

    Parameters:
        - fmt (str): The format string to use for formatting the timestamp. Defaults to "%Y-%m-%d %H-%M-%S".
        - timestamp (float or None): The timestamp to format. If None, the current time will be used. Defaults to None.

    Returns:
        - str: The formatted timestamp string.
    """
    from datetime import datetime
    timestamp = get_default_call_for_None(timestamp, time.time)
    local_time = datetime.fromtimestamp(timestamp)
    date_str = local_time.strftime(fmt)
    return date_str

class _ConfigBase:
    def __init__(self) -> None:
        pass
    def to_dict(self):
        for attr in vars(self):
            self.__dict__[attr] = getattr(self, attr)
        for attr_name, attr_value in self.__dict__.items():
            if hasattr(attr_value, 'to_dict'):
                self.__dict__[attr_name] = attr_value.to_dict()        
        return self.__dict__

class _Config_File(_ConfigBase):
    def __init__(self) -> None:
        self.storage_dir = str(pathlib.Path(__file__).parent.resolve() / 'storage')

class _Config_Web(_ConfigBase):
    def __init__(self) -> None:
        self.auto_launch_sub_thread: bool = False
        if platform.system().lower() == 'windows':
            self.chrome_driver_path = os.path.expanduser("~/AppData/Local/Google/Chrome/Application/chromedriver.exe")
        elif platform.system().lower() == 'linux':
            self.chrome_driver_path = os.path.expanduser("~/bin/chromedriver")
        else:
            print(f'\nmbapy::_Config_Web: unkown os name: {platform.system().lower()}, use windows default chromedriver path\n')
            self.chrome_driver_path = os.path.expanduser("~/AppData/Local/Google/Chrome/Application/chromedriver.exe")
        self.request_header = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
    
class _Config(_ConfigBase):
    def __init__(self) -> None:
        self.err_warning_level: int = 0 # 0: all, no filter, 1: bapy parameter error, 2: bapy inner error... the bigger, the less error
        self.logs = []
        if pkgutil.find_loader('torch'):
            self.is_torch_avaliable = True
        else:
            self.is_torch_avaliable = False
        self.file = _Config_File()
        self.web = _Config_Web()
    
    def _get_default_config_path(self):
        return os.path.join(self.file.storage_dir, '_Config.json')

    def update(self, sub_module_name: str, attr_name: str, attr_value: Union[int, str, List, Dict],
               save_to_file: bool = True):
        """
        Update the attribute of a sub-module in the configuration.

        Parameters:
            sub_module_name (str): The name of the sub-module.
            attr_name (str): The name of the attribute to update.
            attr_value (Union[int, str, List, Dict]): The new value for the attribute. Can be an integer, a string, a list, or a dictionary.
            save_to_file (bool, optional): Indicates whether to save the updated configuration to a file. Defaults to True.

        Returns:
            None

        Raises:
            ValueError: If the sub_module_name is not supported.

        """
        support_module = ['file', 'web']
        if sub_module_name in support_module:
            sub_module = getattr(self, sub_module_name)
            setattr(sub_module, attr_name, attr_value)
            if save_to_file:
                json_str = json.dumps(self.to_dict(), indent=4)
                with open(self._get_default_config_path(),
                          'w', encoding='utf-8', errors='ignore') as json_file:
                    json_file.write(json_str)
            return True
        else:
            return put_err(f'{sub_module_name} not supported, only support: {", ".join(support_module)}', False)
        
    def save_to_file(self, config_file_path:str = None):
        """
        Saves the object to a file in JSON format.

        Args:
            config_file_path (str, optional): The path where the JSON file will be saved. If not provided, the default path will be used.

        Returns:
            dict: A dictionary representation of the object.
        """
        config_file_path = get_default_call_for_None(config_file_path, self._get_default_config_path)
        with open(config_file_path, 'w', encoding='utf-8', errors='ignore') as json_file:
            json_file.write(json.dumps(self.to_dict(), indent=4))
        return self.__dict__
        
    def load_from_file(self, force_update: bool = True, config_file_path:str = None, update_to_file = True):
        """
        Load configuration data from a file.

        Args:
            force_update (bool, optional): Flag to force update the configuration. Defaults to True.
            config_file_path (str, optional): Path to the configuration file. Defaults to None.
            update_to_file (bool, optional): Flag to update the configuration to file after updating. Defaults to True.

        Returns:
            dict: The loaded and updated configuration data.
        """
        config_file_path = get_default_call_for_None(config_file_path, self._get_default_config_path)
        with open(config_file_path, 'r', encoding='utf-8', errors='ignore') as json_file:
            config = json.loads(json_file.read())
            for sub_module_name, sub_configs in config.items():
                for sub_config in sub_configs:
                    if force_update or not hasattr(getattr(self, sub_module_name), sub_config):
                        setattr(getattr(self, sub_module_name), sub_config, sub_config)
        if update_to_file:
            self.save_to_file()
        return config

Configs = _Config()

def get_call_stack():
    """
    Returns the call stack of the current execution context.
    
    Returns:
        stack_info (list): A list containing the names of the functions in the call stack.
            such as: [<module>, lower_caller_name, ..., upper_caller_name_for_get_call_stack]
    """
    stack_info = []
    for frame in inspect.stack():
        stack_info.append(frame[0].f_code.co_name)
        if frame[0].f_code.co_name == '<module>':
            break
    return stack_info[1:][::-1]

def put_err(info:str, ret = None, warning_level = 0, _exit: Union[bool, int] = False):
    """
    Prints an error message along with the caller's name and arguments, if the warning level is greater than or equal to the error warning level specified in the Configs class.
        
    Parameters:
        info (str): The error message to be printed.
        ret (Any, optional): The return value of the function. Defaults to None.
        warning_level (int, optional): The warning level of the error. Defaults to 0.
    
    Returns:
        Any: The value specified by the ret parameter.
    
    Notes: 
        - It is recommended to set warning_level:
            - 0: normal error, will be closed easily.
            - 1: bapy parameter error.
            - 2: bapy inner error.
            - 3 or bigger: more important error.
        - It appends the log to the list of logs in the Configs class and prints it.
    """
    if warning_level >= Configs.err_warning_level:
        frame = inspect.currentframe().f_back
        caller_name = frame.f_code.co_name
        caller_args = inspect.getargvalues(frame).args
        err_str = f'\nERROR INFO : {caller_name:s} {caller_args}:\n {info:s}\n'
        print(err_str)
        Configs.logs.append(err_str)
    if not __NO_ERR__ and _exit:
        exit(_exit)
    return ret

def put_log(info:str, head = "bapy::log", ret = None):
    """
    Logs the given information with a formatted timestamp, call stack, and provided head. 
    Appends the log to the list of logs in the Configs class and prints it.
    
    Parameters:
        - info (str): The information to be logged.
        - head (str): The head of the log message. Default is "bapy::log".
        - ret : The value to return. Default is None.
    
    Returns:
        Any: The value specified by the ret parameter.
    """
    time_str = get_fmt_time()
    log_str = f'\n{head:s} {time_str:s}: {">".join(get_call_stack()[:-1]):s}: {info:s}\n'
    Configs.logs.append(log_str)
    print(log_str)
    return ret

def TimeCosts(runTimes:int = 1, log_per_iter = True):
    """
    A decorator function that measures and logs the time it takes for a function to run.

    Parameters:
        - runTimes (int): The number of times the function should be executed. Default is 1.
        - log_per_iter (bool): Whether to log the time taken for each iteration. Default is True.

    Returns:
        - ret_wrapper (function): The decorated function that measures and logs the time it takes for the original function to run.
    
    Notes:
        inner is func(times, *args, **kwargs)
        
    Examples:
        >>> @TimeCosts(9)
        >>> def f(idx, s):
        >>>     return s+idx
        >>> print(f(8))
        >>> print(f(s = 8))
    """
    def ret_wrapper( func ):
        def core_wrapper(*args, **kwargs):
            t0, ret = time.time(), []
            for times in range(runTimes):
                t1 = time.time()
                ret.append(func(times, *args, **kwargs))
                if log_per_iter:
                    print(f'{times:2d} | {func.__name__:s} used {time.time()-t1:10.3f}s')
            print(f'{func.__name__:s} used {time.time()-t0:10.3f}s in total, {(time.time()-t0)/runTimes:10.3f}s by mean')
            return ret
        return core_wrapper
    return ret_wrapper

def Timer():
    """
    A decorator that measures the execution time of a function and logs it with `mbapy.base.put_log`.
    Access logs by `mbapy.base.Config.logs: List[str]`

    Parameters:
        func (function): The function to be decorated.

    Returns:
        function: The decorated function.
    """
    def ret_wrapper(func):
        def core_wrapper(*args, **kwargs):
            t0 = time.time()
            ret = func(*args, **kwargs)
            put_log(f'{func.__name__:s} used {time.time()-t0:10.3f}s')
            return ret
        return core_wrapper
    return ret_wrapper

def autoparse(init):
    """
    Automatically assign property for __ini__() func.
    
    Note:
         - If has *args and **kwargs, ONLY support 'args' and 'kwargs' as the SPECIAL parameter name.
    
    Example:
    --------
    >>> @autoparse
    >>> def __init__(self, x):
    >>>     do something or do something with self.x
    
    fixed from https://codereview.stackexchange.com/questions/269579/decorating-init-for-automatic-attribute-assignment-safe-and-good-practice
    """
    parnames = list(init.__code__.co_varnames[1:init.__code__.co_argcount])
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

def check_parameters_path(path):
    return os.path.exists(path)
def check_parameters_none(arg):
    return arg is None
def check_parameters_None(arg):
    return arg is None
def check_parameters_len(arg):
    return len(arg) > 0
def check_parameters_bool(arg):
    return bool(arg)

def parameter_checker(*arg_checkers, raise_err = True, **kwarg_checkers):
    """
    A decorator that checks the validity of the arguments passed to a function.

    Args:
        *arg_checkers: Variable number of functions that check the validity of positional arguments.
        raise_err (bool): Flag indicating whether to raise a ValueError when an invalid argument is encountered. Defaults to True.
        **kwarg_checkers: Variable number of functions that check the validity of keyword arguments.

    Returns:
        A decorated function that performs argument validity checks before executing the original function.
        
    Notes:
        - If all checkers get passed, the original function is executed.
        - If any checker fails, the function returns None or raises a ValueError, depending on the value of `raise_err`.

    Example usage:
    >>> @check_arguments(path = check_parameters_path, head = check_parameters_len)
    >>> def my_function(path, len, head):
    >>>     # Function body
    >>>     pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            info_string = f"Parameter checker for {func.__code__.co_name} : Invalid value for argument "
            def check_arg(checker, arg_name: str, arg_value):
                if isinstance(checker, tuple):
                    checker, tmp_info_string = checker[0], checker[1]
                else:
                    tmp_info_string = info_string
                if not checker(arg_value):
                    if raise_err:
                        raise ValueError(tmp_info_string+arg_name)
                    else:
                        return put_err(tmp_info_string+arg_name, False)
                return True
            # args_name will exclude *args and **kwargs with name of args and kwargs
            args_name = list(func.__code__.co_varnames)[:func.__code__.co_argcount+
                                                 func.__code__.co_kwonlyargcount]
            # check positional arguments
            for i, arg_checker in enumerate(arg_checkers):
                if i < len(args):
                    if not check_arg(arg_checker, args_name[i], args[i]):
                        return None
            # check keyword arguments
            for arg_name, kwarg_checker in kwarg_checkers.items():
                # if passed the rigth arg name
                if arg_name in args_name:
                    # get the index of the argument
                    idx = args_name.index(arg_name)
                    # get arg value
                    if idx < len(args):
                        arg_value = args[idx]
                    elif arg_name in kwargs:
                        arg_value = kwargs[arg_name]
                    elif func.__defaults__ and idx-len(args) < len(func.__defaults__):
                        arg_value = func.__defaults__[idx-len(args)]
                    else:
                        raise TypeError(f'{func.__code__.co_name}() minssing required argument')
                    if not check_arg(kwarg_checker, arg_name, arg_value):
                        return None
            return func(*args, **kwargs)
        return wrapper
    return decorator

def rand_choose_times(choices_range:Tuple[int, int] = (0,10), times:int = 100):
    """
    Generates a random sequence of integers within a given range and 
    counts the frequency of each number. Returns the most frequent number.
    
    Parameters:
        - choices_range: A tuple of two integers representing the lower and upper bounds of the range. Default is [0,10].
        - times: An integer representing the number of times the random sequence will be generated. Default is 100.
        
    Returns:
        An integer representing the most frequent number in the generated sequence.
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
    if len(lst) == 0:
        return put_err('lst is 0 length', None)
    if seed is not None:
        np.random.seed(seed)
    return np.random.choice(lst)


def format_secs(sum_secs, fmt: str = None):
    """
    Format seconds into hours, minutes, and seconds.

    Parameters:
        sum_secs (int): The total number of seconds to be formatted.
        fmt (str, optional): The format string to be used for formatting the output. Defaults to None.

    Returns:
        tuple or str: If `fmt` is None, returns a tuple containing the formatted
                        hours, minutes, and seconds.
                      If `fmt` is a string, returns the formatted output using 
                        the provided format string with fmt.format(sum_hh, sum_mm, sum_ss).
                      If `fmt` is neither None nor a string, returns an error message.

    """
    sum_hh = int(sum_secs//3600)
    sum_mm = int((sum_secs-sum_hh*3600)//60)
    sum_ss = int(sum_secs-sum_hh*3600-sum_mm*60)
    if fmt is None:
        return sum_hh, sum_mm, sum_ss
    elif isinstance(fmt, str):
        return fmt.format(sum_hh, sum_mm, sum_ss)
    else:
        return put_err(f'unsupport fmt: {fmt}, return hh, mm, ss',
                       (sum_hh, sum_mm, sum_ss))

class MyArgs():
    def __init__(self, args:dict) -> None:
        self.__args = dict()
        self.get_args(args)
        
    def update_args(self, args:dict, force_update = True):
        for arg_name in list(args.keys()):
            if (arg_name not in self.__args) or force_update:
                setattr(self, arg_name, args[arg_name])
        return self
    
    def get_args(self, args:dict, force_update = True, del_kwargs = True):
        for arg_name, arg_value in args.items():
            if (arg_name not in self.__args) or force_update:
                setattr(self, arg_name, arg_value)
        if del_kwargs:
            for arg_name in self.__args:
                if arg_name in args:
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
    return x if x is not None else deault

def get_default_for_bool(x, deault):
    return x if bool(x) else deault

def get_default_call_for_None(x, deault_func, *args, **kwargs):
    return x if x is not None else deault_func(*args, **kwargs)

def get_wanted_args(defalut_args:dict, kwargs:dict, del_kwargs = True):
    """
    wanted_args:dict with default value
    localVar = locals()
    """
    return MyArgs(defalut_args).get_args(kwargs, True, del_kwargs)

def set_default_kwargs(kwargs: Dict, discard_extra: bool = False, **default_kwargs: Dict):
    """
    Set default keyword arguments in a dictionary.

    Args:
        kwargs (Dict): The dictionary of keyword arguments.
        discard_extra (bool, optional): Whether to discard extra parameters 
        that are in kwargs but not in default_kwargs. Defaults to False.
        **default_kwargs (Dict): The default keyword arguments.

    Returns:
        Dict: The updated dictionary of keyword arguments.
    """
    # del extra params, which is in kwargs but not in default_kwargs
    kwgs = kwargs.copy()
    if discard_extra:
        for name, value in kwargs.items():
            if name not in default_kwargs:
                kwgs.__delitem__(name)
    # set key-value pairs from default_kwargs to kwargs
    for name, value in default_kwargs.items():
        if name not in kwgs:
            kwgs[name] = value
    return kwgs

def get_default_args(kwargs: Dict, **default_kwargs: Dict):
    """
    Generate a dictionary of default arguments by combining the provided kwargs dictionary
    with the default_kwargs dictionary.
    
    Args:
        kwargs (Dict): A dictionary of keyword arguments.
        **default_kwargs (Dict): Any number of dictionaries containing default keyword arguments.
        
    Returns:
        Dict: A dictionary containing the combined default arguments.
    """
    kwgs = {}
    for name, value in default_kwargs.items():
        if name in kwargs:
            kwgs[name] = kwargs[name]
        else:
            kwgs[name] = value
    return kwgs
            
def split_list(lst:list, n = 1, drop_last = False):
    """
    return split sub lists.\n
    when drop_last is True and last one is less than n, drop the last one
    """
    result = [lst[i:i+n] for i in range(0, len(lst), n)]
    if drop_last and len(result[-1]) < n:
        del result[-1]
    return result

def get_storage_path(sub_path:str):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'storage', sub_path)

def get_dll_path_for_sys(module_name:str, **kwargs):
    if platform.system().lower() == 'windows':
        return get_storage_path(f'lib{module_name}.dll')
    elif platform.system().lower() == 'linux':
        return get_storage_path(f'lib{module_name}.so')
    else:
        return put_err(f'Unknown platform: {platform.system()}, return None', None)

class CDLL:
    @autoparse
    def __init__(self, path: str, winmode: int = 0) -> None:
        """load a DLL file from path"""
        if not os.path.isfile(path):
            put_err(f'{path:s} is not exist')
        else:
            # self.dll = ctypes.cdll.LoadLibrary(path)
            self.dll = ctypes.CDLL(path, winmode = winmode)
        # transfer ctype obj to c pointer
        self.ptr = ctypes.pointer # pointer generator
        self.ref = ctypes.byref # reference generator
        self.str = lambda s : bytes(s, 'utf-8') # string generator
        # c type in ctype
        self.PTR = ctypes.POINTER # pointer type
        self.INT = ctypes.c_int # int type
        self.LONG = ctypes.c_long # long type
        self.LONGLONG = ctypes.c_longlong # long long type
        self.ULL = ctypes.c_uint64 # unsigned long long type
        self.FLOAT = ctypes.c_float # float type
        self.BOOL = ctypes.c_bool # bool type
        self.CHAR = ctypes.c_char # char type
        self.STR = ctypes.c_char_p # char* type
        self.VOID = ctypes.c_void_p # void* type
    def convert_c_lst(self, lst:list, c_type = ctypes.c_int):
        return (c_type * len(lst))(*lst)
    def convert_py_lst(self, c_lst, size: int):
        return [c_lst[i][0] for i in range(size)]
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

@parameter_checker(check_parameters_len, raise_err=False)
def run_cmd(command: str):
    """
    Run a command and return the output as a string.

    Args:
        command (str): The command to run.

    Returns:
        str: The output of the command as a string.
    """
    return '\n'.join(os.popen(f'{command}').readlines())

__all__ = [
    '__NO_ERR__',
    'get_num_digits',
    'get_time',
    'get_fmt_time',
    'Configs',
    'get_call_stack',
    'put_log',
    'put_err',
    'TimeCosts',
    'Timer',
    'autoparse',
    'check_parameters_len',
    'check_parameters_path',
    'check_parameters_none',
    'check_parameters_None',
    'check_parameters_bool',
    'parameter_checker',
    'rand_choose_times',
    'rand_choose',
    'format_secs',
    'MyArgs',
    'get_default_for_None',
    'get_default_for_bool',
    'get_default_call_for_None',
    'get_wanted_args',
    'set_default_kwargs',
    'get_default_args',
    'split_list',
    'get_storage_path',
    'get_dll_path_for_sys',
    'CDLL',
    'run_cmd',    
]

if __name__ == '__main__':
    # dev code
    class Test:
        @parameter_checker(b = lambda b: b in ['b', 'B'], raise_err=False)
        def __init__(self, a, b: str = 'b', *args, **kwargs) -> None:
            print(a, b)
            
    Test('A')
    Test('A', 'A', e = 'E')
    Test('A', 'B', 10)