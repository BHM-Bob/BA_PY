# mbapy.base

This module provides various utility functions and classes for general use.  

## Functions

### put_err(info:str, ret = None) -> Any

Put an error message and return a value.  

Parameters:  
- info (str): The error message.  
- ret (Any, optional): The value to return. Defaults to None.  

Returns:  
- Any: The value to return.  

Example:  
```python
put_err('An error occurred', None)
```

### put_log(info:str, head = "log", ret = None) -> Any

Put a log message and return a value.  

Parameters:  
- info (str): The log message.  
- head (str, optional): The log header. Defaults to "log".  
- ret (Any, optional): The value to return. Defaults to None.  

Returns:  
- Any: The value to return.  

Example:  
```python
put_log('This is a log message', 'info', None)
```

### TimeCosts(runTimes:int = 1) -> Callable

A decorator that measures the time taken by a function.  

Parameters:  
- runTimes (int, optional): The number of times to run the function. Defaults to 1.  

Returns:  
- Callable: The decorated function.  

Example:  
```python
@TimeCosts(9)
def f(idx, s):  
    return s+idx

print(f(8))
print(f(s=8))
```

### autoparse(init: Callable) -> Callable

A decorator that automatically assigns properties for the `__init__()` function.  

Parameters:  
- init (Callable): The `__init__()` function to decorate.  

Returns:  
- Callable: The decorated function.  

Example:  
```python
@autoparse
def __init__(self, x):  
    # do something
```

### check_parameters_path(path: str) -> bool

Check if a path exists.  

Parameters:  
- path (str): The path to check.  

Returns:  
- bool: True if the path exists, False otherwise.  

Example:  
```python
check_parameters_path('./data/file.txt')
```

### check_parameters_none(arg: Any) -> bool

Check if an argument is None.  

Parameters:  
- arg (Any): The argument to check.  

Returns:  
- bool: True if the argument is None, False otherwise.  

Example:  
```python
check_parameters_none(None)
```

### check_parameters_len(arg: List[Any]) -> bool

Check if the length of a list is greater than 0.  

Parameters:  
- arg (List[Any]): The list to check.  

Returns:  
- bool: True if the length of the list is greater than 0, False otherwise.  

Example:  
```python
check_parameters_len([1, 2, 3])
```

### check_parameters_bool(arg: bool) -> bool

Check if a boolean value is True.  

Parameters:  
- arg (bool): The boolean value to check.  

Returns:  
- bool: True if the boolean value is True, False otherwise.  

Example:  
```python
check_parameters_bool(True)
```

### parameter_checker(*arg_checkers, raise_err = True, **kwarg_checkers) -> Callable

A decorator that checks the validity of the arguments passed to a function.  

Parameters:  
- *arg_checkers: Variable number of functions that check the validity of positional arguments.  
- raise_err (bool): Flag indicating whether to raise a ValueError when an invalid argument is encountered. Defaults to True.  
- **kwarg_checkers: Variable number of functions that check the validity of keyword arguments.  

Returns:  
- Callable: The decorated function.  

Example:  
```python
@parameter_checker(path = check_parameters_path, head = check_parameters_len)
def my_function(path, len, head):  
    # Function body
```

### rand_choose_times(choices_range:List[int] = [0,10], times:int = 100) -> int

Generates a random sequence of integers within a given range and counts the frequency of each number. Returns the most frequent number.  

Parameters:  
- choices_range (List[int], optional): A list of two integers representing the lower and upper bounds of the range. Defaults to [0,10].  
- times (int, optional): An integer representing the number of times the random sequence will be generated. Defaults to 100.  

Returns:  
- int: The most frequent number in the generated sequence.  

Example:  
```python
rand_choose_times([0, 10], 100)
```

### rand_choose(lst: List[Any], seed = None) -> Any

Selects a random element from the given list.  

Parameters:  
- lst (List[Any]): The list from which to select a random element.  
- seed (int, optional): The seed value to use for random number generation. Defaults to None.  

Returns:  
- Any: The randomly selected element from the list.  

Example:  
```python
rand_choose([1, 2, 3, 4, 5])
```

### format_secs(sumSecs: int) -> Tuple[int, int, int]

Formats a given number of seconds into hours, minutes, and seconds.  

Parameters:  
- sumSecs (int): An integer representing the total number of seconds.  

Returns:  
- Tuple[int, int, int]: A tuple containing three integers representing the number of hours, minutes, and seconds respectively.  

Example:  
```python
format_secs(3600)
```

### MyArgs

A class that represents a collection of arguments.  

Methods:  
- __init__(self, args: dict): Initializes the MyArgs object with the given arguments.  
- get_args(self, args: dict, force_update=True, del_origin=False): Gets the arguments and updates the MyArgs object.  
- add_arg(self, arg_name: str, arg_value, force_update=True): Adds an argument to the MyArgs object.  
- toDict(self): Converts the MyArgs object to a dictionary.  

Example:  
```python
args = MyArgs({'x': 1, 'y': 2})
args.get_args({'y': 3})
args.add_arg('z', 4)
args.toDict()
```

### get_default_for_None(x, default) -> Any

Returns the default value if the given value is None.  

Parameters:  
- x (Any): The value to check.  
- default (Any): The default value to return.  

Returns:  
- Any: The default value if the given value is None, otherwise the given value.  

Example:  
```python
get_default_for_None(None, 0)
```

### get_default_for_bool(x, default) -> bool

Returns the default value if the given boolean value is False.  

Parameters:  
- x (bool): The boolean value to check.  
- default (bool): The default value to return.  

Returns:  
- bool: The default value if the given boolean value is False, otherwise the given value.  

Example:  
```python
get_default_for_bool(False, True)
```

### get_default_call_for_None(x, default_func, *args, **kwargs) -> Any

Calls the default function if the given value is None.  

Parameters:  
- x (Any): The value to check.  
- default_func (Callable): The default function to call.  
- *args: Variable number of positional arguments to pass to the default function.  
- **kwargs: Variable number of keyword arguments to pass to the default function.  

Returns:  
- Any: The result of the default function if the given value is None, otherwise the given value.  

Example:  
```python
get_default_call_for_None(None, time.time)
```

### get_wanted_args(default_args: dict, kwargs: dict, del_kwargs=True) -> MyArgs

Gets the wanted arguments from the given keyword arguments.  

Parameters:  
- default_args (dict): A dictionary with default values for the wanted arguments.  
- kwargs (dict): The keyword arguments to check.  
- del_kwargs (bool, optional): Flag indicating whether to delete the checked keyword arguments. Defaults to True.  

Returns:  
- MyArgs: A MyArgs object containing the wanted arguments.  

Example:  
```python
get_wanted_args({'x': 1, 'y': 2}, {'y': 3})
```

### split_list(lst: List[Any], n=1, drop_last=False) -> List[List[Any]]

Splits a list into sublists of size n.  

Parameters:  
- lst (List[Any]): The list to split.  
- n (int, optional): The size of each sublist. Defaults to 1.  
- drop_last (bool, optional): Flag indicating whether to drop the last sublist if its size is less than n. Defaults to False.  

Returns:  
- List[List[Any]]: A list of sublists.  

Example:  
```python
split_list([1, 2, 3, 4, 5, 6], 2)
```

### get_storage_path(sub_path: str) -> str

Gets the storage path for a given sub path.  

Parameters:  
- sub_path (str): The sub path.  

Returns:  
- str: The storage path.  

Example:  
```python
get_storage_path('data/file.txt')
```

### get_dll_path_for_sys(module_name: str, **kwargs) -> str

Gets the DLL path for the current system.  

Parameters:  
- module_name (str): The name of the module.  
- **kwargs: Additional keyword arguments.  

Returns:  
- str: The DLL path.  

Example:  
```python
get_dll_path_for_sys('my_module')
```

### MyDLL

A class that represents a DLL.  

Methods:  
- __init__(self, path: str): Initializes the MyDLL object with the given path.  
- convert_c_lst(self, lst: List[int], c_type: Type) -> Any: Converts a Python list to a C array.  
- convert_py_lst(self, c_lst: Any, size: int) -> List[int]: Converts a C array to a Python list.  
- get_func(self, func_name: str, func_args: List[Type] = None, func_ret: Type = None) -> Callable: Gets a function from the DLL object.  
- free_ptr(self, ptr: Any, free_func: str = 'freePtr') -> None: Frees the memory pointed to by a pointer.  
- Example:  
```python
dll = MyDLL('./my_module.dll')
dll.convert_c_lst([1, 2, 3], ctypes.c_int)
dll.convert_py_lst(c_lst, 3)
dll.get_func('my_function', [ctypes.c_int, ctypes.c_int], ctypes.c_int)
dll.free_ptr(ptr)
```

### get_time(chr: str = ':') -> str

Returns the current time as a string.  

Parameters:  
- chr (str, optional): The character to replace the ':' separator with. Defaults to ':'.  

Returns:  
- str: The current time as a string.  

Example:  
```python
get_time()
```

### get_fmt_time(fmt: str = "%Y-%m-%d %H-%M-%S", timestamp = None) -> str

Returns a formatted string representing the given timestamp.  

Parameters:  
- fmt (str, optional): The format string to use for formatting the timestamp. Defaults to "%Y-%m-%d %H-%M-%S".  
- timestamp (float or None, optional): The timestamp to format. If None, the current time will be used. Defaults to None.  

Returns:  
- str: The formatted timestamp string.  

Example:  
```python
get_fmt_time("%Y-%m-%d %H:%M:%S", 1634668800)
```

## Classes

### MyArgs

A class that represents a collection of arguments.  

Methods:  
- __init__(self, args: dict): Initializes the MyArgs object with the given arguments.  
- get_args(self, args: dict, force_update=True, del_origin=False): Gets the arguments and updates the MyArgs object.  
- add_arg(self, arg_name: str, arg_value, force_update=True): Adds an argument to the MyArgs object.  
- toDict(self): Converts the MyArgs object to a dictionary.  

Example:  
```python
args = MyArgs({'x': 1, 'y': 2})
args.get_args({'y': 3})
args.add_arg('z', 4)
args.toDict()
```

### _Config

A class that represents the configuration settings.  

Methods:  
- __init__(self): Initializes the _Config object.  
- _get_default_config_path(self): Gets the default configuration file path.  
- update(self, sub_module_name: str, attr_name: str, attr_value: Union[int, str, List, Dict], save_to_file: bool = True) -> bool: Updates the attribute of a sub-module in the configuration.  
- save_to_file(self, config_file_path: str = None) -> dict: Saves the object to a file in JSON format.  
- load_from_file(self, force_update: bool = True, config_file_path: str = None, update_to_file=True) -> dict: Loads configuration data from a file.  

Example:  
```python
config = _Config()
config.update('file', 'storage_dir', './data')
config.save_to_file()
config.load_from_file()
```

### Configs

An instance of the _Config class.  

Example:  
```python
Configs.update('file', 'storage_dir', './data')
Configs.save_to_file()
Configs.load_from_file()
```

## Constants

### __NO_ERR__

A global variable that controls whether to display error messages. Set to False to display error messages.  

Example:  
```python
__NO_ERR__ = False
```