# mbapy.web_utils.task

This module provides functions for launching a sub-thread to run a separate task concurrently with the main thread.  

### Queue
**General description**

A simple implementation of a queue data structure.

### Key2Action
**General description**

A named tuple representing an action triggered by a specific key.

#### Attributes
- `statue` (str): The signal for controlling the `_wait_for_quit` function, such as 'running' and 'exit'.
- `func` (Callable): The action function.
- `args` (list): The arguments for the action function.
- `kwgs` (dict): The keyword arguments for the action function.
- `is_reg` (bool, default=False): Indicates whether to use regular expression to match the key.
- `lock` (_thread.lock, default=None): Thread lock.

### _wait_for_quit
**General description**

Waits for keyboard input and triggers corresponding actions based on the input.

#### Params
- `statuesQue` (Queue): The queue for managing status information.
- `key2action` (Dict[str, List[Key2Action]]): A dictionary mapping keys to a list of Key2Action objects.

#### Returns
- int: 0 upon completion.

#### Notes
- If no match is found without using regular expressions, it will attempt to match using regular expressions.

#### Example
```python
statuesQue = Queue()
key2action = {
    's': [Key2Action('running', some_function, [arg1, arg2], {'kwarg1': val1})]
}
_wait_for_quit(statuesQue, key2action)
```

### statues_que_opts
**General description**

Performs various operations on the data stored in the queue.

#### Params
- `theQue` (Queue): The queue containing the data.
- `var_name` (str): The name of the variable to be operated on.
- `opts` (str): The operation to be performed.
- `args` (Any): Additional arguments for the operation.

#### Returns
- Any: The result of the operation, if applicable.

#### Notes
- The `opts` parameter can take the values: 'getValue', 'setValue', 'putValue', 'reduceBy', and 'addBy'.

#### Example
```python
result = statues_que_opts(statuesQue, "variable1", "getValue")
```

### get_input
**General description**

Gets input from the user.

#### Params
- `promot` (str, default=''): The prompt message.
- `end` (str, default='\n'): The ending character for the prompt message.

#### Returns
- Any: The user input.

### launch_sub_thread
**General description**

Launches a sub-thread to run a separate task concurrently with the main thread.

#### Params
- `statuesQue` (Queue): The queue for managing status information.
- `key2action` (List[Tuple[str, Key2Action]], default={}): A list of tuples representing key-to-action mappings.

#### Notes
- The `statuesQue` queue has two keys for internal usage: `__is_quit__` and `__inputs__`.
- Only if no match is found without using regular expressions, then it tries to match with regular expressions.

#### Example
```python
launch_sub_thread(statuesQue, [('e', Key2Action('exit', statues_que_opts, [statuesQue, "__is_quit__", "setValue", True], {})])
```

### show_prog_info
**General description**

Prints the progress information at regular intervals.

#### Params
- `idx` (int): The current index.
- `sum` (int, default=-1): The total number of items.
- `freq` (int, default=10): The frequency at which progress information is printed.
- `otherInfo` (str, default=''): Additional information to display.

#### Returns
- None

### Timer
**General description**

A simple timer class.

#### Methods
- `OnlyUsed() -> float`: Returns the time elapsed since the last call.
- `__call__() -> float`: Returns the time elapsed since the last call and updates the last call time.

### ThreadsPool
**General description**

A pool of threads for concurrent task execution.

#### Attributes
- `sumThreads` (int): The total number of threads in the pool.
- `sumTasks` (int): The total number of tasks executed.
- `name` (str): The name of the thread pool.
- `timer` (Timer): An instance of the Timer class.
- `sig` (Queue): The queue for sending signals.
- `putDataQues` (List[Queue]): The queues for putting data.
- `getDataQues` (List[Queue]): The queues for getting data.
- `quePtr` (int): Pointer for selecting the queue.

#### Methods
- `put_task(data) -> None`: Puts a task into the queue for execution.
- `loop2quit(wait2quitSignal) -> list`: Waits for all tasks to be completed and returns the results.

#### Notes
- The `self_func` parameter in the constructor is a function that takes four parameters: a queue for getting data, a queue for sending completed data to the main thread, a queue to send a quit signal when receiving a `wait2quitSignal`, and other data.

#### Example
```python
pool = ThreadsPool(5, some_function, other_data, 'MyPool')
pool.put_task(task_data)
results = pool.loop2quit('quit')
```