# Overview
========

This module provides a collection of functions and classes for managing concurrent tasks, handling progress information, and implementing task pools with different execution modes. It includes utilities for launching sub-threads, querying task statuses, and creating thread pools and task pools for efficient task management.

Functions
=========
\_wait\_for\_quit(statuesQue, key2action: Dict\[str, List\[Key2Action\]\]) -> int
---------------------------------------------------------------------------------
### Function Summary

Waits for user input and triggers corresponding actions based on predefined key-to-action mappings.
### Parameters

*   `statuesQue`: A queue for storing status information.
*   `key2action`: A dictionary mapping keys to lists of `Key2Action` objects, which define actions to be executed.

### Return Value

Returns `0` upon successful completion.
### Notes

*   This function listens for keyboard input and matches it with predefined actions.
*   If a key matches, the corresponding function is executed.
*   If no match is found, it attempts to match using regular expressions.

### Example

```python
key2action = {
'save': [Key2Action('running', some_function, [], {}, False, None)]
}
_wait_for_quit(statuesQue, key2action)
```

statues\_que\_opts(theQue, var\_name, opts, \*args) -> Any
----------------------------------------------------------
### Function Summary

Performs operations on a queue-based dictionary, such as getting or setting values.
### Parameters

*   `theQue`: The queue containing the dictionary.
*   `var_name`: The key to operate on.
*   `opts`: The operation to perform (e.g., `getValue`, `setValue`, `putValue`).
*   `*args`: Additional arguments for certain operations.

### Return Value

Returns the result of the operation, if applicable.
### Notes

*   This function modifies or retrieves values from a queue-based dictionary.
*   It supports various operations, including incrementing, decrementing, and setting values.

### Example

```python
result = statues_que_opts(statuesQue, 'some_key', 'getValue')
```

get\_input(promot: str = '', end='\\n') -> Any
----------------------------------------------
### Function Summary

Retrieves user input from a queue.
### Parameters

*   `promot`: The prompt to display before getting input.
*   `end`: The ending character for the prompt.

### Return Value

Returns the user input.
### Notes

*   This function waits for input to be available in the queue.

### Example

```python
user_input = get_input('Enter something: ')
```

launch\_sub\_thread(statuesQue=statuesQue, key2action: List\[Tuple\[str, Key2Action\]\] = \[\]) -> None
-------------------------------------------------------------------------------------------------------
### Function Summary

Launches a sub-thread to handle keyboard input and trigger actions.
### Parameters

*   `statuesQue`: The queue for status information.
*   `key2action`: A list of tuples containing keys and `Key2Action` objects.

### Return Value

None

### Notes

*   This function initializes the sub-thread and starts listening for keyboard input.

### Example

```python
launch_sub_thread(statuesQue, [('save', Key2Action('running', some_function, [], {}, False, None))])
```

show\_prog\_info(idx: int, sum: int = -1, freq: int = 10, otherInfo: str = '') -> None
--------------------------------------------------------------------------------------
### Function Summary

Prints progress information at regular intervals.
### Parameters

*   `idx`: The current index.
*   `sum`: The total number of items.
*   `freq`: The frequency of printing progress.
*   `otherInfo`: Additional information to display.

### Return Value

None

### Notes

*   This function prints progress information to the console.

### Example

```python
show_prog_info(10, 100, freq=5, otherInfo='Processing...')
```

# Classes
=======

Timer
-----
### Class Initialization

```python
timer = Timer()
```
### Initialization Method

Initializes a timer with the current time.
### Members

*   `lastTime`: The last recorded time.

### Methods

#### **call**() -> float

##### Method Summary

Returns the elapsed time since the last call and resets the timer.
##### Parameters

None

##### Return Value

The elapsed time in seconds.
##### Notes

*   This method updates the timer after each call.

##### Example

```python
elapsed_time = timer()
```

ThreadsPool
-----------
### Class Initialization

```python
pool = ThreadsPool(sum_threads, self_func, other_data, name='ThreadsPool')
```
### Initialization Method

Initializes a thread pool with a specified number of threads and a function to execute.
*   `sum_threads`: The number of threads.
*   `self_func`: The function to execute in each thread.
*   `other_data`: Additional data passed to the function.
*   `name`: The name of the thread pool.

### Members

*   `sumThreads`: The number of threads.
*   `sumTasks`: The total number of tasks.
*   `name`: The name of the thread pool.
*   `timer`: A timer object.
*   `sig`: A queue for signaling.
*   `putDataQues`: A list of queues for putting data.
*   `getDataQues`: A list of queues for getting data.

### Methods

#### put\_task(data) -> None

##### Method Summary

Puts a task into the task queue.
##### Parameters

*   `data`: The task data.

##### Return Value

None

##### Notes

*   This method distributes tasks among the threads.

##### Example

```python
pool.put_task(some_data)
```
#### loop2quit(wait2quitSignal) -> list

##### Method Summary

Waits for all tasks to complete and returns the results.
##### Parameters

*   `wait2quitSignal`: The signal to wait for.

##### Return Value

A list of results.
##### Notes

*   This method ensures all tasks are completed before returning.

##### Example

```python
results = pool.loop2quit(some_signal)
```

TaskPool
--------
### Class Initialization

```python
pool = TaskPool(mode='async', n_worker=None, sleep_while_empty=0.1)
```
### Initialization Method

Initializes a task pool with a specified mode and worker count.
*   `mode`: The execution mode (`async`, `thread`, `threads`, `process`, `isolated_process`).
*   `n_worker`: The number of worker threads or processes.
*   `sleep_while_empty`: The sleep time when the task queue is empty.

### Members

*   `MODE`: The execution mode.
*   `N_WORKER`: The number of workers.
*   `IS_STARTED`: A flag indicating if the pool is started.
*   `sleep_while_empty`: The sleep time when the task queue is empty.
*   `_async_loop`: The asyncio event loop.
*   `_thread_task_queue`: The task queue.
*   `_thread_result_queue`: The result queue.
*   `_thread_quit_event`: The quit event.
*   `thread`: The thread object.
*   `tasks`: A dictionary of tasks.
### Methods

#### add\_task(name: str, coro\_func, \*args, \*\*kwargs) -> str

##### Method Summary

Adds a task to the pool.
##### Parameters

*   `name`: The task name.
*   `coro_func`: The coroutine function.
*   `*args`: Additional arguments.
*   `**kwargs`: Additional keyword arguments.

##### Return Value

The task name.
##### Notes

*   This method adds a task to the pool and returns its name.

##### Example

```python
task_name = pool.add_task('task1', some_coroutine, arg1, arg2)
```
#### query\_task(name: str, block: bool = False, timeout: int = 3) -> Any

##### Method Summary

Queries the status or result of a task.
##### Parameters

*   `name`: The task name.
*   `block`: Whether to block until the task is completed.
*   `timeout`: The timeout in seconds.

##### Return Value

The task result or status.
##### Notes

*   This method retrieves the result or status of a task.

##### Example

```python
result = pool.query_task('task1', block=True, timeout=5)
```
#### map\_tasks(tasks: Union\[List\[Tuple\[List, Dict\]\], Dict\[str, Tuple\[List, Dict\]\]\], coro\_func: Callable, batch\_size: int = None, return\_result: bool = True, timeout: int = 3, wait\_busy: bool = False, \*\*kwargs) -> Union\[List\[Any\], Dict\[str, Any\]\]
##### Method Summary

Maps a list of tasks to a coroutine function.
##### Parameters

*   `tasks`: A list or dictionary of tasks.
*   `coro_func`: The coroutine function.
*   `batch_size`: The batch size for processing tasks.
*   `return_result`: Whether to return the results.
*   `timeout`: The timeout in seconds.
*   `wait_busy`: Whether to wait for the pool to be idle.
*   `**kwargs`: Additional keyword arguments.

##### Return Value

A list or dictionary of results.
##### Notes

*   This method processes tasks in batches and returns the results.

##### Example

```python
results = pool.map_tasks([(['arg1'], {'kwarg1': 'value1'})], some_coroutine, batch_size=5)
```
#### wait\_till(condition\_func, wait\_each\_loop: float = 0.5, timeout: float = None, verbose: bool = False, \*args, \*\*kwargs) -> Union\['TaskPool', bool\]

##### Method Summary

Waits until a condition is met or a timeout occurs.

##### Parameters

*   `condition_func`: The condition function to check.
    
*   `wait_each_loop`: The sleep time between checks.
    
*   `timeout`: The maximum time to wait in seconds.
    
*   `verbose`: Whether to display progress information.
    
*   `*args`: Additional arguments for the condition function.
    
*   `**kwargs`: Additional keyword arguments for the condition function.
    

##### Return Value

*   Returns `self` if the condition is met.
    
*   Returns `False` if the timeout is reached.
    

##### Notes

*   This method repeatedly checks the condition function until it returns `True` or the timeout is reached.
    

##### Example
```
pool.wait_till(lambda: some_condition, wait_each_loop=0.2, timeout=10)
```

#### wait\_till\_tasks\_done(task\_names: List\[str\], wait\_each\_loop: float = 0.5) -> Dict\[str, Union\[Any, Literal\[TaskStatus.NOT\_FOUND\], Literal\[TaskStatus.NOT\_FINISHED\]\]\]

##### Method Summary

Waits until specific tasks are completed.

##### Parameters

*   `task_names`: A list of task names to wait for.
    
*   `wait_each_loop`: The sleep time between checks.
    

##### Return Value

A dictionary with task names as keys and their results or status as values.

##### Notes

*   This method waits for the specified tasks to complete and returns their results.
    

##### Example
```
results = pool.wait_till_tasks_done(['task1', 'task2'])
```

#### clear(clear\_tasks: bool = True, clear\_queue: bool = True) -> List\[int\]

##### Method Summary

Clears the task pool, including tasks and queues.

##### Parameters

*   `clear_tasks`: Whether to clear the task dictionary.
    
*   `clear_queue`: Whether to clear the task and result queues.
    

##### Return Value

A list containing the sizes of the task dictionary, task queue, and result queue before clearing.

##### Notes

*   This method does not check whether the task pool is running.
    

##### Example
```
sizes = pool.clear()
```

#### close() -> None

##### Method Summary

Closes the task pool and joins the thread.

##### Parameters

None

##### Return Value

None

##### Notes

*   This method stops the event loop and joins the thread.
    

##### Example
```
pool.close()
```