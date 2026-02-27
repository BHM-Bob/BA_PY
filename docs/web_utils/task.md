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
### Class Overview

TaskPool is a task management class that provides concurrent task execution capabilities using different execution modes. It supports running coroutines in a separate thread using asyncio, or running normal tasks using threading or multiprocessing.

### Class Initialization

```python
pool = TaskPool(mode='async', n_worker=None, sleep_while_empty=0.1, report_error=False, mp_pool_init_kwargs=None)
```

Initializes a task pool with a specified mode and worker count.

**Parameters:**

*   `mode` (str, default='async'): The execution mode. Available options:
    *   `'async'`: Uses asyncio event loop to run coroutines concurrently. Suitable for IO-heavy tasks with coroutine functions. This is the default and most performant mode for async IO.
    *   `'thread'`: Uses threading with Queue to cache tasks and runs ONE task at a time. Suitable for IO-heavy tasks with normal (non-coroutine) functions.
    *   `'threads'`: Uses threading with Queue to cache tasks and runs MULTI tasks simultaneously (determined by n_worker). Suitable for IO-heavy tasks that need parallel execution.
    *   `'process'`: Uses multiprocessing.Pool to run MULTI tasks in separate processes. Suitable for CPU-heavy tasks that need true parallel execution, bypassing the GIL.
    *   `'isolated_process'`: Not yet implemented.
*   `n_worker` (int, default=None): The number of worker threads or processes. Only valid for 'threads' and 'process' modes. Must be None for 'async', 'thread', and 'isolated_process' modes.
*   `sleep_while_empty` (float, default=0.1): The sleep time (in seconds) in the loop while the task queue is empty. Used to reduce CPU usage when waiting for new tasks.
*   `report_error` (bool, default=False): Whether to report errors immediately when a task fails. Only valid when mode is 'thread', 'threads', or 'process'.
*   `mp_pool_init_kwargs` (Dict[str, Any], default=None): Additional keyword arguments to pass to multiprocessing.Pool initialization. Only valid when mode is 'process'.

### Members

**Public Attributes:**

*   `MODE` (str): The execution mode of the task pool.
*   `N_WORKER` (int): The number of worker threads or processes. None for 'async', 'thread', and 'isolated_process' modes.
*   `IS_STARTED` (bool): A flag indicating whether the pool has been started.
*   `sleep_while_empty` (float): The sleep time when the task queue is empty.
*   `REPORT_ERROR` (bool): Whether to report errors immediately when a task fails.
*   `mp_pool_init_kwargs` (Dict[str, Any]): Additional arguments for multiprocessing.Pool initialization.
*   `thread` (Union[threading.Thread, List[threading.Thread]]): The worker thread(s) that execute tasks.
*   `tasks` (Dict[str, Any]): A dictionary mapping task names to their status or result.

**Class Variables (TaskStatus Flags):**

*   `TASK_NOT_FOUND` (TaskStatus.NOT_FOUND): Returned when the task is not found.
*   `TASK_NOT_FINISHED` (TaskStatus.NOT_FINISHED): Returned when the task is not finished.
*   `TASK_NOT_SUCCEEDED` (TaskStatus.NOT_SUCCEEDED): Returned when the task has failed.
*   `TASK_NOT_RETURNED` (TaskStatus.NOT_RETURNED): Returned when the task result has not been returned yet.
*   `TASK_IS_QUEUED` (TaskStatus.QUEUED): Indicates the task is queued but not started.
*   `TASK_IS_RUNNING` (TaskStatus.RUNNING): Indicates the task is currently running.
*   `TASK_IS_SUCCEEDED` (TaskStatus.SUCCEED): Indicates the task has completed successfully.
*   `TASK_IS_FAILED` (TaskStatus.NOT_SUCCEEDED): Indicates the task has failed.
*   `TIME_OUT` (TaskStatus.TIME_OUT): Returned when a timeout occurs.
*   `NO_TASK_LEFT` (TaskStatus.ZERO_LEFT): Returned when there are no tasks left.

**Private Attributes (Internal Use):**

*   `_async_loop` (asyncio.AbstractEventLoop): The asyncio event loop (used in 'async' mode only).
*   `_thread_task_queue` (Queue): The task queue for thread-based modes.
*   `_thread_result_queue` (Queue): The result queue for storing task results.
*   `_thread_quit_event` (threading.Event): Event to signal thread termination.
*   `_condition` (threading.Condition): Condition variable for task queue synchronization.
*   `_locker` (threading.Lock): Lock for protecting shared resources.
*   `_task_elapsed` (List[float]): List to record task execution times.

### Execution Modes Detail

**1. 'async' Mode (Default)**

This is the most performant mode for IO-bound asynchronous operations:
- Uses asyncio event loop to run multiple coroutines concurrently in a single thread
- Ideal for network requests, file IO, and other async IO operations
- Can handle thousands of concurrent tasks with minimal overhead
- The `n_worker` parameter is ignored in this mode

**2. 'thread' Mode**

Suitable for wrapping synchronous functions into an async-like interface:
- Uses a Queue to cache tasks and processes them one at a time
- Useful when you have synchronous IO functions but want non-blocking behavior
- The `n_worker` parameter is ignored in this mode

**3. 'threads' Mode**

Enables parallel execution of multiple synchronous tasks:
- Uses multiple worker threads (specified by `n_worker`) to execute tasks in parallel
- Suitable for IO-bound tasks that need concurrent execution
- Each task runs in its own thread, sharing the same process space
- The `n_worker` parameter must be specified

**4. 'process' Mode**

Enables true parallel execution for CPU-intensive tasks:
- Uses multiprocessing.Pool to run tasks in separate processes
- Bypasses Python's GIL limitation for CPU-bound operations
- Suitable for computationally intensive tasks like image processing, data analysis
- The `n_worker` parameter must be specified (defaults to CPU core count)

### Methods

#### __init__(mode: str = 'async', n_worker: Optional[int] = None, sleep_while_empty: float = 0.1, report_error: bool = False, mp_pool_init_kwargs: Optional[Dict[str, Any]] = None)

See Class Initialization section above.

#### start() -> 'TaskPool'

##### Method Summary

Starts the task pool and begins processing tasks.

##### Parameters

None

##### Return Value

Returns `self` for method chaining.

##### Notes

*   This method must be called before adding tasks to the pool.
*   It is recommended to call this method after creating the TaskPool instance.
*   Can be chained with other methods: `pool = TaskPool(mode='async').start()`

##### Example

```python
pool = TaskPool(mode='async').start()
```

#### add_task(name: str, coro_func, *args, **kwargs) -> str

##### Method Summary

Adds a task to the task pool for execution.

##### Parameters

*   `name` (str): The task name. If empty or None, a unique name will be generated using the function name and UUID.
*   `coro_func`: The coroutine function (for 'async' mode) or regular function (for other modes).
*   `*args`: Positional arguments to pass to the task function.
*   `**kwargs`: Keyword arguments to pass to the task function.

##### Return Value

Returns the task name (either the provided name or the generated name).

##### Notes

*   If a task with the same name already exists, it will be replaced with a warning.
*   The task is added to the queue and will be executed asynchronously.

##### Example

```python
pool = TaskPool(mode='async').start()
# Add a task with custom name
pool.add_task('fetch_data', fetch_url, 'http://example.com')
# Add a task with auto-generated name
pool.add_task(None, process_data, arg1, arg2)
```

#### query_task(name: str, block: bool = False, timeout: int = 3) -> Any

##### Method Summary

Queries the status or result of a task.

##### Parameters

*   `name` (str): The task name to query.
*   `block` (bool, default=False): If True, block until the task is completed or timeout occurs.
*   `timeout` (int, default=3): Maximum time to wait in seconds when block is True.

##### Return Value

*   If task not found: Returns `TaskStatus.NOT_FOUND`.
*   If task not finished and block is False: Returns `TaskStatus.NOT_FINISHED`.
*   If task not finished and block is True but timeout occurs: Returns `TaskStatus.TIME_OUT`.
*   If task completed (success or failure): Returns the result or the exception.

##### Notes

*   This method also updates the internal task status dictionary.

##### Example

```python
# Non-blocking query
result = pool.query_task('task1')
if result == TaskPool.TASK_NOT_FINISHED:
    print("Task still running")

# Blocking query with timeout
result = pool.query_task('task1', block=True, timeout=10)
if result == TaskPool.TIME_OUT:
    print("Task timed out")
elif result == TaskPool.TASK_NOT_SUCCEEDED:
    print("Task failed")
else:
    print(f"Result: {result}")
```

#### pull_task(return_name: bool = False, block: bool = True, timeout: int = 3) -> Any

##### Method Summary

Pulls a task result from the result queue. Unlike `query_task`, this method removes the task from the internal tasks dictionary.

##### Parameters

*   `return_name` (bool, default=False): If True, returns a tuple of (name, result) instead of just the result.
*   `block` (bool, default=True): If True, block until a result is available or timeout occurs.
*   `timeout` (int, default=3): Maximum time to wait in seconds.

##### Return Value

*   If queue is empty and block is False: Returns `TaskStatus.NOT_RETURNED`.
*   If timeout occurs: Returns `TaskStatus.NOT_RETURNED`.
*   Otherwise: Returns the task result (and optionally the name).

##### Notes

*   This method removes the task from the internal dictionary after retrieving the result.
*   Failed tasks will have their errors printed and traced.

##### Example

```python
# Pull next available result
result = pool.pull_task(block=True, timeout=5)
if result != TaskPool.TASK_NOT_RETURNED:
    print(f"Got result: {result}")

# Pull with name
name, result = pool.pull_task(return_name=True)
print(f"Task {name} returned: {result}")
```

#### map_tasks(tasks: Union[List[Tuple[List, Dict]], Dict[str, Tuple[List, Dict]]], coro_func: Callable, batch_size: int = None, return_result: bool = True, timeout: int = 3, wait_busy: bool = False, **kwargs) -> Union[List[Any], Dict[str, Any]]

##### Method Summary

Maps a list or dictionary of tasks to a coroutine/function and returns the results.

##### Parameters

*   `tasks` (Union[List[Tuple[List, Dict]], Dict[str, Tuple[List, Dict]]]): 
    *   As a list: A list of tuples where each tuple contains (args_list, kwargs_dict).
    *   As a dict: A dictionary mapping task names to (args_list, kwargs_dict) tuples.
*   `coro_func` (Callable): The coroutine or function to execute for each task.
*   `batch_size` (int, default=None): If specified and >= 1, splits tasks into batches and passes batches (as a list) to the function. Only applicable when tasks is a list.
*   `return_result` (bool, default=True): If True, returns the results of all tasks. If False, returns the task names immediately.
*   `timeout` (int, default=3): Timeout in seconds for each task when return_result is True.
*   `wait_busy` (bool, default=False): If True, waits for the pool to be idle before adding the next batch.
*   `**kwargs`: Additional keyword arguments to pass to every task execution.

##### Return Value

*   If tasks is a list: Returns a list of results (or task names if return_result is False).
*   If tasks is a dict: Returns a dictionary mapping task names to results (or just task names if return_result is False).

##### Notes

*   This is a convenience method for batch task submission.
*   When batch_size is specified, each batch is passed as a list to the function.

##### Example

```python
# List of tasks (args, kwargs)
tasks = [
    (['url1'], {}),
    (['url2'], {}),
    (['url3'], {}),
]
results = pool.map_tasks(tasks, fetch_url, return_result=True, timeout=10)

# Dictionary of tasks
tasks = {
    'task1': (['data1'], {}),
    'task2': (['data2'], {}),
}
results = pool.map_tasks(tasks, process_data)

# Batch processing
tasks = [(i,) for i in range(100)]
results = pool.map_tasks(tasks, process_batch, batch_size=10, wait_busy=True)
```

#### count_waiting_tasks() -> int

##### Method Summary

Returns the number of tasks that are waiting to be processed.

##### Parameters

None

##### Return Value

*   For 'async' mode: Returns the number of unfinished (pending) tasks.
*   For 'thread', 'threads', and 'process' modes: Returns the number of tasks in the queue that haven't started yet.

##### Notes

*   This is useful for monitoring task queue status.

##### Example

```python
waiting = pool.count_waiting_tasks()
print(f"Tasks waiting: {waiting}")
```

#### count_done_tasks() -> int

##### Method Summary

Returns the number of tasks that have been completed (including successful and failed tasks).

##### Parameters

None

##### Return Value

Returns the total count of completed tasks (both successful and failed).

##### Notes

*   This method queries the result queue to update the internal task status before counting.

##### Example

```python
done = pool.count_done_tasks()
print(f"Tasks completed: {done}")
```

#### get_task_elapsed() -> Tuple[float, float, float]

##### Method Summary

Returns statistics about task execution times.

##### Parameters

None

##### Return Value

Returns a tuple of (max_elapsed, min_elapsed, avg_elapsed) in seconds.

##### Notes

*   This method requires at least one task to have been completed.

##### Example

```python
max_time, min_time, avg_time = pool.get_task_elapsed()
print(f"Execution time - Max: {max_time:.2f}s, Min: {min_time:.2f}s, Avg: {avg_time:.2f}s")
```

#### check_empty() -> bool

##### Method Summary

Checks whether all tasks (both pending and completed) are empty.

##### Parameters

None

##### Return Value

Returns True if there are no pending tasks and all results have been retrieved, False otherwise.

##### Example

```python
if pool.check_empty():
    print("All tasks completed")
```

#### wait_till(condition_func, wait_each_loop: float = 0.5, timeout: float = None, verbose: bool = False, *args, **kwargs) -> Union['TaskPool', bool]

##### Method Summary

Waits until a condition function returns True or a timeout occurs.

##### Parameters

*   `condition_func` (Callable): A function that returns True when the condition is met, False otherwise.
*   `wait_each_loop` (float, default=0.5): Sleep time in seconds between each condition check.
*   `timeout` (float, default=None): Maximum time to wait in seconds. If None, waits indefinitely.
*   `verbose` (bool, default=False): If True, displays a progress bar using tqdm showing done/sum tasks.
*   `*args`: Additional positional arguments to pass to the condition function.
*   `**kwargs`: Additional keyword arguments to pass to the condition function.

##### Return Value

*   Returns `self` (TaskPool) if the condition is met.
*   Returns `False` if the timeout is reached.

##### Notes

*   This method calls `_query_task_queue` in each loop to update task statuses.
*   Can be used to wait for specific conditions like all tasks being done.

##### Example

```python
# Wait until all tasks are done
pool.wait_till(lambda: pool.check_empty(), timeout=30)

# Wait until a specific number of tasks are done
pool.wait_till(lambda done, target=5: done >= target, 
               wait_each_loop=0.1, done=pool.count_done_tasks(), target=5)

# With verbose progress bar
pool.wait_till(lambda: pool.check_empty(), verbose=True, timeout=60)
```

#### wait_till_tasks_done(task_names: List[str], wait_each_loop: float = 0.5) -> Dict[str, Union[Any, Literal[TaskStatus.NOT_FOUND], Literal[TaskStatus.NOT_FINISHED]]]

##### Method Summary

Waits until specific tasks are completed and returns their results.

##### Parameters

*   `task_names` (List[str]): A list of task names to wait for.
*   `wait_each_loop` (float, default=0.5): Sleep time in seconds between each check.

##### Return Value

A dictionary with task names as keys and their results or status as values.

##### Notes

*   This method waits for the specified tasks to complete, then queries each one.

##### Example

```python
results = pool.wait_till_tasks_done(['task1', 'task2', 'task3'])
for name, result in results.items():
    if result == TaskPool.TASK_NOT_FOUND:
        print(f"Task {name} not found")
    elif result == TaskPool.TASK_NOT_FINISHED:
        print(f"Task {name} not finished")
    else:
        print(f"Task {name} result: {result}")
```

#### wait_till_free(wait_each_loop: float = 0.01, timeout: float = None, update_result_queue: bool = True)

##### Method Summary

Waits until the task queue is empty (all pending tasks have been picked up for processing).

##### Parameters

*   `wait_each_loop` (float, default=0.01): Sleep time in seconds between each check.
*   `timeout` (float, default=None): Maximum time to wait in seconds.
*   `update_result_queue` (bool, default=True): Whether to update the result queue in each loop.

##### Notes

*   This is a convenience method that calls `wait_till` with a lambda function checking for zero waiting tasks.

##### Example

```python
# Add multiple tasks
for i in range(10):
    pool.add_task(f'task{i}', some_function, i)

# Wait for all tasks to be picked up
pool.wait_till_free(timeout=30)
print("All tasks are being processed")
```

#### clear(clear_tasks: bool = True, clear_queue: bool = True) -> List[int]

##### Method Summary

Clears the task pool, including tasks dictionary and queues.

##### Parameters

*   `clear_tasks` (bool, default=True): Whether to clear the task dictionary.
*   `clear_queue` (bool, default=True): Whether to clear the task queue and result queue.

##### Return Value

Returns a list containing the sizes of [task dictionary, task queue, result queue] before clearing.

##### Notes

*   This method does not check whether the task pool is running.
*   Use with caution as it will discard pending tasks and unretrieved results.

##### Example

```python
# Clear everything
sizes = pool.clear()
print(f"Cleared - Tasks: {sizes[0]}, Pending: {sizes[1]}, Results: {sizes[2]}")

# Only clear the result queue, keep pending tasks
pool.clear(clear_tasks=False, clear_queue=True)
```

#### close(timeout: float = None) -> None

##### Method Summary

Closes the task pool, stops all workers, and joins the threads.

##### Parameters

*   `timeout` (float, default=None): Maximum time to wait for thread joining in seconds.

##### Notes

*   For 'async' mode: Stops the asyncio event loop and closes it.
*   For 'thread', 'threads', and 'process' modes: Sets the quit event and joins the worker threads.
*   After calling close(), the TaskPool cannot be reused.

##### Example

```python
pool = TaskPool(mode='async').start()
# ... add and process tasks ...
pool.close()  # Wait for all tasks to complete and close the pool
print("Pool closed successfully")
```

### TaskStatus Enum

The TaskStatus enum defines various states for task execution:

```python
class TaskStatus(Enum):
    SUCCEED = 0          # Task completed successfully
    NOT_FOUND = 1        # Task was not found in the pool
    NOT_FINISHED = 2     # Task has not finished execution
    NOT_SUCCEEDED = 3    # Task execution failed
    NOT_RETURNED = 4     # Task result has not been returned yet
    TIME_OUT = 5         # Operation timed out
    ZERO_LEFT = 6        # No tasks left in queue
    RUNNING = 7          # Task is currently running
    QUEUED = 8           # Task is queued but not started
```

### Usage Examples

#### Example 1: Basic Async Usage

```python
import asyncio
from mbapy.web_utils.task import TaskPool

async def fetch_data(url, delay):
    await asyncio.sleep(delay)
    return f"Data from {url}"

# Create and start pool
pool = TaskPool(mode='async').start()

# Add tasks
pool.add_task('task1', fetch_data, 'http://example.com/1', 2)
pool.add_task('task2', fetch_data, 'http://example.com/2', 1)

# Query results
result1 = pool.query_task('task1', block=True, timeout=5)
result2 = pool.query_task('task2', block=True, timeout=5)

print(result1, result2)  # Data from http://example.com/1 Data from http://example.com/2

pool.close()
```

#### Example 2: Using map_tasks for Batch Processing

```python
from mbapy.web_utils.task import TaskPool

def download_file(url):
    import time
    time.sleep(1)  # Simulate IO operation
    return f"Downloaded {url}"

pool = TaskPool(mode='threads', n_worker=3).start()

# Define tasks as list of (args, kwargs) tuples
tasks = [
    (['file1.txt'], {}),
    (['file2.txt'], {}),
    (['file3.txt'], {}),
    (['file4.txt'], {}),
    (['file5.txt'], {}),
]

# Process tasks and get results
results = pool.map_tasks(tasks, download_file, return_result=True, timeout=30)
print(results)

pool.close()
```

#### Example 3: CPU-Intensive Tasks with Process Pool

```python
from mbapy.web_utils.task import TaskPool

def cpu_bound_task(n):
    # Simulate CPU-intensive computation
    return sum(i * i for i in range(n))

pool = TaskPool(mode='process', n_worker=4).start()

# Define tasks as dictionary
tasks = {
    'task1': ([1000000], {}),
    'task2': ([2000000], {}),
    'task3': ([1500000], {}),
}

results = pool.map_tasks(tasks, cpu_bound_task)
print(results)

pool.close()
```

#### Example 4: Waiting for Specific Tasks

```python
from mbapy.web_utils.task import TaskPool
import asyncio

async def async_task(name, duration):
    await asyncio.sleep(duration)
    return f"{name} done"

pool = TaskPool(mode='async').start()

# Add multiple tasks
pool.add_task('task1', async_task, 'task1', 3)
pool.add_task('task2', async_task, 'task2', 1)
pool.add_task('task3', async_task, 'task3', 2)

# Wait for specific tasks to complete
results = pool.wait_till_tasks_done(['task1', 'task2'])
print(results)

# Or wait for all tasks to be done
pool.wait_till(lambda: pool.check_empty(), timeout=10)

pool.close()
```

### Best Practices

1. **Always call close()**: Always close the pool when done to properly release resources.

2. **Use appropriate mode**:
   - Use `'async'` for async IO operations (network requests, etc.)
   - Use `'thread'` or `'threads'` for sync IO operations
   - Use `'process'` for CPU-intensive operations

3. **Handle timeouts**: Always set appropriate timeouts when using blocking queries.

4. **Monitor task status**: Use `count_waiting_tasks()` and `count_done_tasks()` to monitor progress.

5. **Error handling**: Set `report_error=True` in 'thread' or 'process' mode to see errors immediately.

6. **Task naming**: Provide meaningful task names for easier debugging and result tracking.