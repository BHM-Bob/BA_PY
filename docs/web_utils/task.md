# Module Overview
This Python module is designed to handle concurrent operations, input processing, and task management through a series of functions and classes. It includes utilities for creating and managing threads, processing tasks in a pool, and tracking progress with regular updates.

# Types
## Key2Action
**General description**

A named tuple representing an action triggered by a specific key.

### Attributes
- `statue` (str): The signal for controlling the `_wait_for_quit` function, such as 'running' and 'exit'.
- `func` (Callable): The action function.
- `args` (list): The arguments for the action function.
- `kwgs` (dict): The keyword arguments for the action function.
- `is_reg` (bool, default=False): Indicates whether to use regular expression to match the key.
- `lock` (_thread.lock, default=None): Thread lock.

# Functions
## `_wait_for_quit(statuesQue, key2action: Dict[str, List[Key2Action]]) -> int`
### Brief Description
Monitors keyboard input to trigger specific actions and manages a dictionary of actions associated with input patterns, including regular expression matching.

### Parameters
- `statuesQue`: A queue instance for status updates.
- `key2action (Dict[str, List[Key2Action]])`: A dictionary linking keys to lists of `Key2Action`, which are predefined actions with associated parameters.

### Return Value
- Returns `0` to indicate the exit status.

### Notes
- The loop continues until the 'exit' signal is received, managing actions with or without lock mechanisms.

#### Example
```python
statuesQue = Queue()
key2action = {
    's': [Key2Action('running', some_function, [arg1, arg2], {'kwarg1': val1})]
}
_wait_for_quit(statuesQue, key2action)
```

## `statues_que_opts(theQue, var_name, opts, *args)`
### Brief Description
Handles operations on a specified variable within a queue, such as getting, setting, or modifying its value.

### Parameters
- `theQue`: The queue on which operations are performed.
- `var_name`: The name of the variable to manipulate.
- `opts`: The operation to perform, such as 'getValue', 'setValue', 'putValue', 'reduceBy', or 'addBy'.
- `*args`: Additional arguments required for the operation.

### Return Value
- Returns the result of the operation, if applicable.

### Notes
- This function is used to interface with the queue for variable manipulation, supporting a range of options.

## `get_input(promot: str = '', end = '\n')`
### Brief Description
Retrieves input from a queue after displaying an optional prompt, with a wait mechanism in place if the input is not immediately available.

### Parameters
- `promt`: The optional prompt string to display.
- `end`: The string termination character.

### Return Value
- Returns the input value retrieved from the queue.

### Notes
- The function ensures that input is available before proceeding, with a sleep interval to check for input availability.

## `launch_sub_thread(statuesQue = statuesQue, key2action: List[Tuple[str, Key2Action]] = [])`
### Brief Description
Initiates a sub-thread for concurrent execution of tasks, with predefined actions triggered by keyboard input or other events.

### Parameters
- `statuesQue`: The queue instance for status management.
- `key2action`: A list of tuples associating keys with `Key2Action`, defining the actions to be executed upon receiving specific inputs.

### Return Value
- None

### Notes
- This function sets up a global queue and starts a new thread for handling predefined actions, with specific considerations for exit signals and input matching.

## `show_prog_info(idx: int, sum: int = -1, freq: int = 10, otherInfo: str = '')`
### Brief Description
Displays progress information in the console at specified intervals, providing updates on the current status of a process.

### Parameters
- `idx`: The current index or step number in the process.
- `sum`: The total number of steps or items to process; if not specified, it defaults to -1.
- `freq`: The frequency of progress updates.
- `otherInfo`: Additional information to display alongside the progress.

### Return Value
- None

### Notes
- This function is used for providing real-time feedback on the progress of operations.

# Classes

## `Timer()`
### `Timer` Initialization
Initializes a `Timer` object to measure and record time intervals.

### Members
- `lastTime`: The timestamp of the last recorded time for time interval measurements.

### Methods
#### `OnlyUsed() -> float`
Returns the time elapsed since the last recorded time.

#### `__call__() -> float`
Updates the `lastTime` to the current time and returns the time elapsed since the previous `lastTime`.

## `ThreadsPool(sum_threads: int, self_func, other_data, name: str = 'ThreadsPool')`
### `ThreadsPool` Initialization
Creates a pool of threads for parallel task execution, with facilities for data queuing and signal management.

### Members
- `sumThreads`: The total number of threads in the pool.
- `sumTasks`: The count of tasks submitted to the pool.
- `name`: The name of the thread pool for identification.
- `timer`: A `Timer` instance for performance tracking.
- `sig`: A queue used for signaling threads.
- `putDataQues`: A list of queues for task data input.
- `getDataQues`: A list of queues for receiving processed data.

### Methods
#### `put_task(data) -> None`
Submits a task to the thread pool for execution.

#### `loop2quit(wait2quitSignal) -> list`
Waits for all tasks to complete and returns the results.

### Notes
- This class manages a pool of threads, facilitating concurrent task processing with input and output queuing.

## `TaskStatus(Enum)`
### `TaskStatus` Members
- `SUCCEED`: Indicates successful task completion.
- `NOT_FOUND`: Indicates that the task could not be found.
- `NOT_FINISHED`: Indicates that the task has not yet completed.
- `NOT_SUCCEEDED`: Indicates that the task did not succeed.
- `NOT_RETURNED`: Indicates that the task's result was not returned.
- `TIME_OUT`: Indicates that a timeout occurred while waiting for the task.
- `ZERO_LEFT`: Indicates that there are no tasks left to process.

## `TaskPool(mode: str = 'async', n_worker: int = None, sleep_while_empty: float = 0.1)`
### `TaskPool` Initialization
Establishes a task pool that can operate in different modes, such as asynchronous or threaded, for managing and executing tasks.

### Attributes
- `mode`: The operational mode of the task pool, which can be 'async', 'thread', 'threads', or 'process'.
- `loop`: The asyncio event loop used for asynchronous operations.
- `thread`: The threading object used for threaded operations.
- `tasks`: A dictionary mapping task names to their respective statuses or results.
- `TASK_NOT_FOUND`, `TASK_NOT_FINISHED`, `TASK_NOT_SUCCEEDED`: Constants indicating the status of tasks.

### Methods
#### `add_task(name: str, coro_func, *args, **kwargs)`
Adds a new task to the task pool with a unique name.

#### `query_task(name: str, block: bool = False, timeout: int = 3)`
Retrieves the result or status of a task, with options for blocking and timeout.

#### `query_single_task_from_tasks(tasks_name: List[str], block: bool = False, timeout: int = 3)`
Retrieves a result or statu of a task from a list, with options for blocking and timeout.

#### `count_waiting_tasks()`
Returns the count of tasks that have not yet begun execution.

#### `count_done_tasks()`
Returns the count of tasks that have completed, including both successful and failed tasks.

#### `run()`
Starts the task pool's execution流程, launching any required threads or event loops.

#### `check_empty()`
Checks if all tasks have been completed.

#### `wait_till(condition_func, wait_each_loop: float = 0.5, timeout: float = None, verbose: bool = False, *args, **kwargs)`
Waits until a specified condition is met or a timeout occurs, with optional verbose output.

#### `wait_till_tasks_done(task_names: List[str], wait_each_loop: float = 0.5)`
Waits until all specified tasks have completed.

#### `close()`
Shuts down the task pool, stopping event loops and joining threads.

### Notes
- The `TaskPool` class offers a versatile approach to task management, supporting asynchronous coroutines, threading, and multiprocessing for concurrent task execution.

### Example
```python
pool = ThreadsPool(5, some_function, other_data, 'MyPool')
pool.put_task(task_data)
results = pool.loop2quit('quit')
```