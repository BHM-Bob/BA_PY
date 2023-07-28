# Module Name: mbapy.web

This module provides functions and classes for running tasks concurrently with the main thread using sub-threads. It includes a sub-thread launcher function and a thread pool class.

## Function: launch_sub_thread()

Launches a sub-thread to run a separate task concurrently with the main thread.

### Parameters:
- None

### Returns:
- None

### Example:
```python
import mbapy.web

mbapy.web.launch_sub_thread()
```

### Additional Notes:
This function creates a global `statuesQue` queue and puts a dictionary with the keys `quit` and `input` into the queue. The `quit` key is set to `False` and the `input` key is set to `None`. The function then starts a new thread by calling the `_wait_for_quit` function with the `statuesQue` queue as an argument. Finally, the function prints the message "web sub thread started".

## Function: get_input(promot:str = '', end = '\n')

Gets user input from the sub-thread.

### Parameters:
- `promot` (str, optional): The prompt to display to the user. Defaults to an empty string.
- `end` (str, optional): The character to append at the end of the prompt. Defaults to a newline character.

### Returns:
- The user input as a string.

### Example:
```python
import mbapy.web

input_value = mbapy.web.get_input("Enter a value: ")
print("You entered:", input_value)
```

### Additional Notes:
This function retrieves the user input from the `statuesQue` queue. It waits until a non-empty input is available in the queue and then returns the input value. The function also sets the `input` key in the `statuesQue` dictionary to `None` after retrieving the input.

## Function: show_prog_info(idx:int, sum:int = -1, freq:int = 10, otherInfo:str = '')

Prints progress information at regular intervals.

### Parameters:
- `idx` (int): The current index.
- `sum` (int, optional): The total number of items. Defaults to -1.
- `freq` (int, optional): The frequency at which progress information is printed. Defaults to 10.
- `otherInfo` (str, optional): Additional information to display. Defaults to an empty string.

### Returns:
- None

### Example:
```python
import mbapy.web

for i in range(100):
    mbapy.web.show_prog_info(i, sum=100, freq=10, otherInfo="Processing...")
    time.sleep(0.1)
```

### Additional Notes:
This function prints the progress information in the format "{current index} / {total number of items} | {additional information}". The progress information is only printed when the current index is a multiple of the frequency parameter. The function uses a carriage return character to overwrite the previous progress information on the console.

## Class: Timer

A class for measuring time intervals.

### Methods:
- `__init__(self)`: Initializes the Timer object.
- `OnlyUsed(self)`: Returns the time elapsed since the last call to the Timer object.
- `__call__(self) -> float`: Returns the time elapsed since the last call to the Timer object and updates the lastTime attribute.

### Example:
```python
import mbapy.web

timer = mbapy.web.Timer()
time.sleep(1)
print(timer())  # Output: 1.0
```

### Additional Notes:
The Timer class is used to measure time intervals. The `OnlyUsed` method returns the time elapsed since the last call to the Timer object, while the `__call__` method returns the time elapsed since the last call and updates the lastTime attribute.

## Class: ThreadsPool

A class for managing a pool of sub-threads.

### Methods:
- `__init__(self, sum_threads:int, self_func, other_data, name = 'ThreadsPool')`: Initializes the ThreadsPool object.
- `put_task(self, data) -> None`: Puts a task into the sub-thread queue.
- `loop2quit(self, wait2quitSignal) -> list`: Sends a "wait to quit" signal to the sub-threads and waits for them to finish.

### Example:
```python
import mbapy.web

def task_func(put_data_que, get_data_que, sig, other_data):
    # Task implementation goes here
    pass

pool = mbapy.web.ThreadsPool(4, task_func, other_data)
pool.put_task(data)
results = pool.loop2quit(wait2quitSignal)
```

### Additional Notes:
The ThreadsPool class is used to manage a pool of sub-threads. The `__init__` method initializes the ThreadsPool object with the specified number of sub-threads, a task function, and other data. The `put_task` method puts a task into the sub-thread queue, and the `loop2quit` method sends a "wait to quit" signal to the sub-threads and waits for them to finish. The `loop2quit` method returns a list of results from the sub-threads.