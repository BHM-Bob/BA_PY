# mbapy.web_utils.task

This module provides functions for launching a sub-thread to run a separate task concurrently with the main thread.  

## Functions

### launch_sub_thread()

Launches a sub-thread to run a separate task concurrently with the main thread.  

This function creates a global `statuesQue` queue and puts a dictionary with the keys `quit` and `input` into the queue. The `quit` key is set to `False` and the `input` key is set to `None`. 
The function then starts a new thread by calling the `_wait_for_quit` function with the `statuesQue` queue as an argument. 
Finally, the function prints the message "web sub thread started".  

Example:  
```python
launch_sub_thread()
```

### _wait_for_quit(statuesQue)

Waits for user input to quit the sub-thread.  

Parameters:  
- statuesQue (Queue): The queue used for communication between the main thread and the sub-thread.  

Returns:  
- int: Always returns 0.  

### statues_que_opts(theQue, var_name, opts, *args)

Performs various operations on a dictionary stored in a queue.  

Parameters:  
- theQue (Queue): The queue containing the dictionary.  
- var_name (str): The key of the value to operate on.  
- opts (str): The operation to perform. Can be one of the following:  
  - "getValue": Get the value of the specified key.  
  - "setValue": Set the value of the specified key to the first argument.  
  - "putValue": Put the first argument as the value of the specified key in the dictionary.  
  - "reduceBy": Subtract the first argument from the value of the specified key.  
  - "addBy": Add the first argument to the value of the specified key.  
- *args: Additional arguments required for certain operations.  

Returns:  
- Any: The value returned by the operation.  

### get_input(promot='', end='\n')

Gets user input from the sub-thread.  

Parameters:  
- promot (str, optional): The prompt message to display. Defaults to an empty string.  
- end (str, optional): The end character to use when displaying the prompt message. Defaults to '\n'.  

Returns:  
- Any: The user input.  

### show_prog_info(idx, sum=-1, freq=10, otherInfo='')

Prints the progress information at regular intervals.  

Parameters:  
- idx (int): The current index.  
- sum (int, optional): The total number of items. Defaults to -1.  
- freq (int, optional): The frequency at which progress information is printed. Defaults to 10.  
- otherInfo (str, optional): Additional information to display. Defaults to an empty string.  

Returns:  
- None

## Classes

### Timer

A class that represents a timer.  

Methods:  
- __init__(self): Initializes the Timer object.  
- OnlyUsed(self): Returns the time elapsed since the last call to the timer.  
- __call__(self) -> float: Returns the time elapsed since the last call to the timer and updates the timer.  

### ThreadsPool

A class that represents a pool of threads.  

Methods:  
- __init__(self, sum_threads, self_func, other_data, name='ThreadsPool'): Initializes the ThreadsPool object with the specified number of threads, a function to be executed by each thread, additional data for the function, and an optional name for the pool.  
- put_task(self, data): Puts a task into the pool.  
- loop2quit(self, wait2quitSignal): Sends a "wait to quit" signal to each thread in the pool and starts looping to wait for the threads to finish.  
- 

## Constants

### statuesQue

A global variable that represents a queue used for communication between the main thread and the sub-thread.  

Example:  
```python
statuesQue = Queue()
```