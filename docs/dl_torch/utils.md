# mbapy.dl_torch.utils

This module provides functions and classes related to model initialization, training, and checkpoint management.  

## Functions

### init_model_parameter(model: nn.Module) -> nn.Module

Initialize the parameters of the model.  

Parameters:  
- model (nn.Module): The model to initialize.  

Returns:  
- nn.Module: The initialized model.  

### adjust_learning_rate(optimizer: torch.optim.Optimizer, now_epoch: int, args: GlobalSettings) -> None

Adjusts the learning rate of the optimizer based on the current epoch and arguments.  

Parameters:  
- optimizer (torch.optim.Optimizer): The optimizer to adjust the learning rate of.  
- now_epoch (int): The current epoch number.  
- args (GlobalSettings): The parsed command-line arguments.  

Returns:  
- None

### format_secs(sumSecs: int) -> Tuple[int, int, int]

Formats a given number of seconds into hours, minutes, and seconds.  

Parameters:  
- sumSecs (int): The total number of seconds.  

Returns:  
- Tuple[int, int, int]: A tuple containing three integers representing the number of hours, minutes, and seconds respectively.  

### save_checkpoint(epoch: int, args: GlobalSettings, model: nn.Module, optimizer: torch.optim.Optimizer, loss, other: dict, tailName: str) -> None

Saves a checkpoint of the model and optimizer state, along with other information such as epoch, loss, and arguments.  

Parameters:  
- epoch (int): The current epoch number.  
- args (GlobalSettings): The GlobalSettings object containing various hyperparameters and settings.  
- model (nn.Module): The model whose state needs to be saved.  
- optimizer (torch.optim.Optimizer): The optimizer whose state needs to be saved.  
- loss: The current loss value.  
- other (dict): A dictionary containing any other information that needs to be saved.  
- tailName (str): A string to be used in the checkpoint file name for better identification.  

Returns:  
- None

### resume(args: GlobalSettings, model: nn.Module, optimizer: torch.optim.Optimizer, other: dict = {}) -> Tuple[nn.Module, torch.optim.Optimizer, Union[int, float]]

Resumes the training from the last checkpoint if it exists, otherwise starts from scratch.  

Parameters:  
- args (GlobalSettings): The GlobalSettings object containing various hyperparameters and settings.  
- model (nn.Module): The model to be trained.  
- optimizer (torch.optim.Optimizer): The optimizer to be used for training.  
- other (dict, optional): A dictionary containing additional objects to be updated from the checkpoint. Defaults to {}.  

Returns:  
- Tuple[nn.Module, torch.optim.Optimizer, Union[int, float]]: A tuple of the model, optimizer, and old_losses if the checkpoint exists, otherwise a tuple of the model, optimizer, and 0.  

## Classes

### AverageMeter

A class that computes and stores the average and current value.  

Methods:  
- __init__(self, name: str, fmt: str = ":f"): Initializes the AverageMeter object with the given name and format.  
- reset(self): Resets the values of the AverageMeter object.  
- update(self, val, n=1): Updates the values of the AverageMeter object with the given value and count.  
- __str__(self): Returns a string representation of the AverageMeter object.  

### ProgressMeter

A class that displays the progress of training.  

Methods:  
- __init__(self, num_batches: int, meters, prefix: str = "", mp = None): Initializes the ProgressMeter object with the given number of batches, meters, prefix, and Mprint object.  
- display(self, batch): Displays the progress of training.  
- _get_batch_fmtstr(self, num_batches: int): Returns the format string for displaying the batch progress.  

### TimeLast

A class that calculates the estimated time remaining for a task.  

Methods:  
- __init__(self): Initializes the TimeLast object.  
- update(self, left_tasks: int, just_done_tasks: int = 1): Updates the TimeLast object with the number of tasks remaining and the number of tasks just completed.  

### Mprint

A class that provides logging tools.  

Methods:  
- __init__(self, path: str = "log.txt", mode: str = "lazy", cleanFirst: bool = True): Initializes the Mprint object with the given log file path, mode, and cleanFirst flag.  
- mprint(self, *args): Prints the log message and writes it to the log file.  
- logOnly(self, *args): Writes the log message to the log file without printing.  
- exit(self): Writes the remaining log messages to the log file and closes it.  
- __str__(self): Returns a string representation of the Mprint object.  

### GlobalSettings

A class that represents a collection of global settings and hyperparameters.  

Methods:  
- __init__(self, mp: Mprint, model_root: str): Initializes the GlobalSettings object with the given Mprint object and model root directory.  
- toDict(self, printOut: bool = False, mp: Mprint = None): Converts the GlobalSettings object to a dictionary.  
- set_resume(self): Updates the resume paths based on the model root directory.  

Attributes:  
- read (dict): A dictionary for data reading.  
- batch_size (int): The batch size for training.  
- load_shape (List[int]): The shape of the input data.  
- model (Any): The model object.  
- arch (Any): The architecture object.  
- lr (float): The learning rate.  
- in_shape (List[int]): The shape of the input data.  
- out_shape (List[int]): The shape of the output data.  
- load_db_ratio (float): The ratio of the loaded data.  
- epochs (int): The number of training epochs.  
- print_freq (int): The frequency of printing training information.  
- test_freq (int): The frequency of testing the model.  
- momentum (float): The momentum for the optimizer.  
- weight_decay (float): The weight decay for the optimizer.  
- seed (int): The random seed.  
- start_epoch (int): The starting epoch number.  
- moco_m (float): The moco_m parameter.  
- moco_k (int): The moco_k parameter.  
- byolq_k (int): The byolq_k parameter.  
- moco_t (float): The moco_t parameter.  
- cos (bool): Whether to use cosine learning rate schedule.  
- paths (dict): A dictionary of paths.  
- data (str): The data directory.  
- model_root (str): The model root directory.  
- resume_paths (List[str]): The paths of the checkpoint files.  
- resume (str): The path of the resume checkpoint file.  
- mp (Mprint): The Mprint object for logging.  

Example:  
```python
mp = Mprint()
settings = GlobalSettings(mp, 'model/')
settings.toDict(printOut=True, mp=mp)
settings.set_resume()
```

## Constants

### _Params

A dictionary that contains various parameters for the module.  

### USE_VIZDOM

A global variable that controls whether to use the Vizdom library for visualization. Set to False to disable Vizdom.  

### viz

A Vizdom object used for visualization.  

### vizRecord

A list that records the visualization logs.  
