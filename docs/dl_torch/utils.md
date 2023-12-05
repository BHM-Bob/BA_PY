# mbapy.dl_torch.utils

This module provides functions and classes related to model initialization, training, and checkpoint management.  

### launch_visdom -> (visdom.Visdom, list)
**General description**

This function initializes the visdom server and returns the visdom object and an empty list for recording.

#### Returns
- vis (visdom.Visdom): The initialized visdom object.
- viz_record (list): An empty list for recording.

#### Notes
This function should be called before using any visualization functions.

#### Example
```python
vis, viz_record = launch_visdom()
```

### Mprint
**General description**

This class provides logging tools for printing and saving logs to a file.

#### Attrs
- path (str): The path to the log file.
- mode (str): The mode for logging, either "lazy" or "normal".
- top_string (str): The top string for the log.
- string (str): The log string.

#### Methods
- mprint(*args) -> None: Prints the log and saves it to the log file based on the mode.
- log_only(*args) -> None: Logs the message to the log file without printing.
- exit(mode='a+') -> None: Writes the log string to the log file and closes it.
- \_\_str\_\_() -> str: Returns the string representation of the Mprint object.

#### Notes
- The `mprint` method prints the log and saves it to the log file based on the mode.
- The `log_only` method logs the message to the log file without printing.
- The `exit` method writes the log string to the log file and closes it.

#### Example
```python
logger = Mprint("log.txt", "normal")
logger.mprint("Logging message")
logger.exit()
```

### GlobalSettings
**General description**

This class contains global settings and hyperparameters for the model.

#### Attrs
- read (dict): Dictionary for data reading.
- batch_size (int): Batch size for data loading.
- load_shape (list): Shape for data loading.
- model (None): The model object.
- arch (None): The architecture of the model.
- lr (float): Learning rate.
- in_shape (list): Input shape for the model.
- out_shape (list): Output shape for the model.
- load_db_ratio (float): Load database ratio.
- epochs (int): Total number of epochs.
- now_epoch (int): Current epoch number.
- left_epochs (int): Number of epochs left.
- print_freq (int): Frequency of printing logs.
- test_freq (int): Frequency of testing the model.
- momentum (float): Momentum for optimization.
- weight_decay (float): Weight decay for optimization.
- seed (int): Random seed for reproducibility.
- moco_m (float): Moco m parameter.
- moco_k (int): Moco k parameter.
- byolq_k (int): Byolq k parameter.
- moco_t (float): Moco t parameter.
- cos (bool): Cosine flag.
- paths (dict): Dictionary for paths.
- data (str): Data information.
- model_root (str): Root directory for the model.
- resume_paths (list): List of paths for resuming the model.
- resume (str): Path for resuming the model.
- mp (Mprint): Mprint object for logging.

#### Methods
- add_epoch(addon: int) -> bool: Adds an epoch and returns True if the current epoch is greater than the total epochs.
- to_dict(printOut=False, mp=None) -> dict: Converts the object attributes to a dictionary and prints it if printOut is True.
- set_resume() -> None: Sets the resume path for the model.

#### Notes
- The `add_epoch` method adds an epoch and returns True if the current epoch is greater than the total epochs.
- The `to_dict` method converts the object attributes to a dictionary and prints it if printOut is True.
- The `set_resume` method sets the resume path for the model.

#### Example
```python
mp = Mprint()
settings = GlobalSettings(mp, "model_root", seed=777)
settings.add_epoch()
settings.to_dict(printOut=True, mp=mp)
settings.set_resume()
```

### init_model_parameter
**General description**

This function initializes the model parameters.

#### Params
- model (torch.nn.Module): The model to be initialized.

#### Returns
- model (torch.nn.Module): The initialized model.

#### Notes
This function initializes the model parameters using specific initialization methods for different types of layers.

#### Example
```python
model = init_model_parameter(model)
```

### adjust_learning_rate
**General description**

This function adjusts the learning rate of the given optimizer based on the current epoch and arguments.

#### Params
- optimizer (torch.optim.Optimizer): Optimizer to adjust the learning rate of.
- now_epoch (int): Current epoch number.
- args (argparse.Namespace): Parsed command-line arguments.

#### Returns
- None

#### Notes
This function adjusts the learning rate based on the cosine or stepwise learning rate schedule.

#### Example
```python
adjust_learning_rate(optimizer, now_epoch, args)
```

### format_secs
**General description**

This function formats a given number of seconds into hours, minutes, and seconds.

#### Params
- sumSecs (int): Total number of seconds.

#### Returns
- tuple: A tuple containing three integers representing the number of hours, minutes, and seconds respectively.

#### Notes
This function formats the total number of seconds into hours, minutes, and seconds.

#### Example
```python
hours, minutes, seconds = format_secs(3600)
```

### AverageMeter
**General description**

This class computes and stores the average and current value.

#### Attrs
- name (str): Name of the meter.
- fmt (str): Format for displaying the values.

#### Methods
- reset() -> None: Resets the values of the meter.
- update(val, n=1) -> None: Updates the meter with a new value and count.
- \_\_str\_\_() -> str: Returns the string representation of the AverageMeter object.

#### Notes
This class computes and stores the average and current value for a specific metric.

### ProgressMeter
**General description**

This class displays the progress of the training process.

#### Attrs
- num_batches (int): Total number of batches.
- meters (list): List of meters for displaying progress.
- prefix (str): Prefix for the progress display.

#### Methods
- display(batch) -> None: Displays the progress for the current batch.

#### Notes
This class displays the progress of the training process with the specified meters.

### TimeLast
**General description**

This class calculates the time remaining for a given number of tasks.

#### Methods
- update(left_tasks:int, just_done_tasks:int = 1) -> float: Updates the time remaining based on the number of tasks completed.

#### Notes
This class calculates the time remaining for a given number of tasks based on the time taken for the last task.

### save_checkpoint
**General description**

This function saves a checkpoint of the model and optimizer state, along with other information.

#### Params
- epoch (float): Current epoch number.
- args (GlobalSettings): Global settings and hyperparameters.
- model (torch.nn.Module): The model to be saved.
- optimizer (torch.optim.Optimizer): The optimizer to be saved.
- loss (float): Current loss value.
- other (dict): Additional information to be saved.
- tailName (str): A string for better identification in the checkpoint file name.

#### Notes
This function saves the model and optimizer state along with other information to a checkpoint file.

### resume_checkpoint
**General description**

This function resumes the training from the last checkpoint if it exists, otherwise starts from scratch.

#### Params
- args (GlobalSettings): Global settings and hyperparameters.
- model (torch.nn.Module): Model to be trained.
- optimizer (torch.optim.Optimizer): Optimizer to be used for training.

#### Returns
- tuple: Tuple of the model, optimizer, and old_losses if checkpoint exists, otherwise tuple of model, optimizer, and 0.

#### Notes
This function resumes the training from the last checkpoint if it exists, otherwise starts from scratch.

### viz_line
**General description**

This function visualizes a line plot using the visdom server.

#### Params
- Y (float): Y-axis value.
- X (float): X-axis value.
- win (str): Window name for the plot.
- title (str): Title for the plot.
- name (str): Name for the line.
- update (str): Update method for the plot.
- opts (dict): Additional options for the plot.

#### Notes
This function visualizes a line plot using the visdom server and records the plot details.

### re_viz_from_json_record
**General description**

This function re-visualizes the recorded plots from a JSON record file.

#### Params
- path (str): Path to the JSON record file.

#### Notes
This function re-visualizes the recorded plots from a JSON record file.

Hope this helps! Let me know if you need anything else.
