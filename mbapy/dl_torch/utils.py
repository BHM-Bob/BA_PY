import atexit
import glob
import math
import os
import time
from queue import Queue
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..base import (MyArgs, format_secs, get_default_for_None, get_fmt_time,
                    put_err)
from ..file import read_json, save_json

viz = None
viz_record = []

def launch_visdom(env: str = 'main'):
    global viz, viz_record
    import visdom
    viz = visdom.Visdom(env = env)
    viz_record = []
    return viz, viz_record

class Mprint:
    """logging tools"""
    def __init__(self, path="log.txt", mode="lazy", clean_first=True):
        self.path = path
        self.mode = mode
        self.top_string = " "
        self.string = ""

        if clean_first:
            with open(path, "w") as f:
                f.write("Mprint : cleanFirst\n")

    def log_only(self, *args):
        """
        Logs the provided arguments to a file or stores them in memory, depending on the mode.
        
        Parameters:
            *args (tuple): The arguments to be logged.
            
        Returns:
            string: The log string that was written to the file or stored in memory.
        """
        string = f'[{get_fmt_time("%Y%m%d-%H%M%S.%f")} - {self.top_string}] '
        for item in args:
            if type(item) != "":
                item = str(item)
            string += item + " "

        if self.mode != "lazy":
            with open(self.path, "a+") as f:
                f.write(string + "\n")
        else:
            self.string += string + "\n"
            
        return string

    def mprint(self, *args):
        """
        Prints the output of the `log_only` method.

        Args:
            *args: Variable length argument list.

        Returns:
            None
        """
        print(self.log_only(*args))   
            
    def __call__(self, *args, log_only = False):
        if log_only:
            return self.log_only(*args)
        else:
            return self.mprint(*args)   

    def exit(self, mode ='a+'):
        """
        Writes the `string` attribute of the current instance to a file specified by the `path` attribute.

        Parameters:
            mode (str): The mode in which the file should be opened. Default is 'a+'.

        Returns:
            None
        """
        with open(self.path, mode) as f:
            f.write(self.string)
            
    def __str__(self):
        return f'path={self.path:s}, mode={self.mode:s}, top_string={self.top_string:s}'
            
class GlobalSettings(MyArgs):
    def __init__(self, mp:Mprint, model_root:str, seed: int = 777, benchmark = True):
        # data loading
        self.read = {}# for data reading
        self.batch_size =  64
        self.load_shape = [3, 128, 128]
        # model
        self.model = None
        self.arch = None
        self.lr =  0.01
        self.in_shape =  [64, 3, 128, 128]
        self.out_shape =  [64, 128]
        self.load_db_ratio =  1
        # default var
        self.epochs = 1500
        self.now_epoch = 0
        self.left_epochs = self.epochs - self.now_epoch
        self.print_freq = 40
        self.test_freq = 5
        self.momentum = 0.9
        self.weight_decay = 1e-4        
        self.seed = seed
        self.moco_m = 0.999
        self.moco_k = 3200
        self.byolq_k = 3200
        self.moco_t = 0.07
        self.cos = True        
        # fixed var
        self.paths = {}
        self.data = ''
        self.model_root = model_root
        self.resume_paths = glob.glob(os.path.join(self.model_root,'*.tar'))
        self.resume = self.resume_paths[0] if len(self.resume_paths) > 0 else 'None'
        # other
        self.mp = mp#Mp        
        if self.seed is not None:
            import random
            torch.manual_seed(self.seed)  # 设置PyTorch的随机种子，用于生成随机数，确保结果的可重复性。
            torch.cuda.manual_seed(self.seed)  # 设置PyTorch的CUDA随机种子，用于在GPU上生成随机数，确保结果的可重复性。
            torch.cuda.manual_seed_all(self.seed)  # 如果使用多个GPU，设置所有GPU的随机种子，确保结果的可重复性。
            np.random.seed(self.seed)  # 设置NumPy的随机种子，用于生成NumPy模块中的随机数，确保结果的可重复性。
            random.seed(self.seed)  # 设置Python标准库中random模块的随机种子，用于生成Python中的随机数，确保结果的可重复性。
            torch.manual_seed(self.seed)  # 再次设置PyTorch的随机种子，确保在后续代码中生成的随机数仍然是基于相同的种子。
            torch.backends.cudnn.benchmark = False  # 禁用cuDNN的自动寻找最适合当前配置的高效算法，以确保结果的可重复性。
            torch.backends.cudnn.deterministic = True  # 设置cuDNN的随机数生成策略为确定性模式，以确保结果的可重复性。
    def add_epoch(self, addon: int = 1) -> bool:
        """
        return True if self.now_epoch > self.epochs
        """
        self.now_epoch += 1
        self.left_epochs -= 1
        return self.now_epoch > self.epochs
    def to_dict(self, printOut = False, mp = None):
        dic = {}
        for attr in vars(self):
            dic[attr] = getattr(self,attr)
        if printOut and mp is not None:
            [ mp.mprint(attr,' : ',dic[attr]) for attr in dic.keys()]
        elif printOut:
            [ print(attr,' : ',dic[attr]) for attr in dic.keys()]
        return dic
    def set_resume(self):
        self.resume_paths = glob.glob(os.path.join(self.model_root,'*.tar'))
        self.resume = self.resume_paths[0] if len(self.resume_paths) > 0 else 'None'

def init_model_parameter(model):
    """
    Initialize the parameters of a given model.

    Args:
        model (nn.Module): The model to initialize.

    Returns:
        nn.Module: The initialized model.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
            if m.weight is not None:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):  # 线性层初始化
            if m.weight is not None:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        elif isinstance(m, nn.RNN):  # 循环神经网络层初始化
            if m.weight_ih_l0 is not None:
                nn.init.xavier_normal_(m.weight_ih_l0)
            if m.weight_hh_l0 is not None:
                nn.init.xavier_normal_(m.weight_hh_l0)
    return model

def adjust_learning_rate(optimizer, now_epoch, args):
    """
    Adjusts the learning rate of the given optimizer based on the current epoch and arguments.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer to adjust the learning rate of.
        now_epoch (int): Current epoch number.
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        None
    """
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * now_epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if now_epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class AverageMeter(object):
    """
    Computes and stores the average and current value
    from FAIR or MAIR 's MoCo.
    The AverageMeter class is used to compute and store the average and 
    current value. Here's what each class method does:

    - __init__(self, name, fmt=":f"): Initializes the AverageMeter object 
    with a given name and format for printing.
    - reset(self): Resets the values of val, avg, sum, and count to zero.
    - update(self, val, n=1): Updates the val, sum, count, and avg values 
    based on the given val and n.
    - __str__(self): Returns a string representation of the AverageMeter 
    object with the name, current value (val), and average value (avg).
    """
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        """
        Resets all the variables in the class to their initial values.
        """
        self.val = 0
        self.avg = 0.0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates the value of the object with the given value `val`. By default, it increments the count of the object by `n` (default value is 1) and updates the sum and average accordingly.

        Parameters:
            val (Any): The value to be updated in the object.
            n (int): The number of times to increment the count (default is 1).

        Returns:
            None
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)
    
class ProgressMeter(object):
    """from FAIR or MAIR 's MoCo.
    This class is a progress meter that can display the progress of a batch process.
     Here's a breakdown of each class method:

    - __init__(self, num_batches, meters, prefix="", mp = None): Initializes the 
    progress meter with the number of batches, a list of meters, an optional prefix,
    and an optional mp parameter.
    - display(self, batch): Displays the progress of the current batch. It prints
    the prefix and the formatted batch progress, followed by the values of each meter.
    - _get_batch_fmtstr(self, num_batches): Generates a format string for displaying
    the batch progress. It calculates the number of digits in the total number of
    batches and returns a formatted string with the current batch and the total
    number of batches.
    """
    def __init__(self, num_batches, meters, prefix="", mp = None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.mp = mp

    def display(self, batch):
        """
        Display the given batch information and meters.

        Args:
            batch (int): The batch information to be displayed.

        Returns:
            None
        """
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if self.mp is None:
            print("\t".join(entries))
        else:
            self.mp.mprint("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
    
class TimeLast(object):
    """
    The TimeLast class is used to track the time taken for certain tasks.

    - The __init__ method initializes the last_time attribute with the current time.
    - The update method calculates the time taken for a certain number of tasks by 
    subtracting the last_time from the current time and then updating last_time with 
    the current time. It returns the total time taken for the remaining tasks based 
    on the time taken for the just completed tasks.
    """
    def __init__(self):
        """
        Initializes the object.

        This function is the constructor of the class. It initializes the object by setting the `last_time` attribute to the current time using the `time.time()` function.

        Parameters:
            self (object): The instance of the class.

        Returns:
            None
        """
        self.last_time = time.time()

    def update(self, left_tasks:int, just_done_tasks:int = 1):
        used_time = time.time() - self.last_time
        self.last_time = time.time()
        sum_last_time = left_tasks * used_time / just_done_tasks
        return sum_last_time            
            
def save_checkpoint(epoch: float, args:GlobalSettings, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer, loss: float, other:dict, tailName:str):
    """
    Saves a checkpoint of the model and optimizer state, along with other information 
    using `to_save_dict.update(other)`. The checkpoint file is named as 
    "checkpoint_tailName_%Y%m%d-%H%M%S.%f.pth.tar" 
    and saved in the directory specified by args.model_root.

    Params:
        - epoch: An integer representing the current epoch number.
        - args: A GlobalSettings object containing various hyperparameters and settings.
        - model: The model whose state needs to be saved.
        - optimizer: The optimizer whose state needs to be saved.
        - loss: The current loss value.
        - other: A dictionary containing any other information that needs to be saved.
        - tailName: A string to be used in the checkpoint file name for better identification.
    """
    state = {
        "epoch": epoch + 1,
        "arch": args.arch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss": loss,
        "args":args.to_dict(),
    }
    state.update(other)
    filename = os.path.join(args.model_root,
                            f"checkpoint_{tailName:s}_{get_fmt_time('%Y%m%d-%H%M%S.%f')}.pth.tar")
    torch.save(state, filename)

def resume_checkpoint(args: GlobalSettings, model: torch.nn.Module,
                      optimizer: torch.optim.Optimizer, strict=False) -> Tuple[torch.nn.Module, torch.optim.Optimizer, float, Dict]:
    """
    Resumes the training from the last checkpoint if it exists, otherwise starts from scratch.

    Params:
        - args: Namespace object containing parsed command-line arguments.
        - model: Model to be trained.
        - optimizer: Optimizer to be used for training.
        - other: Optional dictionary containing additional objects to be updated from checkpoint.

    Returns:
        - Tuple of the model, optimizer, and old_losses if checkpoint exists, otherwise tuple of model, optimizer, and 0.
    """
    if args.resume and os.path.isfile(args.resume):
        args.mp.mprint("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.now_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"], strict = strict)
        optimizer.load_state_dict(checkpoint["optimizer"])
        old_losses = checkpoint["loss"]
        args.mp.mprint(f'loaded checkpoint {args.resume} (epoch {checkpoint["epoch"]})')
        return model, optimizer, old_losses, checkpoint
    else:
        args.mp.mprint("=> no checkpoint found at '{}'".format(args.resume))
        args.mp.log_only(str(model))
        return model, optimizer, 0, {}
    
def viz_line(Y:float, X:float, win:str, title:str = None, name:str = None,
            update:str = 'append', opts:dict = {}):
    """
    Generates a line visualization using the specified data.

    Parameters:
        Y (float): The Y-axis values of the line plot.
        X (float): The X-axis values of the line plot.
        win (str): The window ID of the visualization.
        title (str, optional): The title of the visualization. Defaults to None.
        name (str, optional): The name of the line plot. Defaults to None.
        update (str, optional): The update mode for the visualization. Defaults to 'append'.
        opts (dict, optional): Additional options for the visualization. Defaults to {}.

    Returns:
        str: The updated window ID of the visualization.
    """
    global viz_record
    if not viz_record:
        update = None
    if opts and 'title' in opts:
        win = viz.line(Y = [Y],X = [X],
                       win=win, update=update, opts =  opts)
        viz_record.append([Y, X, win, opts['title'], name, update, opts])
    else:
        title = get_default_for_None(title, win)
        if 'legend' not in opts and 'label' not in opts:
            name = get_default_for_None(name, win)
        opts.update({'title' : title})
        win = viz.line(Y = [Y], X = [X], name = name,
                       win=win, update=update, opts =  opts)
        viz_record.append([Y, X, win, title, name, update, opts])
    return win

def re_viz_from_json_record(path):
    if os.path.isfile(path):
        for log in read_json(path):
            viz.line(Y = [log[0]], X = [log[1]], win = log[2], name = log[4],
                    update = log[5],opts =  log[6])
            

__all__ = [
    'viz',
    'viz_record',
    'launch_visdom',
    'Mprint',
    'GlobalSettings',
    'init_model_parameter',
    'adjust_learning_rate',
    'format_secs',
    'AverageMeter',
    'ProgressMeter',
    'TimeLast',
    'save_checkpoint',
    'resume_checkpoint',
    'viz_line',
    're_viz_from_json_record',
]