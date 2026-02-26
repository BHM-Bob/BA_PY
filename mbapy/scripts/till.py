'''
Date: 2023-08-16 16:07:51
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2025-08-06 16:53:59
Description: wait till conditions are met
'''
import argparse
import os
import re
import subprocess
import time
from datetime import datetime
from typing import List, Optional

import psutil
from tqdm import tqdm


def get_gpu_memory_info():
    """Get GPU memory information using nvidia-smi command.
    
    Returns:
        dict: Dictionary containing total, used, and free memory in bytes
    """
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free',
                               '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            match = re.search(r'(\d+),\s*(\d+),\s*(\d+)', result.stdout)
            if match:
                total_mb = int(match.group(1))
                used_mb = int(match.group(2))
                free_mb = int(match.group(3))
                
                total_bytes = total_mb * 1024 * 1024
                used_bytes = used_mb * 1024 * 1024
                free_bytes = free_mb * 1024 * 1024
                
                return {
                    'total': total_bytes,
                    'used': used_bytes,
                    'free': free_bytes
                }
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError, ValueError):
        pass
    
    return None


def get_gpu_utilization():
    """Get GPU utilization using nvidia-smi command.
    
    Returns:
        dict: Dictionary containing GPU utilization percentage
    """
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu',
                               '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            match = re.search(r'(\d+)', result.stdout)
            if match:
                return {
                    'utilization': int(match.group(1))
                }
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError, ValueError):
        pass
    
    return None


def parse_time_string(time_str: str) -> int:
    """Parse time string to seconds.
    
    Supports formats:
    - Pure numbers: "60" -> 60 seconds
    - With units: "3s", "1m2s", "1h3s", "1h8m3s"
    
    Args:
        time_str: Time string to parse
        
    Returns:
        int: Total seconds
        
    Raises:
        ValueError: If the format is invalid
    """
    # Check if it's a pure number
    if time_str.isdigit():
        return int(time_str)
    
    # Regular expression to match time units
    pattern = r'^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$'
    match = re.match(pattern, time_str)
    
    if not match:
        raise ValueError(f"Invalid time format: '{time_str}'. Expected format like '3s', '1m2s', '1h3s', '1h8m3s'")
    
    hours = match.group(1)
    minutes = match.group(2)
    seconds = match.group(3)
    
    total_seconds = 0
    if hours:
        total_seconds += int(hours) * 3600
    if minutes:
        total_seconds += int(minutes) * 60
    if seconds:
        total_seconds += int(seconds)
    
    return total_seconds


def wait_till_time(time_input):
    """Wait for specified duration.
    
    Args:
        time_input: Can be integer seconds or time string like '3s', '1m2s', '1h3s', '1h8m3s'
    """
    # Parse the time input
    if isinstance(time_input, int):
        seconds = time_input
    elif isinstance(time_input, str):
        try:
            seconds = parse_time_string(time_input)
        except ValueError as e:
            print(f"Error: {e}")
            return
    else:
        print(f"Error: Invalid time input type: {type(time_input)}")
        return
    
    # Format the time for display
    if seconds < 60:
        time_display = f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        time_display = f"{minutes}m{remaining_seconds}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        remaining_seconds = seconds % 60
        time_display = f"{hours}h{minutes}m{remaining_seconds}s"
    
    print(f'Waiting for {time_display} ({seconds} seconds)...')
    for _ in tqdm(range(seconds), total=seconds):
        time.sleep(1)


def wait_till_start_time(start_time: Optional[str] = None):
    """Wait until specified start time.
    
    Args:
        start_time: Start time in format 'YYYY-MM-DD HH:MM:SS'
    """
    if start_time is not None:
        start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')  # pyright: ignore[reportAssignmentType]
        total_seconds = int((start_time - datetime.now()).total_seconds())  # pyright: ignore[reportOperatorIssue]
        print(f'sleep until start time: {start_time}, now: {datetime.now()}, total: {start_time - datetime.now()}')  # pyright: ignore[reportOperatorIssue]
        for _ in tqdm(range(total_seconds), total=total_seconds):
            time.sleep(1)


def wait_till_gpu_memory(required_memory_gb: float, check_interval: int = 30, wait_iter: int = 5):
    """Wait until GPU memory is available.
    
    Args:
        required_memory_gb: Required GPU memory in GB
        check_interval: Check interval in seconds
        wait_iter: Wait for more iterations if GPU memory is available
    """
    required_memory_bytes = required_memory_gb * 1024 * 1024 * 1024
    available_iter = 0
    
    print(f'Waiting for GPU memory: {required_memory_gb} GB available...')
    st = time.time()
    while True:
        memory_info = get_gpu_memory_info()
        
        if memory_info is not None:
            total_memory = memory_info['total']
            used_memory = memory_info['used']
            free_memory = memory_info['free']
            print(f'\rGPU memory - Total: {total_memory / 1024**3:.2f} GB, Used: {used_memory / 1024**3:.2f} GB, Free: {free_memory / 1024**3:.2f} GB, total wait: {time.time() - st:.2f} s', end='')
            
            if free_memory >= required_memory_bytes:
                print(f'\nGPU memory available: {free_memory / 1024**3:.2f} GB >= {required_memory_gb} GB, iter: {available_iter}/{wait_iter}')
                available_iter += 1
                if available_iter >= wait_iter:
                    break
            else:
                available_iter = 0
        else:
            print('GPU memory wait timeout, exit')
            exit(-1)

        time.sleep(check_interval)


def wait_till_gpu_utilization(max_utilization: int, check_interval: int = 30, wait_iter: int = 5):
    """Wait until GPU utilization drops below specified threshold.
    
    Args:
        max_utilization: Maximum GPU utilization percentage to wait for
        check_interval: Check interval in seconds
        wait_iter: Wait for more iterations if GPU utilization is below threshold
    """
    available_iter = 0
    
    print(f'Waiting for GPU utilization to drop below {max_utilization}%...')
    st = time.time()
    while True:
        gpu_info = get_gpu_utilization()
        
        if gpu_info is not None:
            utilization = gpu_info['utilization']
            print(f'\rGPU utilization: {utilization}%, total wait: {time.time() - st:.2f} s', end='')
            
            if utilization <= max_utilization:
                print(f'\nGPU utilization below threshold: {utilization}% <= {max_utilization}%, iter: {available_iter}/{wait_iter}')
                available_iter += 1
                if available_iter >= wait_iter:
                    break
            else:
                available_iter = 0
        else:
            print('GPU utilization wait timeout, exit')
            exit(-1)

        time.sleep(check_interval)


def wait_till_process_end(pid: int, check_interval: int = 10):
    """Wait until process with specified PID ends.
    
    Args:
        pid: Process ID to wait for
        check_interval: Check interval in seconds
    """
    print(f'Waiting for process {pid} to end...')
    st = time.time()
    while True:
        if not psutil.pid_exists(pid):
            print(f'\nProcess {pid} has ended after {time.time() - st:.2f} seconds')
            break
        print(f'\rProcess {pid} still running, total wait: {time.time() - st:.2f} s', end='')
        time.sleep(check_interval)


def wait_till_cpu_utilization(max_utilization: int, check_interval: int = 30, wait_iter: int = 5):
    """Wait until CPU utilization drops below specified threshold.
    
    Args:
        max_utilization: Maximum CPU utilization percentage to wait for
        check_interval: Check interval in seconds
        wait_iter: Wait for more iterations if CPU utilization is below threshold
    """
    available_iter = 0
    
    print(f'Waiting for CPU utilization to drop below {max_utilization}%...')
    st = time.time()
    while True:
        cpu_util = psutil.cpu_percent(interval=1)
        print(f'\rCPU utilization: {cpu_util:.1f}%, total wait: {time.time() - st:.2f} s', end='')
        
        if cpu_util <= max_utilization:
            print(f'\nCPU utilization below threshold: {cpu_util:.1f}% <= {max_utilization}%, iter: {available_iter}/{wait_iter}')
            available_iter += 1
            if available_iter >= wait_iter:
                break
        else:
            available_iter = 0

        time.sleep(check_interval)


def wait_till_memory(required_memory_gb: float, check_interval: int = 30, wait_iter: int = 5):
    """Wait until system memory is available.
    
    Args:
        required_memory_gb: Required system memory in GB
        check_interval: Check interval in seconds
        wait_iter: Wait for more iterations if memory is available
    """
    required_memory_bytes = required_memory_gb * 1024 * 1024 * 1024
    available_iter = 0
    
    print(f'Waiting for system memory: {required_memory_gb} GB available...')
    st = time.time()
    while True:
        mem = psutil.virtual_memory()
        total_memory = mem.total
        used_memory = mem.used
        free_memory = mem.available
        print(f'\rMemory - Total: {total_memory / 1024**3:.2f} GB, Used: {used_memory / 1024**3:.2f} GB, Free: {free_memory / 1024**3:.2f} GB, total wait: {time.time() - st:.2f} s', end='')
        
        if free_memory >= required_memory_bytes:
            print(f'\nMemory available: {free_memory / 1024**3:.2f} GB >= {required_memory_gb} GB, iter: {available_iter}/{wait_iter}')
            available_iter += 1
            if available_iter >= wait_iter:
                break
        else:
            available_iter = 0

        time.sleep(check_interval)


def wait_till_file(path: str, check_interval: int = 10):
    """Wait until file or folder exists at specified path.
    
    Args:
        path: Path to file or folder to wait for
        check_interval: Check interval in seconds
    """
    print(f'Waiting for file/folder: {path}...')
    st = time.time()
    while True:
        if os.path.exists(path):
            print(f'\nFile/folder {path} exists after {time.time() - st:.2f} seconds')
            break
        print(f'\rWaiting for {path}, total wait: {time.time() - st:.2f} s', end='')
        time.sleep(check_interval)


def wait_till_folder_file_count(path: str, min_count: int, check_interval: int = 10):
    """Wait until folder contains at least specified number of files (non-recursive).
    
    Args:
        path: Path to folder to check
        min_count: Minimum number of files required
        check_interval: Check interval in seconds
    """
    print(f'Waiting for folder {path} to contain at least {min_count} files...')
    st = time.time()
    while True:
        if os.path.isdir(path):
            file_count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
            print(f'\rFolder {path} has {file_count} files, total wait: {time.time() - st:.2f} s', end='')
            
            if file_count >= min_count:
                print(f'\nFolder {path} has {file_count} files >= {min_count} after {time.time() - st:.2f} seconds')
                break
        else:
            print(f'\rFolder {path} does not exist, total wait: {time.time() - st:.2f} s', end='')
        
        time.sleep(check_interval)


def main(sys_args: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description='Wait till conditions are met')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    parser_time = subparsers.add_parser('time', help='Wait for specified duration')
    parser_time.add_argument('time', type=str, help='Duration to wait (e.g., 60, 3s, 1m2s, 1h3s, 1h8m3s)')
    
    parser_start_time = subparsers.add_parser('start-time', help='Wait until specified start time')
    parser_start_time.add_argument('time', type=str, help='Start time in format YYYY-MM-DD HH:MM:SS')
    
    parser_cuda_mem = subparsers.add_parser('cuda-mem', help='Wait until GPU memory is available')
    parser_cuda_mem.add_argument('memory', type=float, help='Required GPU memory in GB')
    parser_cuda_mem.add_argument('--check-interval', type=int, default=30, help='Check interval in seconds (default: 30)')
    parser_cuda_mem.add_argument('--wait-iter', type=int, default=5, help='Wait iterations after condition met (default: 5)')
    
    parser_cuda = subparsers.add_parser('cuda', help='Wait until GPU utilization drops below threshold')
    parser_cuda.add_argument('max-utilization', type=int, help='Maximum GPU utilization percentage')
    parser_cuda.add_argument('--check-interval', type=int, default=30, help='Check interval in seconds (default: 30)')
    parser_cuda.add_argument('--wait-iter', type=int, default=5, help='Wait iterations after condition met (default: 5)')
    
    parser_process = subparsers.add_parser('process', help='Wait until process ends')
    parser_process.add_argument('pid', type=int, help='Process ID to wait for')
    parser_process.add_argument('--check-interval', type=int, default=30, help='Check interval in seconds (default: 30)')
    
    parser_cpu = subparsers.add_parser('cpu', help='Wait until CPU utilization drops below threshold')
    parser_cpu.add_argument('max-utilization', type=int, help='Maximum CPU utilization percentage')
    parser_cpu.add_argument('--check-interval', type=int, default=30, help='Check interval in seconds (default: 30)')
    parser_cpu.add_argument('--wait-iter', type=int, default=5, help='Wait iterations after condition met (default: 5)')
    
    parser_mem = subparsers.add_parser('mem', help='Wait until system memory is available')
    parser_mem.add_argument('memory', type=float, help='Required system memory in GB')
    parser_mem.add_argument('--check-interval', type=int, default=30, help='Check interval in seconds (default: 30)')
    parser_mem.add_argument('--wait-iter', type=int, default=5, help='Wait iterations after condition met (default: 5)')
    
    parser_file = subparsers.add_parser('file', help='Wait until file or folder exists')
    parser_file.add_argument('path', type=str, help='Path to file or folder')
    parser_file.add_argument('--check-interval', type=int, default=30, help='Check interval in seconds (default: 30)')
    
    parser_folder = subparsers.add_parser('folder', help='Wait until folder contains minimum number of files')
    parser_folder.add_argument('path', type=str, help='Path to folder')
    parser_folder.add_argument('min-count', type=int, help='Minimum number of files required')
    parser_folder.add_argument('--check-interval', type=int, default=30, help='Check interval in seconds (default: 30)')
    
    args = parser.parse_args(sys_args)
    
    if args.command == 'time':
        wait_till_time(args.time)
    elif args.command == 'start-time':
        wait_till_start_time(args.time)
    elif args.command == 'cuda-mem':
        wait_till_gpu_memory(args.memory, args.check_interval, args.wait_iter)
    elif args.command == 'cuda':
        wait_till_gpu_utilization(getattr(args, 'max-utilization'), args.check_interval, args.wait_iter)
    elif args.command == 'process':
        wait_till_process_end(args.pid, args.check_interval)
    elif args.command == 'cpu':
        wait_till_cpu_utilization(getattr(args, 'max-utilization'), args.check_interval, args.wait_iter)
    elif args.command == 'mem':
        wait_till_memory(args.memory, args.check_interval, args.wait_iter)
    elif args.command == 'file':
        wait_till_file(args.path, args.check_interval)
    elif args.command == 'folder':
        wait_till_folder_file_count(args.path, getattr(args, 'min-count'), args.check_interval)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
