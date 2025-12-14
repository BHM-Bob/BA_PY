'''
Date: 2024-07-16 13:43:25
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2025-08-09 09:18:53
Description: 
'''

import argparse
import math
import os
from typing import Dict, List, Tuple

if __name__ == '__main__':
    from mbapy.file import format_file_size, get_paths_with_extension
    from mbapy.scripts._script_utils_ import _print, clean_path, show_args
else:
    from ..file import format_file_size, get_paths_with_extension
    from ._script_utils_ import _print, clean_path, show_args


def create_size_bins(sizes: List[int]) -> Dict[str, Tuple[int, int]]:
    """
    创建文件大小的bincount统计
    
    Parameters:
        sizes: 文件大小列表（字节）
        
    Returns:
        包含大小范围和文件数量的字典
    """
    if not sizes:
        return {}
    
    # 定义大小范围（字节）
    size_ranges = [
        (0, 1024),                    # 0-1KB
        (1024, 10*1024),              # 1-10KB
        (10*1024, 100*1024),          # 10-100KB
        (100*1024, 1024*1024),        # 100KB-1MB
        (1024*1024, 10*1024*1024),    # 1-10MB
        (10*1024*1024, 100*1024*1024), # 10-100MB
        (100*1024*1024, 1024*1024*1024), # 100MB-1GB
        (1024*1024*1024, float('inf')) # 1GB以上
    ]
    
    bins = {}
    for min_size, max_size in size_ranges:
        if max_size == float('inf'):
            count = sum(1 for size in sizes if size >= min_size)
            label = f"≥{format_file_size(min_size)}"
        else:
            count = sum(1 for size in sizes if min_size <= size < max_size)
            label = f"{format_file_size(min_size)}-{format_file_size(max_size)}"
        
        if count > 0:
            bins[label] = (count, min_size)
    
    return bins


def display_size_distribution(bins: Dict[str, Tuple[int, int]], f_handle) -> None:
    """
    以TXTUI美观的格式显示文件大小分布
    
    Parameters:
        bins: 包含大小范围和文件数量的字典
        f_handle: 文件句柄
    """
    if not bins:
        _print("No files to display distribution", f_handle)
        return
    
    total_files = sum(count for count, _ in bins.values())
    max_count = max(count for count, _ in bins.values())
    
    # 按大小范围排序（从小到大）
    sorted_bins = sorted(bins.items(), key=lambda x: x[1][1])
    
    # 计算最大标签长度用于对齐
    max_label_len = max(len(label) for label, _ in sorted_bins)
    
    # 显示分布表
    _print(f"{'Size Range':<{max_label_len}} | {'Count':>6} | {'Percentage':>10} | {'Histogram'}", f_handle)
    _print('-' * (max_label_len + 40), f_handle)
    
    for label, (count, _) in sorted_bins:
        percentage = (count / total_files) * 100
        bar_length = max(1, int((count / max_count) * 30))  # 最大30个字符的直方图
        histogram = '█' * bar_length
        
        _print(f"{label:<{max_label_len}} | {count:>6} | {percentage:>9.1f}% | {histogram}", f_handle)
    
    _print('-' * (max_label_len + 40), f_handle)
    _print(f"{'Total':<{max_label_len}} | {total_files:>6} | {'100.0%':>10} |", f_handle)
    

def main(sys_args: List[str] = None):
    args_paser = argparse.ArgumentParser(description = 'delete files with specific suffix or sub-string in name')
    args_paser.add_argument('-t', '--type', type = str, nargs='+', default=[],
                            help='format of files to remove, splited by ",". Default is %(default)s')
    args_paser.add_argument('-n', '--name', type = str, default='',
                            help='sub-string of name of files to remove. Default is %(default)s')
    args_paser.add_argument('-i', '--input', type=str, default='.',
                            help='files path or dir path.')
    args_paser.add_argument('-o', '--output', type=str, default=None,
                            help='output list txt file path or dir path. Default is %(default)s.')
    args_paser.add_argument('-r', '--recursive', action='store_true', default=False,
                            help='FLAG, recursive search. Default is %(default)s.')
    args_paser.add_argument('-s', '--sum', action='store_true', default=False,
                            help='FLAG, sum size of files. Default is %(default)s.')
    args_paser.add_argument('--sort-by-size', action='store_true', default=False,
                            help='FLAG, sort files by size. Default is %(default)s.')
    args = args_paser.parse_args(sys_args)
    
    # process IO path
    args.input = clean_path(args.input)
    if args.output is not None:
        if os.path.isdir(args.output):
            args.output = os.path.join(args.output, '__mbapy_scripts_find_file_result.txt')
        args.output = clean_path(args.output)
    f_handle = open(args.output, 'w') if args.output is not None else None
    # show args
    show_args(args, ['type', 'name', 'input', 'output', 'recursive', 'sort_by_size'])
    
    paths = get_paths_with_extension(args.input, args.type,
                                     args.recursive, args.name, c_version=True)
    
    # 如果启用了按大小排序，则对文件进行排序
    if args.sort_by_size:
        # 获取文件大小并排序
        file_sizes = [(path, os.path.getsize(path)) for path in paths]
        file_sizes.sort(key=lambda x: x[1], reverse=True)  # 从大到小排序
        paths = [(path, filesize) for path, filesize in file_sizes]
    
    _print(f'files finded: {len(paths)} in dir: {args.input}', f_handle)
    # show info
    filesizes = []
    for path in paths:
        if isinstance(path, tuple):
            path, filesize = path
        else:
            filesize = os.path.getsize(path)
        filesizes.append(filesize)
        if not args.sum:
            info_str = f'{format_file_size(filesize):>} {path[len(str(args.input)):]}'
            _print(info_str, f_handle)
    total_size = sum(filesizes)
    _print(f'files finded: {len(paths)} in dir: {args.input}', f_handle) # print again in bottom
    _print(f'total files size: {format_file_size(total_size)}, avg size: {format_file_size(total_size/len(paths))}', f_handle)
    
    # 如果启用了按大小排序，则添加bincount统计
    if args.sort_by_size:
        _print('', f_handle)
        _print('File Size Distribution:', f_handle)
        _print('=' * 50, f_handle)
        
        # 创建bincount统计
        size_bins = create_size_bins(filesizes)
        
        # 显示bincount统计
        display_size_distribution(size_bins, f_handle)
    
    return paths


if __name__ == "__main__":
    main()