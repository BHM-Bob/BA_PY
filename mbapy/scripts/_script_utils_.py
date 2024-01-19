'''
Date: 2024-01-12 16:06:35
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2024-01-12 16:07:12
FilePath: \BA_PY\mbapy\scripts\_script_utils_.py
Description: 
'''
from pathlib import Path

def clean_path(path: str):
    return Path(path.replace('"', '').replace("'", '')).resolve()