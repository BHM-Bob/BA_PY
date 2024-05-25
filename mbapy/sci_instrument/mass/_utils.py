'''
Date: 2024-05-22 10:00:28
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2024-05-25 08:38:01
Description: 
'''

from typing import Callable, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy

if __name__ == '__main__':
    from mbapy.base import put_err
    from mbapy.plot import get_palette
    from mbapy.sci_instrument.mass._base import MassData
else:
    from ...base import put_err
    from ...plot import get_palette
    from ._base import MassData
    
