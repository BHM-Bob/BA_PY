'''
Author: BHM-Bob 2262029386@qq.com
Date: 2023-04-04 16:45:23
LastEditors: BHM-Bob
LastEditTime: 2023-04-04 17:26:13
Description: 
'''
import scipy
import numpy as np

#置信区间
def get_interval(mean = None, se = None, data = None, alpha:float = 0.95):
    """± 1.96 * SE or other depends on alpha"""
    assert se is not None or data is not None, 'se is None and data is None'
    kwargs = {
        'scale':se if se is not None else scipy.stats.sem(data)       
    }
    if data is not None:
        kwargs.update({'df':len(data) - 1, 'mean':np.mean(data)})
    return scipy.stats.norm.interval(alpha = alpha, **kwargs)

#单样本T检验
from scipy.stats import ttest_1samp

def ttest_ind(x1, x2):
    """独立样本t检验(双样本T检验):检验两组独立样本均值是否相等"""
    levene = scipy.stats.levene(x1, x2)
    #levene 检验P值 > 0.05，接受原假设，认为两组方差相等
    #如不相等， scipy.stats.ttest_ind() 函数中的参数 equal_var 需要设置成 False
    return levene.pvalue, scipy.stats.ttest_ind(x1, x2, equal_var=levene.pvalue > 0.05)

#配对样本T检验:比较从同一总体下分出的两组样本在不同场景下，均值是否存在差异(两个样本的样本量要相同)
from scipy.stats import ttest_rel

