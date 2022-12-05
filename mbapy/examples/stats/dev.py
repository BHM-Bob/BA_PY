'''
Author: BHM-Bob 2262029386@qq.com
Date: 2022-12-05 16:18:43
LastEditors: BHM-Bob
LastEditTime: 2022-12-05 16:23:17
Description: 
'''
import numpy as np
import pandas as pd
from scipy import stats

data=[[25,21,10],[82,88,30],[223,16,5]]
df=pd.DataFrame(data,index=['美式咖啡','拿铁咖啡','卡布奇诺'],columns=['IT','行政','工程'])
kt=stats.chi2_contingency(df)
# stats.chisquare()
print('卡方值=%.4f, p值=%.4f, 自由度=%i expected_frep=%s'%kt)