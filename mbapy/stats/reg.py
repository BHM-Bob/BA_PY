'''
Author: BHM-Bob 2262029386@qq.com
Date: 2023-04-06 20:44:44
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-09-02 22:15:28
Description: 
'''
from typing import Dict, List

import numpy as np
import pandas as pd
import scipy
from sklearn.linear_model import LinearRegression

if __name__ == '__main__':
    # dev mode
    from mbapy.stats.cluster import (KBayesian, KMeans, cluster,
                                     cluster_support_methods)
else:
    # release mode
    from .cluster import KBayesian, KMeans, cluster, cluster_support_methods

def linear_reg(x:str, y:str, df:pd.DataFrame):
    """
    Perform linear regression on the given DataFrame.

    Parameters:
        x (str): The column name for the independent variable.
        y (str): The column name for the dependent variable.
        df (pd.DataFrame): The DataFrame containing the data.

    Returns:
        dict: A dictionary containing the regression model, slope, intercept, and R-squared value.
            - 'regressor' (LinearRegression): The fitted regression model.
            - 'a' (float): The slope of the regression line.
            - 'b' (float): The intercept of the regression line.
            - 'r2' (float): The R-squared value of the regression.
    """
    x = np.array(df[x]).reshape(-1, 1)
    y = np.array(df[y]).reshape(-1, 1)
    regressor = LinearRegression()
    regressor = regressor.fit(x, y)
    equation_a, equation_b = regressor.coef_.item(), regressor.intercept_.item()
    equation_r2 = regressor.score(x, y)
    return {
        'regressor':regressor,
        'a':equation_a,
        'b':equation_b,
        'r2':equation_r2,
    }
    



if __name__ == '__main__':
    # dev code
    pass
