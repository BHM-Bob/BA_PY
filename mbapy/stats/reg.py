'''
Author: BHM-Bob 2262029386@qq.com
Date: 2023-04-06 20:44:44
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-08-02 23:35:32
Description: 
'''
import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.cluster import DBSCAN, Birch, KMeans, MeanShift, MiniBatchKMeans
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture

if __name__ == '__main__':
    # dev mode
    from mbapy.base import put_err
else:
    # release mode
    from ..base import put_err

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
    
def cluster(data, n_clusters:int, method:str):
    if method == 'DBSCAN':
        return DBSCAN(eps=0.5, min_samples=n_clusters).fit_predict(data)
    elif method == 'Birch':
        return Birch(threshold=0.01, n_clusters=n_clusters).predict(data)
    elif method == 'KMeans':
        return KMeans(n_clusters=n_clusters).predict(data)
    elif method == 'MiniBatchKMeans':
        return MiniBatchKMeans(n_clusters=n_clusters).predict(data)
    elif method == 'MeanShift':
        return MeanShift().fit_predict(data)
    elif method == 'MiniBatchKMeans':
        return MiniBatchKMeans(n_clusters=n_clusters).fit_predict(data)
    elif method == 'GaussianMixture':
        model = GaussianMixture(n_components=n_clusters)
        model.fit(data)
        return model.predict(data)
    else:
        return put_err(f'Unknown method {method}, return None', None)


if __name__ == '__main__':
    # dev code
    # 定义数据集
    X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2,
                               n_redundant=0, n_clusters_per_class=1, random_state=4)
    yhat = cluster(X, 2, 'GaussianMixture')
    # 检索唯一群集
    clusters = np.unique(yhat)
    # 为每个群集的样本创建散点图
    for cluster in clusters:
        # 获取此群集的示例的行索引
        row_ix = np.where(yhat == cluster)
        # 创建这些样本的散布
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
    # 绘制散点图
    pyplot.show()