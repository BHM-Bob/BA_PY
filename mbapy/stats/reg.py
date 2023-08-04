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
from sklearn.cluster import (DBSCAN, Birch, KMeans, MeanShift, MiniBatchKMeans,
                             AgglomerativeClustering, AffinityPropagation)
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture

if __name__ == '__main__':
    # dev mode
    from mbapy.base import put_err, set_default_kwargs
else:
    # release mode
    from ..base import put_err, set_default_kwargs

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
    
def cluster(data, n_clusters:int, method:str, **kwargs):
    """
    Notes:
        - Kmeans: 此算法尝试最小化群集内数据点的方差。K 均值最适合用于较小的数据集，因为它遍历所有数据点。
                这意味着，如果数据集中有大量数据点，则需要更多时间来对数据点进行分类。
        - GaussianMixture: 高斯混合模型使用多个高斯分布来拟合任意形状的数据。
                在这个混合模型中，有几个单一的高斯模型充当隐藏层。因此，
                该模型计算数据点属于特定高斯分布的概率，即它将属于的聚类。
    """
    if method == 'DBSCAN':
        kwargs = set_default_kwargs(kwargs, eps = 0.5, min_samples = 3)
        return DBSCAN(**kwargs).fit_predict(data)
    elif method == 'Birch':
        model = Birch(n_clusters=n_clusters, **kwargs)
        return model.fit_predict(data)
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
    elif method == 'AgglomerativeClustering':
        model = AgglomerativeClustering(n_clusters = n_clusters)
        return model.fit(data).labels_
    elif method == 'AffinityPropagation':
        model = AffinityPropagation(**kwargs)
        return model.fit_predict(data)
    else:
        return put_err(f'Unknown method {method}, return None', None)


if __name__ == '__main__':
    # dev code
    # 定义数据集
    X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2,
                               n_redundant=0, n_clusters_per_class=1, random_state=4)
    yhat = cluster(X, 2, 'AffinityPropagation')
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