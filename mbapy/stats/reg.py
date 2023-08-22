'''
Author: BHM-Bob 2262029386@qq.com
Date: 2023-04-06 20:44:44
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-08-22 23:26:27
Description: 
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import (DBSCAN, AffinityPropagation,
                             AgglomerativeClustering, Birch, KMeans, MeanShift,
                             MiniBatchKMeans)
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
    
cluster_support_methods = ['DBSCAN', 'Birch', 'KMeans', 'MiniBatchKMeans',
                           'MeanShift', 'GaussianMixture', 'AgglomerativeClustering',
                           'AffinityPropagation']
    
def cluster(data, n_clusters:int, method:str, **kwargs):
    """
    Clusters data using various clustering methods.

    Parameters:
        data (array-like): The input data to be clustered.
        n_clusters (int): The number of clusters to create.
        method (str): The clustering method to use, one of 
            ['DBSCAN', 'Birch', 'KMeans', 'MiniBatchKMeans', 'MeanShift',
            'GaussianMixture', 'AgglomerativeClustering', 'AffinityPropagation'].
        **kwargs: Additional keyword arguments specific to each clustering method.

    Returns:
        array-like: The cluster labels assigned to each data point.
        
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
        return KMeans(n_clusters=n_clusters).fit_predict(data)
    elif method == 'MiniBatchKMeans':
        return MiniBatchKMeans(n_clusters=n_clusters).fit_predict(data)
    elif method == 'MeanShift':
        return MeanShift().fit_predict(data)
    elif method == 'GaussianMixture':
        kwargs = set_default_kwargs(kwargs, n_components=n_clusters, random_state = 777)
        model = GaussianMixture(**kwargs)
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
    from mbapy.stats import pca
    n_classes = 3
    # 模拟数据集
    X, _ = make_classification(n_samples=1000*n_classes, n_features=2, n_informative=2,
                               n_redundant=0, n_classes=n_classes, n_clusters_per_class=1, random_state=4)
    # 真实数据集MWM
    # df = pd.read_excel(r'data/plot.xlsx',sheet_name='MWM')
    # tags = [col for col in df.columns if col not in ['Unnamed: 0', 'Animal No.', 'Trial Type', 'Title', 'Start time', 'Memo', 'Day', 'Animal Type']]
    # 真实数据集XM
    df = pd.read_excel(r'data/plot.xlsx',sheet_name='xm')
    tags = [col for col in df.columns if col not in ['solution', 'type']]
    
    X = df.loc[ : ,tags].values
    pos = pca(X, 2)
    print(X.shape, pos.shape)

    fig, axs = plt.subplots(2, 4, figsize = (12, 7))
    for i in range(2):
        for j in range(4):
            method = cluster_support_methods[i*4+j]
            yhat = cluster(X, n_classes, method)
            # 检索唯一群集
            clusters_id = np.unique(yhat)
            # 为每个群集的样本创建散点图
            for cluster_id in clusters_id:
                # 获取此群集的示例的行索引
                row_ix = np.where(yhat == cluster_id)
                # 创建这些样本的散布
                axs[i][j].scatter(pos[row_ix, 0], pos[row_ix, 1])
            axs[i][j].set_title(method)
    # 绘制散点图
    plt.show()