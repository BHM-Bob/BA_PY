'''
Author: BHM-Bob 2262029386@qq.com
Date: 2023-04-06 20:44:44
LastEditors: BHM-Bob G 2262029386@qq.com
LastEditTime: 2023-08-07 19:15:07
Description: 
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import (DBSCAN, Birch, KMeans as sk_KMeans, MeanShift, MiniBatchKMeans,
                             AgglomerativeClustering, AffinityPropagation)
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist

if __name__ == '__main__':
    # dev mode
    from mbapy.base import put_err, set_default_kwargs, autoparse, get_default_for_None
else:
    # release mode
    from ..base import put_err, set_default_kwargs, autoparse, get_default_for_None

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
    
class KMeans:
    """
    Parameters:
        - n_clusters(int): number of clusters, if set to None, will auto search from 1 to sum of data to make fit for tolerance.
    """
    @autoparse
    def __init__(self, n_clusters:int = None, tolerance:float = 0.0001, max_iter:int = 1000,
                 init_method = 'prob', **kwargs) -> None:
        self.centers = None # should be a ndarray with shape [n, D]
        self.data_group_id = None
        
    def reset(self, **kwargs):
        self.centers = None
        self.data_group_id = None
        
    def _calcu_length_mat(self, data:np.ndarray, centers:np.ndarray = None,
                      backend:str = 'scipy', metric = 'euclidean'):
        """return length mat with shape: [N, n], N is sum data, n is sum clusters(centers)"""
        centers = get_default_for_None(centers, self.centers)
        if backend == 'scipy':
            return cdist(data, centers, metric = metric) # [N, n]
        else:
            raise NotImplementedError
        
    def _choose_center_from_data(self, data: np.ndarray, centers:np.ndarray):
        # get length for every data to every centers, and calcu sum for each data
        length = self._calcu_length_mat(data, centers).sum(axis = -1) # [N, ]
        # get new_center_idx by prob or just using max
        if self.init_method == 'prob':                
            prob = length / np.sum(length)
            idx = np.random.choice(data.shape[0], p=prob)
        elif self.init_method == 'max':
            idx = np.argmax(length)
        return data[idx]
        
    def _init_centers(self, data:np.ndarray):
        # choose the first center randomly
        first_center_idx = int(np.random.uniform(0, data.shape[0]))
        self.centers = data[first_center_idx].reshape(1, -1) # [n=1, D]
        # generate left centers by kmeans++
        for _ in range(self.n_clusters - 1):
            new_center = self._choose_center_from_data(data, self.centers)
            self.centers = np.vstack([self.centers, new_center])
        return self.centers
    
    def loss_fn(self, data:np.ndarray, centers:np.ndarray = None):
        centers = get_default_for_None(centers, self.centers)
        self.loss = self._calcu_length_mat(data, centers).mean()
        return self.loss
    
    def fit(self, data:np.ndarray, **kwargs):
        self._init_centers(data)
        prev_loss = np.nan
        for _ in range(self.max_iter):
            # sort data to groups id by now centers, get data_group_id
            new_centers = np.zeros([self.n_clusters, data.shape[-1]])
            length_mat = self._calcu_length_mat(data) # [N, n]
            data_group_id = length_mat.argmin(axis = -1) # [N, ]
            # sort data to groups, set new centers to be the mean of each group
            for group_x_id in range(self.n_clusters):
                group_x_data_id = np.argwhere(data_group_id == group_x_id).reshape(-1) # data id for one group
                if group_x_data_id.shape[0] == 0:
                    # empty group(center), perform init for this center
                    non_null_centers = self.centers[np.unique(data_group_id)]
                    new_centers[group_x_id] = self._choose_center_from_data(data, non_null_centers)
                    # if there has multi null groups,
                    # need update data_group_id to update non_null_centers
                    self.centers[group_x_id] = new_centers[group_x_id]
                    data_group_id = self._calcu_length_mat(data).argmin(axis = -1)
                else:
                    new_centers[group_x_id] = data[group_x_data_id].mean(axis=0)
            # check if loss has no changes
            if self.loss_fn(data) == prev_loss:
                break
            prev_loss = self.loss
            # move centers to new centers(average of groups)
            self.centers = new_centers
            self.data_group_id = data_group_id
        return self.centers
    
    def fit_times(self, data:np.ndarray, times:int = 3, **kwargs):
        self.fit(data)
        loss_records = [self.loss]
        centers_records = [self.centers]
        for _ in range(times - 1):
            self.fit(data)
            loss_records.append(self.loss)
            centers_records.append(self.centers)
        min_idx = np.argmin(np.array(loss_records))
        self.loss = loss_records[min_idx]
        self.centers = centers_records[min_idx]
        return self.centers

    def predict(self, data:np.ndarray):
        length_mat = self._calcu_length_mat(data) # [N, n]
        return length_mat.argmin(axis = -1) # [N, ]
        
    def fit_predict(self, data:np.ndarray, predict_data = None, fit_times = 1, **kwargs):
        if fit_times == 1:
            self.fit(data)
        else:
            self.fit_times(data, fit_times)
        return self.predict(get_default_for_None(predict_data, data))
    
cluster_support_methods = ['DBSCAN', 'Birch', 'KMeans', 'MiniBatchKMeans',
                           'MeanShift', 'GaussianMixture', 'AgglomerativeClustering',
                           'AffinityPropagation']
    
def cluster(data, n_clusters:int, method:str, norm = None, norm_dim = None, **kwargs):
    """
    Notes:
        - Kmeans: 此算法尝试最小化群集内数据点的方差。K 均值最适合用于较小的数据集，因为它遍历所有数据点。
                这意味着，如果数据集中有大量数据点，则需要更多时间来对数据点进行分类。
        - GaussianMixture: 高斯混合模型使用多个高斯分布来拟合任意形状的数据。
                在这个混合模型中，有几个单一的高斯模型充当隐藏层。因此，
                该模型计算数据点属于特定高斯分布的概率，即它将属于的聚类。
    """
    # TODO : imp norm_dim
    if norm_dim is not None:
        raise NotImplementedError
    if norm is not None:
        if norm == 'div_max':
            data = data/data.max()
            
    if method == 'DBSCAN':
        kwargs = set_default_kwargs(kwargs, eps = 0.5, min_samples = 3)
        return DBSCAN(**kwargs).fit_predict(data)
    elif method == 'Birch':
        model = Birch(n_clusters=n_clusters, **kwargs)
        return model.fit_predict(data)
    elif method == 'KMeans':
        return sk_KMeans(n_clusters=n_clusters, n_init = 'auto').fit_predict(data)
    elif method == 'MiniBatchKMeans':
        return MiniBatchKMeans(n_clusters=n_clusters, n_init= 'auto').fit_predict(data)
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
    # X, _ = make_classification(n_samples=1000*n_classes, n_features=2, n_informative=2,
    #                            n_redundant=0, n_classes=n_classes, n_clusters_per_class=1, random_state=4)
    # 真实数据集MWM
    df = pd.read_excel(r'data/plot.xlsx',sheet_name='MWM')
    tags = [col for col in df.columns if col not in ['Unnamed: 0', 'Animal No.', 'Trial Type', 'Title', 'Start time', 'Memo', 'Day', 'Animal Type']]
    # 真实数据集XM
    # df = pd.read_excel(r'data/plot.xlsx',sheet_name='xm')
    # tags = [col for col in df.columns if col not in ['solution', 'type']]
    
    X = df.loc[ : ,tags].values
    pos = pca(X, 2)
    print(X.shape, pos.shape)

    fig, axs = plt.subplots(3, 3, figsize = (10, 10))
    for i in range(3):
        for j in range(3):
            if i == 0 and j == 0:
                method = 'mbapy-KMeans'
                model = KMeans(n_classes)
                yhat = model.fit_predict(X, fit_times=10)
            else:
                method = cluster_support_methods[i*3+j - 1]
                yhat = cluster(X, n_classes, method, 'div_max')
            # 检索唯一群集
            clusters_id = np.unique(yhat)
            # 为每个群集的样本创建散点图
            for cluster_id in clusters_id:
                # 获取此群集的示例的行索引
                row_ix = np.where(yhat == cluster_id)
                # 创建这些样本的散布
                axs[i][j].scatter(pos[row_ix, 0], pos[row_ix, 1])
            if method == 'mbapy-KMeans':
                # plot centers
                axs[i][j].scatter(model.centers[:, 0], model.centers[:, 1])
            axs[i][j].set_title(method)
    # 绘制散点图
    plt.show()