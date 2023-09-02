'''
Author: BHM-Bob 2262029386@qq.com
Date: 2023-04-06 20:44:44
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-09-02 21:59:39
Description: 
'''
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from sklearn.cluster import (DBSCAN, AffinityPropagation,
                             AgglomerativeClustering, Birch)
from sklearn.cluster import KMeans as sk_KMeans
from sklearn.cluster import MeanShift, MiniBatchKMeans
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture

try:
    import torch
    from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, hp, tpe
except:
    pass # mbapy now do not compulsively require hyperopt

if __name__ == '__main__':
    # dev mode
    from mbapy.base import (autoparse, get_default_for_None, get_num_digits,
                            put_err, set_default_kwargs)
else:
    # release mode
    from ..base import (autoparse, get_default_for_None, get_num_digits,
                        put_err, set_default_kwargs)

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
    """ KMeans clustering algorithm implementation.

    Parameters:
        - n_clusters(int): number of clusters, if set to None, will auto search from 1 to sum of data to make fit for tolerance.
        - tolerance(float): tolerance for convergence, default is 0.0001.
        - max_iter(int): maximum number of iterations, default is 1000.
        - init_method(str): initialization method, either 'prob' or 'max', default is 'prob'.
        - backend(str): backend for calculating distance, default is 'scipy'.
            - 'scipy': use scipy cdist.
            - 'pytorch': use pytorch cdist.
        - **kwargs: additional keyword arguments.
    """
    class BackEnd:
        def __init__(self, backend:str) -> None:
            self.backend = backend
            if backend == 'scipy':
                self._backend = np
                self.array = np.array
            elif backend == 'pytorch':
                self._backend = torch
                self.array = torch.tensor
        def cat(self, *args, **kwargs):
            if self.backend == 'scipy':
                return np.concatenate(*args, **kwargs, axis = 0)
            elif self.backend == 'pytorch':
                return torch.cat(*args, **kwargs, dim = 0)
        def cdist(self, data, centers):
            if self.backend == 'scipy':
                return scipy.spatial.distance.cdist(data, centers, metric = 'euclidean')
            elif self.backend == 'pytorch':
                if isinstance(centers, np.ndarray):
                    centers = torch.tensor(centers, dtype = torch.float64, device = data.device)
                return torch.cdist(data.to(dtype = torch.float64), centers.to(dtype = torch.float64))
        def random_choice(self, n:int, p):
            if self.backend == 'scipy':
                return np.random.choice(n, p = p)
            elif self.backend == 'pytorch':
                return torch.multinomial(p, n)[0]
        
    @autoparse
    def __init__(self, n_clusters:int = None, tolerance:float = 0.0001, max_iter:int = 1000,
                 init_method = 'prob', backend:str = 'scipy', **kwargs) -> None:
        self.centers = None # should be a ndarray with shape [n, D]
        self.data_group_id = None
        self._backend = self.BackEnd(backend)
        
    def reset(self, **kwargs):
        """
        Reset the centers and data group id to None.
        """
        self.centers = None
        self.data_group_id = None
        
    def _calcu_length_mat(self, data:np.ndarray, centers:np.ndarray = None):
        """
        Calculate the length matrix between data and centers.

        Parameters:
            - data(ndarray): data to be clustered, should be a ndarray with shape [N, D].
            - centers(ndarray): centers of the clusters, should be a ndarray with shape [n, D], default is None.

        Returns:
            - length_mat(ndarray): length matrix with shape [N, n], N is sum data, n is sum clusters(centers).
        """
        centers = get_default_for_None(centers, self.centers)
        return self._backend.cdist(data, centers) # [N, n]
        
    def _choose_center_from_data(self, data: np.ndarray, centers:np.ndarray):
        """
        Choose a center point from the given data and centers.
            - If self.init_method is 'prob', then choose the center point with probability;
            - If self.init_method is 'max', then choose the center point with maximum length.

        Parameters:
            - data (np.ndarray): The input data array.
            - centers (np.ndarray): The centers array.

        Returns:
            np.ndarray: The selected center point from the data.
        """
        # get length for every data to every centers, and calcu sum for each data
        length = self._calcu_length_mat(data, centers).sum(axis = -1) # [N, ]
        # get new_center_idx by prob or just using max
        if self.init_method == 'prob':
            prob = length / self._backend._backend.sum(length)
            idx = self._backend.random_choice(data.shape[0], p=prob)
        elif self.init_method == 'max':
            idx = self._backend._backend.argmax(length)
        return data[idx]
    
    def _init_centers(self, data:np.ndarray):
        """
        Initializes the cluster centers for the K-Means clustering algorithm.

        Parameters:
            - data (np.ndarray): The input data for clustering.
            - backend (str, optional): The backend for calculating distance. Defaults to 'scipy'.

        Returns:
            np.ndarray: The initialized cluster centers.
        """
        # choose the first center randomly
        first_center_idx = int(np.random.uniform(0, data.shape[0]))
        self.centers = data[first_center_idx].reshape(1, -1) # [n=1, D]
        # generate left centers by kmeans++
        for _ in range(self.n_clusters - 1):
            new_center = self._choose_center_from_data(data, self.centers).reshape(1, -1)
            self.centers = self._backend.cat([self.centers, new_center])
        return self.centers
    
    def loss_fn(self, data:np.ndarray, centers:np.ndarray = None):
        """
        Calculate the loss function for the given data and centers.

        Parameters:
        - data (np.ndarray): The input data array.
        - centers (np.ndarray): The centers array. If not provided, the default centers will be used.

        Returns:
        - float: The calculated loss.
        """
        centers = get_default_for_None(centers, self.centers)
        self.loss = self._calcu_length_mat(data, centers).mean()
        return self.loss
    
    def fit(self, data:np.ndarray, **kwargs):
        """
        Fits the K-means clustering model to the given data.

        Parameters:
            - data (np.ndarray): The input data for clustering.
            - backend (str, optional): The backend for calculating distance. Defaults to 'scipy'.
            - **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: The final cluster centers.
        """
        self._init_centers(data)
        prev_loss = None
        for _ in range(self.max_iter):
            # sort data to groups id by now centers, get data_group_id
            kwgs = {'device' : data.device} if self.backend == 'pytorch' else {}
            new_centers = self._backend._backend.zeros([self.n_clusters, data.shape[-1]], **kwgs)
            length_mat = self._calcu_length_mat(data) # [N, n]
            data_group_id = length_mat.argmin(-1) # [N, ]
            # sort data to groups, set new centers to be the mean of each group
            for group_x_id in range(self.n_clusters):
                group_x_data_id = self._backend._backend.argwhere(data_group_id == group_x_id).reshape(-1) # data id for one group
                if group_x_data_id.shape[0] == 0:
                    # empty group(center), perform init for this center
                    non_null_centers = self.centers[self._backend._backend.unique(data_group_id)]
                    new_centers[group_x_id] = self._choose_center_from_data(data, non_null_centers)
                    # if there has multi null groups,
                    # need update data_group_id to update non_null_centers
                    self.centers[group_x_id] = new_centers[group_x_id]
                    data_group_id = self._calcu_length_mat(data).argmin(-1)
                else:
                    new_centers[group_x_id] = data[group_x_data_id].mean(0)
            # check if loss has no changes, no tolerance here exactly
            if self.loss_fn(data) == prev_loss:
                break
            prev_loss = self.loss
            # move centers to new centers(average of groups)
            self.centers = new_centers
            self.data_group_id = data_group_id
        # move centers to cpu if backend is pytorch
        if self.backend == 'pytorch':
            self.centers = self.centers.cpu().numpy()
            self.loss = self.loss.cpu().numpy()
        return self.centers
    
    def fit_times(self, data:np.ndarray, times:int = 3, **kwargs):
        """
        Fits the model to the given data for a specified number of times.

        Args:
            - data (np.ndarray): The input data to fit the model on.
            - times (int, optional): The number of times to fit the model. Defaults to 3.
            - backend (str, optional): The backend for calculating distance. Defaults to 'scipy'.
            - **kwargs: Additional keyword arguments to be passed to the fit method.

        Returns:
            np.ndarray: The centers of the fitted model.
        """
        self.fit(data)
        loss_records = [self.loss]
        centers_records = [self.centers]
        for _ in range(times - 1):
            self.reset()
            self.fit(data)
            loss_records.append(self.loss)
            centers_records.append(self.centers)
        min_idx = np.argmin(np.array(loss_records))
        self.loss = loss_records[min_idx]
        self.centers = centers_records[min_idx]
        return self.centers

    def predict(self, data:np.ndarray):
        """
        Predicts the class labels for the given data.

        Parameters:
            data (np.ndarray): The input data with shape [N, n].

        Returns:
            np.ndarray: The predicted class labels with shape [N, ].
        """
        length_mat = self._calcu_length_mat(data) # [N, n]
        # move centers to cpu if backend is pytorch
        if self.backend == 'pytorch':
            length_mat = length_mat.cpu().numpy()
        # return labels
        return length_mat.argmin(axis = -1) # [N, ]
        
    def fit_predict(self, data:np.ndarray, predict_data = None, fit_times = 3, **kwargs):
        """
        Fits the model to the training data and makes predictions on the given data.

        Parameters:
            data (np.ndarray): The training data.
            predict_data (Optional[np.ndarray]): The data to make predictions on. Defaults to None.
            fit_times (int): The number of times to fit the model. Defaults to 1.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: The predicted values.
        """
        if fit_times == 1:
            self.fit(data, **kwargs)
        else:
            self.fit_times(data, fit_times, **kwargs)
        return self.predict(get_default_for_None(predict_data, data))
    
class KBayesian(KMeans):
    """
    KBayesian is a subclass of KMeans that implements the Bayesian version of the K-means clustering algorithm.
    It extends the KMeans class and adds additional functionality for Bayesian optimization.

    Parameters:
    - n_clusters (int): The number of clusters to form as well as the number of centroids to generate. Default is None.
    - max_iter (int): The maximum number of iterations. Default is 200.
    - randseed (int): The random seed for reproducibility. Default is 0.
    - **kwargs: Additional keyword arguments.

    Attributes:
    - space (list): A list of lists representing the search space for Bayesian optimization.

    Methods:
    - reset(**kwargs): Reset the KBayesian instance to its initial state.
    - _init_space(data: np.ndarray): Initialize the search space for Bayesian optimization.
    - _loss_fn(obj, data: np.ndarray, centers: np.ndarray = None): Calculate the loss function for Bayesian optimization.
    - _objective(params): Define the objective function for Bayesian optimization.
    - fit(data: np.ndarray, **kwargs): Fit the KBayesian model to the data using Bayesian optimization.
    - predict(data: np.ndarray): Predict the cluster labels for the given data.
    - fit_predict(data: np.ndarray, predict_data=None, fit_times=1, **kwargs): Fit the model to the data and predict the cluster labels.

    """
    @autoparse
    def __init__(self, n_clusters: int = None, max_iter: int = 200, randseed = 0, **kwargs) -> None:
        """
        Initialize a KBayesian instance.

        Parameters:
        - n_clusters (int): The number of clusters to form as well as the number of centroids to generate. Default is None.
        - max_iter (int): The maximum number of iterations. Default is 200.
        - randseed (int): The random seed for reproducibility. Default is 0.
        - **kwargs: Additional keyword arguments.
        """
        super().__init__(n_clusters, 0, max_iter, '', **kwargs)
        self.space = [[] for _ in range(self.n_clusters)]
        
    def reset(self, **kwargs):
        super().reset(**kwargs)
        self.space = [[] for _ in range(self.n_clusters)]
        
    def _init_space(self, data: np.ndarray):
        digits = get_num_digits(self.n_clusters * data.shape[-1]) + 1
        for n_i in range(self.n_clusters):
            for dim_i in range(data.shape[-1]):
                idx = f'{n_i*data.shape[-1]+dim_i:>{digits}}'
                self.space[n_i].append(hp.uniform(idx,
                                                  data[:, dim_i].min(),
                                                  data[:, dim_i].max()))
        return self.space
    
    def _gather_space(self, shape:List[int], space:Dict[str, float]):
        return np.array(list(space.values())).reshape(shape)
        
    @staticmethod
    def _loss_fn(obj, data:np.ndarray, centers:np.ndarray = None):
        obj.loss = obj.loss_fn(data, centers)
        return obj.loss
    
    @staticmethod
    def _objective(params):
        """
        Define the objective function for Bayesian optimization.

        Parameters:
        - params: A dictionary containing the parameters.
            - obj (KBayesian): The KBayesian model.
            - space (np.ndarray): The initial centers.
            - data (np.ndarray): The input data.

        Returns:
        - result: A dictionary containing the loss value, status, and other information.

        """
        obj, space, data = params['obj'], params['space'], params['data']
        obj.centers = np.array(space)
        obj.loss = obj._loss_fn(obj, data, obj.centers)
        return {'loss': obj.loss,
                'status': STATUS_FAIL if np.isnan(obj.loss) else STATUS_OK,
                'other_stuff': {'centers': obj.centers}}
    
    def fit(self, data:np.ndarray, **kwargs):
        """
        Fit the model to the given data using Bayesian optimization.

        Parameters:
            - data (np.ndarray): The input data for fitting the model.
            - **kwargs: Additional keyword arguments.
                - verbose (bool): Whether to print the progress of the optimization. Default is False.
                - max_evals (int): The maximum number of evaluations. Default is self.max_iter.

        Returns:
            np.ndarray: An array containing the best values found by the optimization.
        """
        kwargs = set_default_kwargs(kwargs, True, verbose=False, max_evals=self.max_iter)
        trials = Trials()
        best = fmin(self._objective,
                    space={'obj': self, 'space': self._init_space(data), 'data': data},
                    algo=tpe.suggest,
                    trials=trials,
                    rstate= np.random.default_rng(self.randseed),
                    **kwargs)
        self.centers = self._gather_space([self.n_clusters, data.shape[-1]], best)
        self.loss = self.loss_fn(data, self.centers)
        return np.array(list(best.values()))
    
    def predict(self, data: np.ndarray):
        """
        Predict the cluster labels for the given data.

        Parameters:
        - data (np.ndarray): The input data.

        Returns:
        - labels (np.ndarray): The predicted cluster labels.

        """
        return super().predict(data)
    
    def fit_predict(self, data: np.ndarray, predict_data=None, fit_times=1, **kwargs):
        """
        Fit the model to the data and predict the cluster labels.

        Parameters:
        - data (np.ndarray): The input data.
        - predict_data: The data to predict. Default is None.
        - fit_times (int): The number of times to fit the model. Default is 1.
        - **kwargs: Additional keyword arguments.

        Returns:
        - labels (np.ndarray): The predicted cluster labels.

        """
        return super().fit_predict(data, predict_data, fit_times, **kwargs)
    
    
cluster_support_methods = ['DBSCAN', 'Birch', 'KMeans', 'MiniBatchKMeans',
                           'MeanShift', 'GaussianMixture', 'AgglomerativeClustering',
                           'AffinityPropagation', 'BAKMeans', 'KBayesian']
    
def cluster(data, n_clusters:int, method:str, norm = None, norm_dim = None, **kwargs):
    """
    Clusters data using various clustering methods.

    Parameters:
        - data (array-like): The input data to be clustered.
        - n_clusters (int): The number of clusters to create.
        - method (str): The clustering method to use, one of 
            ['DBSCAN', 'Birch', 'KMeans', 'MiniBatchKMeans', 'MeanShift',
            'GaussianMixture', 'AgglomerativeClustering', 'AffinityPropagation',
            'BAKMeans', 'KBayesian'].
        - norm (str, optional): The normalization method to use. Defaults to None.
        - norm_dim (int, optional): The dimension to normalize over. Defaults to None.
        - **kwargs: Additional keyword arguments specific to each clustering method.

    Returns:
        - labels (np.ndarray): The cluster labels.
        - centers (np.ndarray or None): The cluster centers. if is not supported, None will be returned.
        - loss (float): The loss value. if is not supported, -1 will be returned.
        
    Notes:
        - Kmeans: 此算法尝试最小化群集内数据点的方差。K 均值最适合用于较小的数据集，因为它遍历所有数据点。
                这意味着，如果数据集中有大量数据点，则需要更多时间来对数据点进行分类。
        - GaussianMixture: 高斯混合模型使用多个高斯分布来拟合任意形状的数据。
                在这个混合模型中，有几个单一的高斯模型充当隐藏层。因此，
                该模型计算数据点属于特定高斯分布的概率, 即它将属于的聚类。
        - KBayesian: KBayesian是一种以KMeans为框架的聚类算法, 但其将移动聚类中心的方法改为由Bayesian优化驱动.
    """
    # TODO : imp norm_dim
    if norm_dim is not None:
        raise NotImplementedError
    if norm is not None:
        if norm == 'div_max':
            data = data/data.max()
            
    if method == 'DBSCAN':
        kwargs = set_default_kwargs(kwargs, eps = 0.5, min_samples = 3)
        return DBSCAN(**kwargs).fit_predict(data), None, -1
    elif method == 'Birch':
        model = Birch(n_clusters=n_clusters, **kwargs)
        return model.fit_predict(data), model.subcluster_centers_, -1
    elif method == 'KMeans':
        model = sk_KMeans(n_clusters=n_clusters)
        return model.fit_predict(data), model.cluster_centers_, -1
    elif method == 'MiniBatchKMeans':
        model = MiniBatchKMeans(n_clusters=n_clusters)
        return model.fit_predict(data), model.cluster_centers_, -1
    elif method == 'MeanShift':
        model = MeanShift()
        return model.fit_predict(data), model.cluster_centers_, -1
    elif method == 'GaussianMixture':
        kwargs = set_default_kwargs(kwargs, n_components=n_clusters, random_state = 777)
        model = GaussianMixture(**kwargs)
        return model.fit_predict(data), model.means_, -1
    elif method == 'AgglomerativeClustering':
        model = AgglomerativeClustering(n_clusters = n_clusters)
        return model.fit(data).labels_, None, -1
    elif method == 'AffinityPropagation':
        model = AffinityPropagation(**kwargs)
        return model.fit_predict(data), None, -1
    elif method == 'BAKMeans':
        model = KMeans(n_clusters=n_clusters, **kwargs)
        return model.fit_predict(data, **kwargs), model.centers, model.loss
    elif method == 'KBayesian':
        model = KBayesian(n_clusters=n_clusters, **kwargs)
        return model.fit_predict(data, **kwargs), model.centers, model.loss
    else:
        return put_err(f'Unknown method {method}, return None', None, 1)


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
    
    r = cluster(torch.tensor(X, device = 'gpu'), n_classes, 'BAKMeans', backend='pytorch')

    fig, axs = plt.subplots(2, 5, figsize = (10, 10))
    for i in range(2):
        for j in range(5):
            method = cluster_support_methods[i*5+j]
            yhat, center, loss = cluster(X, n_classes, method, 'div_max')
            if center is not None and center.shape[0] >= 2:
                center_pos = pca(center, 2)
            else:
                center_pos = None
            # 检索唯一群集
            clusters_id = np.unique(yhat)
            # 为每个群集的样本创建散点图
            for cluster_id in clusters_id:
                # 获取此群集的示例的行索引
                row_ix = np.where(yhat == cluster_id)
                # 创建这些样本的散布
                axs[i][j].scatter(pos[row_ix, 0], pos[row_ix, 1])
            if center_pos is not None:
                # plot centers
                axs[i][j].scatter(center_pos[:, 0], center_pos[:, 1])
            if loss is not None:
                axs[i][j].text(0, 0, f'loss: {loss:.4f}')
            axs[i][j].set_title(method)
    # 绘制散点图
    plt.show()