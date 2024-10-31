import copy
from typing import Dict, List

import numpy as np
import scipy
from sklearn.cluster import (DBSCAN, AffinityPropagation,
                             AgglomerativeClustering, Birch)
from sklearn.cluster import KMeans as sk_KMeans
from sklearn.cluster import MeanShift, MiniBatchKMeans
from sklearn.mixture import GaussianMixture

if __name__ == '__main__':
    # dev mode
    from mbapy_lite.base import (autoparse, get_default_args,
                                 get_default_call_for_None,
                                 get_default_for_None, get_num_digits, put_err,
                                 set_default_kwargs)
else:
    # release mode
    from ..base import (autoparse, get_default_args, get_default_call_for_None,
                        get_default_for_None, get_num_digits, put_err,
                        set_default_kwargs)
    

class KMeansBackend:
    def __init__(self, backend:str) -> None:
        self.backend = backend
        if backend == 'scipy':
            self._backend = np
            self.array = np.array
    def cat(self, *args, **kwargs):
        if self.backend == 'scipy':
            return np.concatenate(*args, **kwargs, axis = 0)
    def cdist(self, data, centers):
        if self.backend == 'scipy' or isinstance(data, np.ndarray):
            return scipy.spatial.distance.cdist(data, centers, metric = 'euclidean')
    def random_choice(self, n:int, p):
        if self.backend == 'scipy':
            return np.random.choice(n, p = p)
    def sample(self, data, mini_batch:float):
        if self.backend == 'scipy':
            idxs = np.random.permutation(np.arange(data.shape[0]))[:int(data.shape[0]*mini_batch)]
        return data[idxs, :]
        

class KMeans:
    """ 
    KMeans clustering algorithm implementation.

    Attributes:
    - centers (np.ndarray): The final cluster centers.
    
    Methods:
    - reset: Reset the KMeans instance to its initial state.
    - loss_fn: Calculate the loss function for the given data and centers.
    - fit: Fit the KMeans model to the given data.
    - fit_times: Fit the model to the data multiple times and predict the cluster labels, return the best one.
    - fit_predict: Fit the model to the data and predict the cluster labels.
    - predict: Predict the cluster labels for the given data.
    """
        
    @autoparse
    def __init__(self, n_clusters: int = None, tolerance: float = 0.0001, max_iter: int = 200,
                 mini_batch: float = 1., init_method: str = 'prob', backend: str = 'scipy', **kwargs) -> None:
        """
        Parameters:
            - n_clusters(int): number of clusters, if set to None, will auto search from 1 to sum of data to make fit for tolerance.
            - tolerance(float): tolerance for convergence, default is 0.0001.
            - max_iter(int): maximum number of iterations, default is 200.
            - mini_batch(float): mini batch size ratio, default is 1.0.
            - init_method(str): initialization method, either 'prob' or 'max', default is 'prob'.
            - backend (str): The backend for calculating distance. Default is 'scipy'.
                - 'scipy': mainly because scipy.spatial.distance.cdist, it use numpy.NDArray.
                - 'pytorch': mainly because torch.cdist, it use torch.Tensor.
            - **kwargs: additional keyword arguments.
        """
        self.centers = None # should be a ndarray with shape [n, *, D]
        self.data_group_id = None
        self.loss_record = []
        
        if isinstance(backend, str):
            self._backend = KMeansBackend(backend)
        elif isinstance(backend, KMeansBackend):
            self._backend = backend
        else:
            raise ValueError(f"backend should be str or a class of KMeansBackend class, but got {backend}")
        
    def reset(self, **kwargs):
        """
        Reset the centers and data group id to None.
        """
        self.centers = None
        self.data_group_id = None
        self.loss_record = []
        
    def _calcu_length_mat(self, data:np.ndarray, centers:np.ndarray = None):
        """
        Calculate the length matrix between data and centers.

        Parameters:
            - data(ndarray): data to be clustered, should be a ndarray with shape [N, *, D].
            - centers(ndarray): centers of the clusters, should be a ndarray with shape [n, *, D], default is None.

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
        Initializes the cluster centers for the KMeans algorithm.
            **If self.centers is not None, do nothing for flexibility**.

        Parameters:
            data (np.ndarray): The input data used to determine the cluster centers.

        Returns:
            np.ndarray: The initialized cluster centers.
        """
        # skip if self.centers is not None, to make flexibility
        if self.centers is None:            
            # choose the first center randomly
            first_center_idx = int(np.random.uniform(0, data.shape[0]))
            self.centers = data[first_center_idx][None, ] # [n=1, *, D]
            # generate left centers by kmeans++
            for _ in range(self.n_clusters - 1):
                new_center = self._choose_center_from_data(data, self.centers)[None, ]
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
    
    def _single_iter(self, data):
        """
        Sorts the data into groups based on the current centers 
        and calculates the new centers for each group.
        """
        # sort data to groups id by now centers, get data_group_id
        kwgs = {'device' : data.device} if self.backend == 'pytorch' else {}
        new_centers = self._backend._backend.zeros([self.n_clusters] + list(data.shape[1:]), **kwgs)
        length_mat = self._calcu_length_mat(data) # [N, n]
        data_group_id = length_mat.argmin(-1) # [N, ]
        # sort data to groups, set new centers to be the mean of each group
        for group_x_id in range(self.n_clusters):
            # data id for one group
            group_x_data_id = self._backend._backend.argwhere(data_group_id == group_x_id).reshape(-1)
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
        return new_centers, data_group_id        
    
    def fit(self, data:np.ndarray, reset = True, **kwargs):
        """
        Fits the K-means clustering model to the given data.

        Parameters:
            - data (np.ndarray): The input data for clustering.
            - **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: The final cluster centers.
        """
        # reset
        if reset:
            self.reset()
        # MiniBatchKMeans
        if self.mini_batch < 1:
            data = self._backend.sample(data, self.mini_batch)
        # KMeans++
        self._init_centers(data)
        prev_loss = None
        for _ in range(self.max_iter):
            new_centers, data_group_id = self._single_iter(data)
            # check if loss has no changes, no tolerance here exactly
            if self.loss_fn(data) == prev_loss:
                break
            prev_loss = self.loss
            self.loss_record.append(self.loss)
            # move centers to new centers(average of groups)
            self.centers = new_centers
            self.data_group_id = data_group_id
        # move centers to cpu if backend is pytorch
        if self.backend == 'pytorch':
            self.centers = self.centers.cpu().numpy()
            self.loss = self.loss.cpu().numpy()
            self.loss_record = list(map(lambda x: x.cpu().numpy(), self.loss_record))
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
        self.fit(data, **kwargs)
        loss_records = [self.loss]
        centers_records = [self.centers]
        for _ in range(times - 1):
            self.reset()
            self.fit(data, **kwargs)
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
    
BAKMeans = KMeans
    
    
cluster_support_methods = ['DBSCAN', 'Birch', 'KMeans', 'MiniBatchKMeans',
                           'MeanShift', 'GaussianMixture', 'AgglomerativeClustering',
                           'AffinityPropagation', 'BAKMeans']
    
def cluster(data, n_clusters:int, method:str,
            norm = None, norm_dim = None, copy_norm = True, **kwargs):
    """
    Clusters data using various clustering methods.

    Parameters:
        - data (array-like): The input data to be clustered.
        - n_clusters (int): The number of clusters to create.
        - method (str): The clustering method to use, one of 
            ['DBSCAN', 'Birch', 'KMeans', 'MiniBatchKMeans', 'MeanShift',
            'GaussianMixture', 'AgglomerativeClustering', 'AffinityPropagation',
            'BAKMeans', 'KBayesian', 'KOptim'].
        - norm (str, optional): The normalization method to use. Defaults to None.
        - norm_dim (int, optional): The dimension to normalize over. Defaults to None.
        - copy_norm (bool, optional): Whether to copy the data before normalizing. Defaults to True.
        - **kwargs: Additional keyword arguments specific to each clustering method.
            - backend (str, optional): for BAKMeans, The backend for calculating distance. Defaults to 'scipy', valid values are ['scipy', 'pytorch'].

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
        - KOptim: KOptim是一种以KMeans为框架的聚类算法, 但其将确定聚类中心的方法改为由梯度优化驱动.
    """
    # kwgs
    kwgs = get_default_args(kwargs, backend = 'scipy')
    # norm
    if copy_norm:
        _data = copy.deepcopy(data)
    else:
        _data = data
    # TODO : imp norm_dim
    if norm_dim is not None:
        raise NotImplementedError
    if norm is not None:
        if norm == 'div_max':
            norm_ratio = _data.max()
            _data /= norm_ratio
    loss = -1
    # match clustering method and do clustering
    if method == 'DBSCAN':
        kwargs = set_default_kwargs(kwargs, discard_extra=True, eps = 0.5, min_samples = 3)
        labels, centers = DBSCAN(**kwargs).fit_predict(_data), None
    elif method == 'Birch':
        model = Birch(n_clusters=n_clusters, **kwargs)
        labels, centers = model.fit_predict(_data), model.subcluster_centers_
    elif method == 'KMeans':
        model = sk_KMeans(n_clusters=n_clusters)
        labels, centers = model.fit_predict(_data), model.cluster_centers_
    elif method == 'MiniBatchKMeans':
        model = MiniBatchKMeans(n_clusters=n_clusters)
        labels, centers = model.fit_predict(_data), model.cluster_centers_
    elif method == 'MeanShift':
        model = MeanShift()
        labels, centers = model.fit_predict(_data), model.cluster_centers_
    elif method == 'GaussianMixture':
        kwargs = set_default_kwargs(kwargs, discard_extra=True, n_components=n_clusters, random_state = 777)
        model = GaussianMixture(**kwargs)
        labels, centers = model.fit_predict(_data), model.means_
    elif method == 'AgglomerativeClustering':
        model = AgglomerativeClustering(n_clusters = n_clusters)
        labels, centers = model.fit(_data).labels_, None
    elif method == 'AffinityPropagation':
        model = AffinityPropagation(**kwargs)
        labels, centers = model.fit_predict(_data), None
    elif method == 'BAKMeans':
        model = KMeans(n_clusters=n_clusters, **kwargs)
        labels, centers, loss = model.fit_predict(_data, **kwargs), model.centers, model.loss
    else:
        return put_err(f'Unknown method {method}, return None', None, 1)
    if centers is not None and loss == -1:
        loss = scipy.spatial.distance.cdist(_data, centers, metric = 'euclidean').mean()
    # re norm
    if norm is not None and centers is not None:
        if norm == 'div_max':
            if kwgs['backend'] == 'pytorch':
                norm_ratio = norm_ratio.cpu().numpy()
            centers *= norm_ratio
    return labels, centers, loss
   

if __name__ == '__main__':
    # dev code
    np.random.seed(2220)
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.datasets import make_classification
    n_classes = 4
    df = pd.read_excel(r'data/plot.xlsx',sheet_name='MWM')
    tags = [col for col in df.columns if col not in ['Unnamed: 0', 'Animal No.', 'Trial Type', 'Title', 'Start time', 'Memo', 'Day', 'Animal Type']]
    X = df.loc[ : ,tags].values
    
    # labels, centers, loss = cluster(X, n_classes, 'BAKMeans', norm = 'div_max')
    model = BAKMeans(n_classes, backend='scipy')
    
    for i in range(5):
        model.fit(X)
        plt.plot(list(range(len(model.loss_record))), model.loss_record, label = f'iter{i}')
    
    plt.show()