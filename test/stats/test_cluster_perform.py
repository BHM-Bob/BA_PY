
import torch
from sklearn.datasets import make_classification

from mbapy.base import TimeCosts
from mbapy.stats import pca
from mbapy.stats.cluster import cluster, cluster_support_methods

n_classes = 4
X, _ = make_classification(n_samples=10000*n_classes, n_features=512, n_classes=n_classes,
                           n_clusters_per_class=1, random_state=4)
pos = pca(X, 2)
print(X.shape, pos.shape)

@TimeCosts(5, True)
def func(times, data, n_classes, method, **kwargs):
    labels, centers, loss = cluster(data, n_classes, method, 'div_max', **kwargs)
    print(loss)

# for method in ['KMeans', 'MiniBatchKMeans']:
#     print(method)
#     func(X, n_classes, method)
        
print('BAKMeans scipy')
func(X, n_classes, 'BAKMeans', mini_batch = 1, backend = 'scipy')
        
print('BAKMeans pytorch cpu')
X = torch.tensor(X, device = 'cpu')
func(X, n_classes, 'BAKMeans', mini_batch = 0.5, backend = 'pytorch')

print('BAKMeans pytorch cuda')
X = torch.tensor(X, device = 'cuda')
func(X, n_classes, 'BAKMeans', mini_batch = 0.5, backend = 'pytorch')