'''
Date: 2023-09-02 22:07:00
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-09-02 22:23:01
Description: just need run on a success
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import make_classification

from mbapy.stats import pca
from mbapy.stats.cluster import cluster, cluster_support_methods

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

r = cluster(torch.tensor(X, device = 'cuda'), n_classes, 'BAKMeans', backend='pytorch')

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