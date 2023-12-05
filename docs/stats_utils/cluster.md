# mbapy.stats.cluster

### KMeans
**Description**

KMeans clustering algorithm implementation.

#### Attributes
- space (list): A list of lists representing the search space for Bayesian optimization.
- centers (np.ndarray): The final cluster centers.

#### Methods
- reset: Reset the KMeans instance to its initial state.
- loss_fn: Calculate the loss function for the given data and centers.
- fit: Fit the KMeans model to the given data.
- fit_times: Fit the model to the data multiple times and predict the cluster labels, return the best one.
- fit_predict: Fit the model to the data and predict the cluster labels.
- predict: Predict the cluster labels for the given data.

#### Notes
- KMeans is suitable for smaller datasets as it iterates through all data points to minimize the variance within clusters.

#### Example
```python
# Initialize KMeans model
kmeans = KMeans(n_clusters=3)
# Fit the model to the data
kmeans.fit(data)
# Predict the cluster labels for new data
labels = kmeans.predict(new_data)
```

### KBayesian
**Description**

KBayesian is a subclass of KMeans that implements the Bayesian version of the K-means clustering algorithm. It extends the KMeans class and adds additional functionality for Bayesian optimization.

#### Attributes
- space (list): A list of lists representing the search space for Bayesian optimization.
- centers (np.ndarray): The final cluster centers.

#### Methods
- reset: Reset the KBayesian instance to its initial state.
- _init_space: Initialize the search space for Bayesian optimization.
- _loss_fn: Calculate the loss function for Bayesian optimization.
- _objective: Define the objective function for Bayesian optimization.
- fit: Fit the KBayesian model to the data using Bayesian optimization.
- predict: Predict the cluster labels for the given data.
- fit_predict: Fit the model to the data and predict the cluster labels.

#### Notes
- KBayesian uses Bayesian optimization to move cluster centers.

#### Example
```python
# Initialize KBayesian model
kbayesian = KBayesian(n_clusters=3)
# Fit the model to the data using Bayesian optimization
kbayesian.fit(data)
# Predict the cluster labels for new data
labels = kbayesian.predict(new_data)
```

### KOptim
**Description**

KOptim is a subclass of KMeans that implements the gradient optimization version of the K-means clustering algorithm.

#### Attributes
- centers (np.ndarray): The final cluster centers.
- loss (np.ndarray): The loss value.

#### Methods
- fit: Fit the model to the given data.
- fit_times: Fit the model to the given data for a specified number of times.

#### Notes
- KOptim uses gradient optimization to determine cluster centers.

#### Example
```python
# Initialize KOptim model
koptim = KOptim(n_clusters=3)
# Fit the model to the data
koptim.fit(data)
```

### cluster
**Description**

Clusters data using various clustering methods.

#### Parameters
- data (array-like): The input data to be clustered.
- n_clusters (int): The number of clusters to create.
- method (str): The clustering method to use, one of ['DBSCAN', 'Birch', 'KMeans', 'MiniBatchKMeans', 'MeanShift', 'GaussianMixture', 'AgglomerativeClustering', 'AffinityPropagation', 'BAKMeans', 'KBayesian', 'KOptim'].
- norm (str, optional): The normalization method to use. Defaults to None.
- norm_dim (int, optional): The dimension to normalize over. Defaults to None.
- copy_norm (bool, optional): Whether to copy the data before normalizing. Defaults to True.
- **kwargs: Additional keyword arguments specific to each clustering method.

#### Returns
- labels (np.ndarray): The cluster labels.
- centers (np.ndarray or None): The cluster centers. if is not supported, None will be returned.
- loss (float): The loss value. if is not supported, -1 will be returned.

#### Notes
- This function provides a unified interface for clustering data using different clustering methods.

#### Example
```python
# Cluster the data using KMeans
labels, centers, loss = cluster(data, n_clusters=3, method='KMeans')
```