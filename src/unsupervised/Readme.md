# Unsupervised Learning Models

The `unsupervised` directory contains custom implementations of three widely used unsupervised machine learning algorithms:

* K-Means Clustering
* Principal Component Analysis
* DBSCAN

All implementations are written from scratch using NumPy. They follow a simple interface similar to scikit-learn, which makes them easy to use in experiment scripts and helper functions.

The main purpose of these implementations is to demonstrate the internal logic of clustering and dimensionality reduction algorithms without relying on ready-made machine learning model classes.

## Directory Structure

```text
src/
└── unsupervised/
    ├── kmeans.py
    ├── pca.py
    ├── dbscan.py
    └── README.md
```

## Files

| File        | Class    | Purpose                                                                       |
| ----------- | -------- | ----------------------------------------------------------------------------- |
| `kmeans.py` | `KMeans` | Divides samples into a predefined number of clusters.                         |
| `pca.py`    | `PCA`    | Reduces the number of features while preserving as much variance as possible. |
| `dbscan.py` | `DBSCAN` | Detects density-based clusters and identifies noise points.                   |

---

# K-Means

The `KMeans` class implements the K-Means clustering algorithm.

K-Means is a partition-based clustering algorithm. It divides the input dataset into a fixed number of clusters by minimizing the distance between samples and their assigned cluster centroids.

## Import

```python
from src.unsupervised.kmeans import KMeans
```

## Basic Usage

```python
model = KMeans(
    n_clusters=3,
    max_iter=300,
    tol=1e-4,
    random_state=42,
)

labels = model.fit_predict(X)
```

## Parameters

| Parameter      | Type          |  Default | Description                                         |
| -------------- | ------------- | -------: | --------------------------------------------------- |
| `n_clusters`   | `int`         | Required | Number of clusters to create.                       |
| `max_iter`     | `int`         |    `300` | Maximum number of centroid update iterations.       |
| `tol`          | `float`       |   `1e-4` | Convergence tolerance based on centroid movement.   |
| `random_state` | `int \| None` |   `None` | Seed used for reproducible centroid initialization. |

## Algorithm Workflow

The implementation follows these steps:

1. Convert the input into a floating-point NumPy array.
2. Validate the input shape and model parameters.
3. Randomly choose `n_clusters` training samples as initial centroids.
4. Calculate the squared Euclidean distance between every sample and centroid.
5. Assign each sample to its closest centroid.
6. Recalculate each centroid as the mean of the samples assigned to it.
7. Measure the total centroid shift.
8. Stop when the shift is less than or equal to `tol`, or when `max_iter` is reached.
9. Calculate final labels and inertia.

The distance matrix is calculated using NumPy broadcasting:

```python
distances = np.sum(
    (X[:, None, :] - centroids[None, :, :]) ** 2,
    axis=2,
)
```

## Empty Cluster Handling

A cluster may receive no samples during an iteration.

Instead of leaving the cluster without a centroid, the implementation selects a random training sample and uses it as the new centroid.

This allows the algorithm to continue training without producing invalid values.

## Methods

### `fit(X)`

Fits the model to the input data.

```python
model.fit(X)
```

Returns the fitted `KMeans` instance.

### `predict(X)`

Assigns new samples to the nearest fitted centroid.

```python
new_labels = model.predict(X_new)
```

The model must be fitted before calling `predict`.

### `fit_predict(X)`

Fits the model and directly returns the cluster labels.

```python
labels = model.fit_predict(X)
```

## Fitted Attributes

| Attribute        | Description                                                            |
| ---------------- | ---------------------------------------------------------------------- |
| `centroids_`     | Final centroid coordinates.                                            |
| `labels_`        | Cluster label assigned to each training sample.                        |
| `inertia_`       | Sum of squared distances between samples and their assigned centroids. |
| `n_iter_`        | Number of iterations completed during fitting.                         |
| `n_features_in_` | Number of features seen during training.                               |

## Inertia

The inertia value measures cluster compactness.

It is calculated as:

```text
Inertia = Σ ||xᵢ - c(labelᵢ)||²
```

A lower inertia means that samples are located closer to their assigned centroids.

However, inertia should not be used alone to compare different values of `n_clusters`, because it generally decreases when the number of clusters increases.

---

# Principal Component Analysis

The `PCA` class implements Principal Component Analysis for linear dimensionality reduction.

PCA transforms the original feature space into a smaller set of orthogonal directions called principal components.

The first principal component captures the largest possible variance, the second captures the largest remaining variance, and so on.

## Import

```python
from src.unsupervised.pca import PCA
```

## Basic Usage

```python
pca = PCA(n_components=2)

X_reduced = pca.fit_transform(X)
```

## Parameter

| Parameter      | Type  | Description                             |
| -------------- | ----- | --------------------------------------- |
| `n_components` | `int` | Number of principal components to keep. |

The value must be at least `1` and cannot exceed the number of input features.

## Algorithm Workflow

The implementation performs the following steps:

1. Convert the input to a `float64` NumPy array.
2. Validate that the input is two-dimensional.
3. Calculate the mean of each feature.
4. Center the dataset by subtracting the mean.
5. Compute the covariance matrix.
6. Apply eigenvalue decomposition using `np.linalg.eigh`.
7. Sort eigenvalues and eigenvectors in descending order.
8. Select the first `n_components` eigenvectors.
9. Store explained variance and explained variance ratios.

The centered data is calculated as:

```python
X_centered = X - self.mean_
```

The covariance matrix is calculated using:

```python
cov = np.cov(X_centered, rowvar=False)
```

## Methods

### `fit(X)`

Learns the principal components from the input dataset.

```python
pca.fit(X)
```

### `transform(X)`

Projects input samples onto the learned principal components.

```python
X_reduced = pca.transform(X)
```

The number of features must match the data used during fitting.

### `fit_transform(X)`

Fits PCA and returns the transformed data.

```python
X_reduced = pca.fit_transform(X)
```

### `inverse_transform(X_transformed)`

Reconstructs an approximation of the original data.

```python
X_reconstructed = pca.inverse_transform(X_reduced)
```

Some information is lost when fewer components than original features are retained, so the reconstructed data is usually not identical to the original data.

## Fitted Attributes

| Attribute                   | Description                                                      |
| --------------------------- | ---------------------------------------------------------------- |
| `components_`               | Principal component directions.                                  |
| `explained_variance_`       | Variance captured by each selected component.                    |
| `explained_variance_ratio_` | Fraction of total variance explained by each selected component. |
| `mean_`                     | Mean value of each original feature.                             |
| `n_features_in_`            | Number of features in the training data.                         |

## Explained Variance Ratio

The explained variance ratio shows how much of the dataset's total variance is preserved by each component.

For example:

```python
print(pca.explained_variance_ratio_)
```

Possible output:

```text
[0.62, 0.21]
```

This means that the first component explains 62% of the variance and the second explains 21%.

Together, they preserve approximately 83% of the original variance.

If the total variance is zero, the implementation returns a zero array instead of dividing by zero.

---

# DBSCAN

The `DBSCAN` class implements Density-Based Spatial Clustering of Applications with Noise.

Unlike K-Means, DBSCAN does not require the number of clusters to be specified in advance.

Instead, clusters are created from dense groups of nearby points.

## Import

```python
from src.unsupervised.dbscan import DBSCAN
```

## Basic Usage

```python
model = DBSCAN(
    eps=0.5,
    min_samples=5,
)

labels = model.fit_predict(X)
```

## Parameters

| Parameter     | Type    | Description                                               |
| ------------- | ------- | --------------------------------------------------------- |
| `eps`         | `float` | Maximum distance between neighboring samples.             |
| `min_samples` | `int`   | Minimum number of points required to form a dense region. |

Both parameters must be positive.

## Point Types

DBSCAN classifies samples into three conceptual groups.

### Core Point

A point is a core point when at least `min_samples` samples, including itself, exist within distance `eps`.

### Border Point

A border point does not have enough neighbors to become a core point, but it lies inside the neighborhood of a core point.

### Noise Point

A point that cannot be assigned to any cluster is considered noise.

Noise points receive the label:

```text
-1
```

## Algorithm Workflow

The implementation performs the following steps:

1. Initialize all labels as `-1`.
2. Track whether each sample has been visited.
3. For every unvisited sample, find all neighbors within `eps`.
4. If the neighborhood is too small, keep the sample as noise.
5. Otherwise, create a new cluster.
6. Expand the cluster through connected core points.
7. Continue until every sample has been processed.

Neighbor search is performed using Euclidean distance:

```python
distances = np.linalg.norm(X - X[index], axis=1)
```

Samples whose distance is less than or equal to `eps` are treated as neighbors.

## Cluster Expansion

The `_expand_cluster` method processes all density-connected neighbors.

When a new core point is found, its neighbors are added to the current cluster expansion queue.

A set is used to prevent the same neighbor from being repeatedly appended.

This continues until no additional density-connected points remain.

## Methods

### `fit(X)`

Runs DBSCAN clustering and stores the results.

```python
model.fit(X)
```

### `fit_predict(X)`

Fits the model and returns the cluster labels.

```python
labels = model.fit_predict(X)
```

DBSCAN does not provide a `predict` method for unseen data because assigning new points requires additional rules that are not part of the standard basic algorithm.

## Fitted Attributes

| Attribute              | Description                                     |
| ---------------------- | ----------------------------------------------- |
| `labels_`              | Cluster labels for all training samples.        |
| `core_sample_indices_` | Indices of samples identified as core points.   |
| `n_clusters_`          | Number of discovered clusters, excluding noise. |
| `n_features_in_`       | Number of input features.                       |

---

# Comparison

| Property                    | K-Means                       | PCA                      | DBSCAN               |
| --------------------------- | ----------------------------- | ------------------------ | -------------------- |
| Main task                   | Clustering                    | Dimensionality reduction | Clustering           |
| Requires number of clusters | Yes                           | No                       | No                   |
| Detects noise               | No                            | Not applicable           | Yes                  |
| Supports new data           | Yes, with `predict`           | Yes, with `transform`    | No direct prediction |
| Cluster shape               | Usually compact and spherical | Not applicable           | Arbitrary shapes     |
| Sensitive to scaling        | Yes                           | Yes                      | Yes                  |
| Random initialization       | Yes                           | No                       | No                   |

## Choosing an Algorithm

Use K-Means when:

* The approximate number of clusters is known.
* Clusters are relatively compact.
* Fast assignment of new samples is required.

Use PCA when:

* The dataset has many features.
* Visualization in two or three dimensions is needed.
* Reducing noise or redundancy is useful.
* Faster downstream model training is desired.

Use DBSCAN when:

* The number of clusters is unknown.
* Clusters may have irregular shapes.
* Detecting outliers or noise is important.
* Dense regions are separated by sparse regions.

---

# Input Requirements

All classes expect a two-dimensional NumPy-compatible array:

```text
(number of samples, number of features)
```

Example:

```python
import numpy as np

X = np.array([
    [1.0, 2.0],
    [1.2, 1.8],
    [8.0, 9.0],
    [8.5, 9.2],
])
```

It is recommended to standardize features before using these algorithms when the features have significantly different scales.

For example:

```python
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
```

---

# Validation and Errors

The implementations perform input and parameter validation.

Possible errors include:

* Input is not two-dimensional.
* `n_clusters` is less than one.
* `n_clusters` exceeds the number of samples.
* `max_iter` is less than one.
* `tol` is negative.
* `n_components` is invalid.
* PCA receives fewer than two samples.
* `eps` is not positive.
* `min_samples` is not positive.
* New data has the wrong number of features.
* A transformation or prediction method is called before fitting.

Example:

```python
model = KMeans(n_clusters=3)
model.predict(X)
```

This raises:

```text
ValueError: fit first
```

---

# Complete Example

```python
import numpy as np

from src.unsupervised.dbscan import DBSCAN
from src.unsupervised.kmeans import KMeans
from src.unsupervised.pca import PCA

X = np.array([
    [1.0, 1.0],
    [1.1, 1.2],
    [0.9, 0.8],
    [8.0, 8.0],
    [8.2, 7.9],
    [7.8, 8.1],
])

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

kmeans = KMeans(
    n_clusters=2,
    random_state=42,
)
kmeans_labels = kmeans.fit_predict(X_reduced)

dbscan = DBSCAN(
    eps=0.5,
    min_samples=2,
)
dbscan_labels = dbscan.fit_predict(X_reduced)

print("K-Means labels:", kmeans_labels)
print("DBSCAN labels:", dbscan_labels)
print("Explained variance ratio:", pca.explained_variance_ratio_)
```

---

# Dependency

The implementations depend only on NumPy.

```bash
pip install numpy
```

To install all project dependencies, use:

```bash
pip install -r requirements.txt
```

These custom implementations are mainly intended for educational analysis, controlled experiments, and comparison with established machine learning libraries.
