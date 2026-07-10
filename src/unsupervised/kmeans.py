import numpy as np
class KMeans:
    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int | None = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.centroids_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float | None = None
        self.n_iter_: int | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: np.ndarray) -> "KMeans":
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        n_samples, n_features = X.shape

        if self.n_clusters < 1:
            raise ValueError("n_clusters must be at least 1")

        if self.n_clusters > n_samples:
            raise ValueError("n_clusters cannot be greater than number of samples")

        if self.max_iter < 1:
            raise ValueError("max_iter must be at least 1")

        if self.tol < 0:
            raise ValueError("tol must be non-negative")

        self.n_features_in_ = n_features

        rng = np.random.default_rng(self.random_state)

        indices = rng.choice(n_samples, self.n_clusters, replace=False)
        centroids = X[indices].copy()

        iteration = 0

        for iteration in range(self.max_iter):
            distances = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            labels = np.argmin(distances, axis=1)

            new_centroids = centroids.copy()

            for k in range(self.n_clusters):
                points = X[labels == k]

                if len(points) == 0:
                    random_index = rng.integers(0, n_samples)
                    new_centroids[k] = X[random_index]
                else:
                    new_centroids[k] = points.mean(axis=0)

            shift = np.linalg.norm(new_centroids - centroids)

            centroids = new_centroids

            if shift <= self.tol:
                break

        distances = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = np.argmin(distances, axis=1)

        self.centroids_ = centroids
        self.labels_ = labels
        self.inertia_ = float(np.sum((X - centroids[labels]) ** 2))
        self.n_iter_ = iteration + 1

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.centroids_ is None:
            raise ValueError("fit first")

        if self.n_features_in_ is None:
            raise ValueError("fit first")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has wrong number of features")

        centroids = self.centroids_

        distances = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        return np.argmin(distances, axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)

        assert self.labels_ is not None

        return self.labels_
