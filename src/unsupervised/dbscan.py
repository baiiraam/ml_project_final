from typing import Any
import numpy as np
import numpy.typing as npt

class DBSCAN:
    def __init__(self, eps: float, min_samples: int) -> None:
        self.eps = eps
        self.min_samples = min_samples
        self.labels_: npt.NDArray[np.integer[Any]] | None = None
        self.core_sample_indices_: npt.NDArray[np.integer[Any]] | None = None
        self.n_clusters_: int | None = None
        self.n_features_in_: int | None = None
        
    def fit(self, X: np.ndarray) -> "DBSCAN":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.min_samples <= 0:
            raise ValueError("min_samples must be positive")
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        labels = np.full(n_samples, -1, dtype=int)
        visited = np.zeros(n_samples, dtype=bool)
        is_core = np.zeros(n_samples, dtype=bool)
        cluster_id = 0
        for i in range(n_samples):
            if visited[i]:
                continue
            visited[i] = True
            neighbors = self._region_query(X, i)
            if len(neighbors) < self.min_samples:
                labels[i] = -1
                continue
            is_core[i] = True
            labels[i] = cluster_id
            self._expand_cluster(
                X=X,
                labels=labels,
                visited=visited,
                is_core=is_core,
                neighbors=neighbors,
                cluster_id=cluster_id,
            )
            cluster_id += 1

        self.labels_ = labels
        self.core_sample_indices_ = np.where(is_core)[0]
        self.n_clusters_ = cluster_id
        return self

    def _region_query(self, X: np.ndarray, index: int) -> list[int]:
        distances = np.linalg.norm(X - X[index], axis=1)
        neighbors = np.where(distances <= self.eps)[0]
        return [int(i) for i in neighbors]
        
    def _expand_cluster(
        self,
        X: np.ndarray,
        labels: npt.NDArray[np.integer[Any]],
        visited: npt.NDArray[np.bool_],
        is_core: npt.NDArray[np.bool_],
        neighbors: list[int],
        cluster_id: int,
    ) -> None:
        p = 0
        neighbor_set = set(neighbors)
        while p < len(neighbors):
            j = neighbors[p]
            if not visited[j]:
                visited[j] = True
                new_neighbors = self._region_query(X, j)
                if len(new_neighbors) >= self.min_samples:
                    is_core[j] = True
                    for point in new_neighbors:
                        if point not in neighbor_set:
                            neighbors.append(point)
                            neighbor_set.add(point)
            if labels[j] == -1:
                labels[j] = cluster_id
            p += 1

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        assert self.labels_ is not None
        return self.labels_
