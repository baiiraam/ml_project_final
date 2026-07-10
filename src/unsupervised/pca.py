import numpy as np
from numpy.typing import NDArray

class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components

        self.components_: NDArray[np.float64] | None = None
        self.explained_variance_: NDArray[np.float64] | None = None
        self.explained_variance_ratio_: NDArray[np.float64] | None = None
        self.mean_: NDArray[np.float64] | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: NDArray[np.float64]) -> "PCA":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        n_samples, n_features = X.shape
        if n_samples < 2:
            raise ValueError("PCA needs at least 2 samples")
        if self.n_components < 1 or self.n_components > n_features:
            raise ValueError("wrong n_components")
        self.n_features_in_ = n_features
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_
        cov = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[indices]
        eigenvectors = eigenvectors[:, indices]
        eigenvalues = np.maximum(eigenvalues, 0.0)
        self.components_ = eigenvectors[:, : self.n_components].T.astype(np.float64)
        self.explained_variance_ = eigenvalues[: self.n_components].astype(np.float64)

        total_variance = float(eigenvalues.sum())

        if total_variance == 0.0:
            self.explained_variance_ratio_ = np.zeros(
                self.n_components,
                dtype=np.float64,
            )
        else:
            self.explained_variance_ratio_ = (
                self.explained_variance_ / total_variance
            )

        return self

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if (
            self.components_ is None
            or self.mean_ is None
            or self.n_features_in_ is None
        ):
            raise ValueError("fit first")

        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has wrong number of features")

        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(
        self,
        X_transformed: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if self.components_ is None or self.mean_ is None:
            raise ValueError("fit first")

        X_transformed = np.asarray(X_transformed, dtype=np.float64)

        if X_transformed.ndim != 2:
            raise ValueError("X_transformed must be a 2D array")

        if X_transformed.shape[1] != self.n_components:
            raise ValueError("X_transformed has wrong number of components")

        return X_transformed @ self.components_ + self.mean_
