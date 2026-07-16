from __future__ import annotations
from typing import Iterator, Optional
import time
import hashlib
import numpy as np
from src.utils.dt_scale import DecisionTree  # Using fast vectorized DecisionTree


class DecisionStump(DecisionTree):
    """
    Decision stump (depth-1 decision tree) for use as a weak learner in AdaBoost.
    """

    def __init__(
        self,
        criterion: str = "gini",
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(
            max_depth=1,
            criterion=criterion,
            random_state=random_state,
        )


class AdaBoostClassifier:
    """
    AdaBoost classifier using the SAMME algorithm for multi-class classification.

    Parameters
    ----------
    n_estimators : int, default=50
        Maximum number of weak learners to train.

    learning_rate : float, default=1.0
        Learning rate (shrinkage parameter). Scales each estimator's contribution.

    criterion : str, default="gini"
        Splitting criterion for decision stumps ("gini" or "entropy").

    random_state : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        criterion: str = "gini",
        random_state: Optional[int] = None,
    ) -> None:
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'")

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.random_state = random_state

        self.classes_: Optional[np.ndarray] = None
        self.estimators_: list[DecisionStump] = []

        self.estimator_weights_: np.ndarray = np.array([], dtype=np.float64)
        self.estimator_errors_: np.ndarray = np.array([], dtype=np.float64)

        # Performance metrics
        self.fit_time_: float = 0.0
        self.predict_time_: float = 0.0

        # Secure Cache variables
        self._predictions_cache: Optional[np.ndarray] = None
        self._cache_key: Optional[str] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> AdaBoostClassifier:
        """
        Fit AdaBoost classifier to training data.
        """
        start_time = time.perf_counter()

        X = np.asarray(X)
        y = np.asarray(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        if n_classes < 2:
            raise ValueError("AdaBoostClassifier requires at least two classes.")

        n_samples = X.shape[0]

        # Initialize sample weights uniformly
        sample_weights = np.full(n_samples, 1.0 / n_samples, dtype=np.float64)

        # Reset estimators
        self.estimators_ = []
        self.estimator_weights_ = np.array([], dtype=np.float64)
        self.estimator_errors_ = np.array([], dtype=np.float64)

        alphas: list[float] = []
        errors: list[float] = []

        max_error = 1.0 - (1.0 / n_classes)

        for estimator_idx in range(self.n_estimators):
            seed = None
            if self.random_state is not None:
                seed = self.random_state + estimator_idx

            stump = DecisionStump(
                criterion=self.criterion,
                random_state=seed,
            )

            stump.fit(X, y, sample_weight=sample_weights)

            predictions = stump.predict(X)
            incorrect = predictions != y
            error = np.sum(sample_weights * incorrect.astype(float))
            error = max(float(error), 1e-10)

            if error >= max_error:
                break

            alpha = np.log((1.0 - error) / error) + np.log(n_classes - 1)
            alpha *= self.learning_rate

            self.estimators_.append(stump)
            alphas.append(alpha)
            errors.append(error)

            if error <= 1e-10:
                break

            sample_weights *= np.exp(alpha * incorrect.astype(float))
            sample_weights /= sample_weights.sum()

        self.estimator_weights_ = np.array(alphas, dtype=np.float64)
        self.estimator_errors_ = np.array(errors, dtype=np.float64)

        # Reset predictions cache when model is refitted
        self._predictions_cache = None
        self._cache_key = None

        self.fit_time_ = time.perf_counter() - start_time
        return self

    def _get_all_predictions(self, X: np.ndarray) -> np.ndarray:
        """
        Get predictions from all stumps efficiently using MD5 caching.
        """
        X = np.asarray(X)
        cache_key = hashlib.md5(X.tobytes()).hexdigest()

        if self._predictions_cache is not None and self._cache_key == cache_key:
            return self._predictions_cache

        predictions = np.array([stump.predict(X) for stump in self.estimators_])
        self._predictions_cache = predictions
        self._cache_key = cache_key

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples.
        """
        start_time = time.perf_counter()

        if not self.estimators_:
            raise ValueError("The model has not been fitted yet.")
        if self.classes_ is None:
            raise ValueError("Classes not initialized. Call fit() first.")

        X = np.asarray(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        votes = np.zeros((n_samples, n_classes), dtype=np.float64)
        predictions = self._get_all_predictions(X)

        for alpha, preds in zip(self.estimator_weights_, predictions):
            for c_idx, class_label in enumerate(self.classes_):
                votes[:, c_idx] += alpha * (preds == class_label).astype(float)

        exp_votes = np.exp(votes - np.max(votes, axis=1, keepdims=True))
        result = exp_votes / np.sum(exp_votes, axis=1, keepdims=True)

        self.predict_time_ = time.perf_counter() - start_time
        return result

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples.
        """
        if self.classes_ is None:
            raise ValueError("The model has not been fitted yet.")

        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def staged_predict(self, X: np.ndarray) -> Iterator[np.ndarray]:
        """
        Yield predictions after each boosting stage.
        """
        if not self.estimators_:
            raise ValueError("The model has not been fitted yet.")
        if self.classes_ is None:
            raise ValueError("Classes not initialized. Call fit() first.")

        X = np.asarray(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        votes = np.zeros((n_samples, n_classes), dtype=np.float64)
        predictions = self._get_all_predictions(X)

        for alpha, preds in zip(self.estimator_weights_, predictions):
            for c_idx, class_label in enumerate(self.classes_):
                votes[:, c_idx] += alpha * (preds == class_label).astype(float)

            yield self.classes_[np.argmax(votes, axis=1)]

    @property
    def estimator_weights(self) -> np.ndarray:
        return self.estimator_weights_

    @property
    def estimator_errors(self) -> np.ndarray:
        return self.estimator_errors_

    @property
    def n_features_in_(self) -> int:
        if not self.estimators_:
            raise ValueError("Model has not been fitted yet.")
        return self.estimators_[0]._n_features