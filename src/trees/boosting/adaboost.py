# from __future__ import annotations
# from typing import Iterator, Optional
# import numpy as np

# from src.trees.decision_tree import DecisionTree

# class DecisionStump(DecisionTree):
#     def __init__(
#         self,
#         criterion: str = "gini",
#         random_state: Optional[int] = None,
#     ) -> None:
#         super().__init__(
#             max_depth=1,
#             min_samples_split=2,
#             criterion=criterion,
#             max_features=None,
#             random_state=random_state,
#         )


# class AdaBoostClassifier:
#     def __init__(
#         self,
#         n_estimators: int = 50,
#         learning_rate: float = 1.0,
#         criterion: str = "gini",
#         random_state: Optional[int] = None,
#     ) -> None:
#         self.n_estimators = n_estimators
#         self.learning_rate = learning_rate
#         self.criterion = criterion
#         self.random_state = random_state
#         self.classes_: np.ndarray | None = None
#         self.estimators_: list[DecisionStump] = []
#         self.estimator_weights_: np.ndarray = np.array([], dtype=np.float64)
#         self.estimator_errors_: np.ndarray = np.array([], dtype=np.float64)

#     def fit(self, X: np.ndarray, y: np.ndarray) -> AdaBoostClassifier:
#         if X.shape[0] != y.shape[0]:
#             raise ValueError("X and y must have the same number of samples.")

#         self.classes_ = np.unique(y)
#         n_classes = len(self.classes_)
#         n_features = X.shape[1]
#         n = X.shape[0]
#         w = np.full(n, 1.0 / n)

#         alphas: list[float] = []
#         errors: list[float] = []

#         # Adaptive threshold for multiclass
#         if n_classes > 2:
#             base_threshold = 1.0 - 1.0 / n_classes
#             if n_features > 100:
#                 error_threshold = 0.95
#             else:
#                 error_threshold = base_threshold + 0.01
#         else:
#             error_threshold = 0.5

#         for m in range(self.n_estimators):
#             rng: Optional[int] = None
#             if self.random_state is not None:
#                 rng = self.random_state + m

#             stump = DecisionStump(criterion=self.criterion, random_state=rng)
#             stump.fit(X, y, sample_weight=w)
#             pred = stump.predict(X)
#             incorrect = (pred != y).astype(float)
#             err = np.sum(w * incorrect) / np.sum(w)
#             err = max(err, 1e-10)

#             if err >= error_threshold:
#                 break

#             alpha = np.log((1.0 - err) / err)
#             if n_classes > 2:
#                 alpha += np.log(n_classes - 1)
#             alpha *= self.learning_rate

#             w = w * np.exp(alpha * incorrect)
#             w = w / w.sum()

#             self.estimators_.append(stump)
#             alphas.append(alpha)
#             errors.append(err)

#         self.estimator_weights_ = np.array(alphas, dtype=np.float64)
#         self.estimator_errors_ = np.array(errors, dtype=np.float64)
#         return self

#     def predict(self, X: np.ndarray) -> np.ndarray:
#         if not self.estimators_:
#             raise ValueError("The model has not been fitted yet.")
#         assert self.classes_ is not None
#         votes = np.zeros((X.shape[0], len(self.classes_)))
#         for alpha, stump in zip(self.estimator_weights_, self.estimators_):
#             pred = stump.predict(X)
#             for i, c in enumerate(self.classes_):
#                 votes[:, i] += alpha * (pred == c).astype(float)
#         return self.classes_[np.argmax(votes, axis=1)]

#     def predict_proba(self, X: np.ndarray) -> np.ndarray:
#         if not self.estimators_:
#             raise ValueError("The model has not been fitted yet.")
#         assert self.classes_ is not None
#         votes = np.zeros((X.shape[0], len(self.classes_)))
#         for alpha, stump in zip(self.estimator_weights_, self.estimators_):
#             pred = stump.predict(X)
#             for i, c in enumerate(self.classes_):
#                 votes[:, i] += alpha * (pred == c).astype(float)
#         exp_votes = np.exp(votes - votes.max(axis=1, keepdims=True))
#         return exp_votes / exp_votes.sum(axis=1, keepdims=True)

#     @property
#     def estimator_weights(self) -> np.ndarray:
#         return self.estimator_weights_

#     @property
#     def estimator_errors(self) -> np.ndarray:
#         return self.estimator_errors_

#     def staged_predict(self, X: np.ndarray) -> Iterator[np.ndarray]:
#         if not self.estimators_:
#             raise ValueError("The model has not been fitted yet.")
#         assert self.classes_ is not None
#         votes = np.zeros((X.shape[0], len(self.classes_)))
#         for alpha, stump in zip(self.estimator_weights_, self.estimators_):
#             pred = stump.predict(X)
#             for i, c in enumerate(self.classes_):
#                 votes[:, i] += alpha * (pred == c).astype(float)
#             yield self.classes_[np.argmax(votes, axis=1)]
















from __future__ import annotations
from typing import Iterator, Optional
import numpy as np

from src.trees.decision_tree import DecisionTree


class DecisionStump(DecisionTree):
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

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.random_state = random_state
        self.classes_: np.ndarray | None = None
        self.estimators_: list[DecisionStump] = []
        self.estimator_weights_: np.ndarray = np.array([], dtype=np.float64)
        self.estimator_errors_: np.ndarray = np.array([], dtype=np.float64)

    def fit(self, X: np.ndarray, y: np.ndarray) -> AdaBoostClassifier:
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("AdaBoostClassifier supports binary classification only.")

        self.estimators_ = []
        self.estimator_weights_ = np.array([], dtype=np.float64)
        self.estimator_errors_ = np.array([], dtype=np.float64)

        n = X.shape[0]
        w = np.full(n, 1.0 / n)
        alphas: list[float] = []
        errors: list[float] = []

        for m in range(self.n_estimators):
            rng: Optional[int] = None
            if self.random_state is not None:
                rng = self.random_state + m

            stump = DecisionStump(
                criterion=self.criterion,
                random_state=rng,
            )
            stump.fit(X, y, sample_weight=w)

            pred = stump.predict(X)
            incorrect = (pred != y).astype(float)
            err = np.sum(w * incorrect) / np.sum(w)
            err = max(float(err), 1e-10)

            if err >= 0.5:
                break

            alpha = np.log((1.0 - err) / err)
            alpha *= self.learning_rate

            self.estimators_.append(stump)
            alphas.append(alpha)
            errors.append(err)

            if err <= 1e-10:
                break

            w = w * np.exp(alpha * incorrect)
            w = w / w.sum()

        self.estimator_weights_ = np.array(alphas, dtype=np.float64)
        self.estimator_errors_ = np.array(errors, dtype=np.float64)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.estimators_:
            raise ValueError("The model has not been fitted yet.")

        assert self.classes_ is not None
        votes = np.zeros((X.shape[0], len(self.classes_)))

        for alpha, stump in zip(self.estimator_weights_, self.estimators_):
            pred = stump.predict(X)
            for i, c in enumerate(self.classes_):
                votes[:, i] += alpha * (pred == c).astype(float)

        return self.classes_[np.argmax(votes, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.estimators_:
            raise ValueError("The model has not been fitted yet.")

        assert self.classes_ is not None
        votes = np.zeros((X.shape[0], len(self.classes_)))

        for alpha, stump in zip(self.estimator_weights_, self.estimators_):
            pred = stump.predict(X)
            for i, c in enumerate(self.classes_):
                votes[:, i] += alpha * (pred == c).astype(float)

        exp_votes = np.exp(votes - votes.max(axis=1, keepdims=True))
        return exp_votes / exp_votes.sum(axis=1, keepdims=True)

    @property
    def estimator_weights(self) -> np.ndarray:
        return self.estimator_weights_

    @property
    def estimator_errors(self) -> np.ndarray:
        return self.estimator_errors_

    def staged_predict(self, X: np.ndarray) -> Iterator[np.ndarray]:
        if not self.estimators_:
            raise ValueError("The model has not been fitted yet.")

        assert self.classes_ is not None
        votes = np.zeros((X.shape[0], len(self.classes_)))

        for alpha, stump in zip(self.estimator_weights_, self.estimators_):
            pred = stump.predict(X)
            for i, c in enumerate(self.classes_):
                votes[:, i] += alpha * (pred == c).astype(float)

            yield self.classes_[np.argmax(votes, axis=1)]