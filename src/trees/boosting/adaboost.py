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

        self.estimator_weights_: np.ndarray = np.array(
            [],
            dtype=np.float64,
        )

        self.estimator_errors_: np.ndarray = np.array(
            [],
            dtype=np.float64,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> AdaBoostClassifier:
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        self.classes_ = np.unique(y)

        if len(self.classes_) < 2:
            raise ValueError("AdaBoostClassifier requires at least two classes.")

        self.estimators_ = []

        self.estimator_weights_ = np.array(
            [],
            dtype=np.float64,
        )

        self.estimator_errors_ = np.array(
            [],
            dtype=np.float64,
        )

        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        sample_weights = np.full(
            n_samples,
            1.0 / n_samples,
        )

        alphas: list[float] = []
        errors: list[float] = []

        max_error = 1.0 - (1.0 / n_classes)

        for estimator_index in range(self.n_estimators):
            seed: Optional[int] = None

            if self.random_state is not None:
                seed = self.random_state + estimator_index

            stump = DecisionStump(
                criterion=self.criterion,
                random_state=seed,
            )

            stump.fit(
                X,
                y,
                sample_weight=sample_weights,
            )

            predictions = stump.predict(X)
            incorrect = predictions != y

            error = np.sum(sample_weights * incorrect.astype(float))

            error = max(
                float(error),
                1e-10,
            )

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

        self.estimator_weights_ = np.array(
            alphas,
            dtype=np.float64,
        )

        self.estimator_errors_ = np.array(
            errors,
            dtype=np.float64,
        )

        return self

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        if not self.estimators_:
            raise ValueError("The model has not been fitted yet.")

        assert self.classes_ is not None

        votes = np.zeros((X.shape[0], len(self.classes_)))

        for alpha, stump in zip(
            self.estimator_weights_,
            self.estimators_,
        ):
            predictions = stump.predict(X)

            for class_index, class_label in enumerate(self.classes_):
                votes[:, class_index] += alpha * (predictions == class_label).astype(
                    float
                )

        return self.classes_[np.argmax(votes, axis=1)]

    def predict_proba(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        if not self.estimators_:
            raise ValueError("The model has not been fitted yet.")

        assert self.classes_ is not None

        votes = np.zeros((X.shape[0], len(self.classes_)))

        for alpha, stump in zip(
            self.estimator_weights_,
            self.estimators_,
        ):
            predictions = stump.predict(X)

            for class_index, class_label in enumerate(self.classes_):
                votes[:, class_index] += alpha * (predictions == class_label).astype(
                    float
                )

        exp_votes = np.exp(votes - votes.max(axis=1, keepdims=True))

        return exp_votes / exp_votes.sum(
            axis=1,
            keepdims=True,
        )

    @property
    def estimator_weights(self) -> np.ndarray:
        return self.estimator_weights_

    @property
    def estimator_errors(self) -> np.ndarray:
        return self.estimator_errors_

    def staged_predict(
        self,
        X: np.ndarray,
    ) -> Iterator[np.ndarray]:
        if not self.estimators_:
            raise ValueError("The model has not been fitted yet.")

        assert self.classes_ is not None

        votes = np.zeros((X.shape[0], len(self.classes_)))

        for alpha, stump in zip(
            self.estimator_weights_,
            self.estimators_,
        ):
            predictions = stump.predict(X)

            for class_index, class_label in enumerate(self.classes_):
                votes[:, class_index] += alpha * (predictions == class_label).astype(
                    float
                )

            yield self.classes_[np.argmax(votes, axis=1)]
