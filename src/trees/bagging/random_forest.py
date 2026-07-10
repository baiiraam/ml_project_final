from __future__ import annotations

from multiprocessing import Pool
from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from src.trees.decision_tree import DecisionTree

# Constant used only to build a reproducible per-tree seed stream.
_SEED_UPPER_BOUND = 2**31 - 1

def _fit_one_tree(
    args: Tuple[
        np.ndarray,
        np.ndarray,
        Optional[int],
        int,
        Optional[Union[int, str]],
        str,
        bool,
        int,
    ],
) -> Tuple[DecisionTree, np.ndarray]:
    """
    Worker function for a single bagged tree.
    Must live at module level (not as a bound method) so that it can be
    pickled and sent to worker processes when n_jobs > 1.
    Returns the fitted tree together with a boolean OOB mask (True where a
    sample was NOT used to train this particular tree).
    """
    (X, y, max_depth, min_samples_split, max_features, criterion, bootstrap, seed) = (
        args
    )

    n_samples = X.shape[0]
    rng = np.random.RandomState(seed)
    sample_indices: NDArray[np.int64]

    if bootstrap:
        sample_indices = np.asarray(
            rng.randint(0, n_samples, size=n_samples), dtype=np.int64
        )
    else:
        sample_indices = np.arange(n_samples, dtype=np.int64)

    oob_mask = np.ones(n_samples, dtype=bool)
    oob_mask[sample_indices] = False

    tree = DecisionTree(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        criterion=criterion,
        max_features=max_features,
        random_state=seed,
    )
    tree.fit(X[sample_indices], y[sample_indices])
    return tree, oob_mask


class RandomForestClassifier:
    """
    Bagging ensemble of DecisionTree instances with per-split feature
    sub-sampling (the classic Breiman 2001 Random Forest recipe).
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        max_features: Optional[Union[int, str]] = "sqrt",
        min_samples_split: int = 2,
        criterion: str = "gini",
        bootstrap: bool = True,
        oob_score: bool = False,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.estimators_: List[DecisionTree] = []
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: int = 0
        self.n_features_: int = 0

        self._oob_score: Optional[float] = None
        self._feature_importances: Optional[np.ndarray] = None

    @property
    def _fitted_classes(self) -> np.ndarray:
        """
        Returns `self.classes_` with its type narrowed from Optional[np.ndarray]
        to np.ndarray.

        This informs static type checkers (Pylance/mypy) that `self.classes_`
        cannot be None at this point and provides a clear error message if
        prediction methods are called before fitting the model.
        """
        if self.classes_ is None:
            raise RuntimeError("Model has not been fitted yet: call fit(X, y) first.")
        return self.classes_

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        X = np.asarray(X)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]

        master_rng = np.random.RandomState(self.random_state)

        seeds: np.ndarray = master_rng.randint(
            0, _SEED_UPPER_BOUND, size=self.n_estimators
        )

        job_args = [
            (
                X,
                y,
                self.max_depth,
                self.min_samples_split,
                self.max_features,
                self.criterion,
                self.bootstrap,
                int(seeds[i]),
            )
            for i in range(self.n_estimators)
        ]
        if self.n_jobs is not None and self.n_jobs != 1:
            n_workers = None if self.n_jobs < 0 else self.n_jobs
            with Pool(processes=n_workers) as pool:
                results = pool.map(_fit_one_tree, job_args)
        else:
            results = [_fit_one_tree(a) for a in job_args]

        self.estimators_ = [tree for tree, _ in results]
        oob_masks = [mask for _, mask in results]
        self._feature_importances = self._aggregate_feature_importances()
        if self.oob_score:
            self._oob_score = self._compute_oob_score(X, y, oob_masks)
        return self

    def _aggregate_feature_importances(self) -> np.ndarray:
        importances = np.zeros(self.n_features_)
        for tree in self.estimators_:
            importances += tree.feature_importances()
        return importances / len(self.estimators_)

    def _compute_oob_score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        oob_masks: List[np.ndarray],
    ) -> float:
        n_samples = X.shape[0]
        vote_sums = np.zeros((n_samples, self.n_classes_))
        was_ever_oob = np.zeros(n_samples, dtype=bool)

        for tree, mask in zip(self.estimators_, oob_masks):
            if not np.any(mask):
                continue
            proba = self._aligned_proba(tree, X[mask])
            vote_sums[mask] += proba
            was_ever_oob[mask] = True

        if not np.any(was_ever_oob):
            # Extremely unlikely (needs huge n_estimators or tiny N), but
            # guard against it rather than dividing by zero.
            return float("nan")

        predictions = self._fitted_classes[np.argmax(vote_sums[was_ever_oob], axis=1)]
        return float(np.mean(predictions == y[was_ever_oob]))

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def _aligned_proba(self, tree: DecisionTree, X: np.ndarray) -> np.ndarray:
        """
        Aligns the probability output of a DecisionTree
        with the global class ordering used by the RandomForest.
        """
        tree_proba = tree.predict_proba(X)
        if tree.classes_ is None:
            raise RuntimeError("DecisionTree classes_ is not available")
        # Tree has seen all classes
        if len(tree.classes_) == self.n_classes_:
            return tree_proba

        # If some classes didn't load during Bootstrap
        aligned = np.zeros((X.shape[0], self.n_classes_))
        col_idx = np.searchsorted(self._fitted_classes, tree.classes_)
        aligned[:, col_idx] = tree_proba
        return aligned

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        probs = np.zeros((X.shape[0], self.n_classes_))
        for tree in self.estimators_:
            probs += self._aligned_proba(tree, X)
        return probs / len(self.estimators_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return self._fitted_classes[np.argmax(probs, axis=1)]

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    @property
    def oob_score_(self) -> float:
        if not self.oob_score:
            raise AttributeError(
                "oob_score_ is only available when oob_score=True was set "
                "before calling fit()."
            )
        if self._oob_score is None:
            raise AttributeError("Call fit() before accessing oob_score_.")
        return self._oob_score

    @property
    def feature_importances_(self) -> np.ndarray:
        if self._feature_importances is None:
            raise AttributeError("Call fit() before accessing feature_importances_.")
        return self._feature_importances

    def __repr__(self) -> str:
        fitted = len(self.estimators_) > 0
        status = f"n_trees={len(self.estimators_)}" if fitted else "not fitted"
        return (
            f"RandomForestClassifier(n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth}, max_features={self.max_features!r}, "
            f"bootstrap={self.bootstrap}, {status})"
        )
