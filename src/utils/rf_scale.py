from __future__ import annotations
import time
from multiprocessing import Pool
from typing import List, Optional, Tuple, Union
import numpy as np
from src.utils.dt_scale import DecisionTree

_SEED_UPPER_BOUND = 2**31 - 1

# Parallel Processing Globals to eliminate payload pickle overhead across workers
_SHARED_X: Optional[np.ndarray] = None
_SHARED_Y: Optional[np.ndarray] = None


def _init_worker(X: np.ndarray, y: np.ndarray) -> None:
    global _SHARED_X, _SHARED_Y
    _SHARED_X = X
    _SHARED_Y = y


def _fit_one_tree(
    args: Tuple[Optional[int], int, Optional[Union[int, str]], str, bool, int],
) -> Tuple[DecisionTree, np.ndarray]:
    max_depth, min_samples_split, max_features, criterion, bootstrap, seed = args

    if _SHARED_X is None or _SHARED_Y is None:
        raise RuntimeError("Worker arrays were not properly shared.")

    n_samples = _SHARED_X.shape[0]
    rng = np.random.RandomState(seed)

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

    tree.fit(_SHARED_X[sample_indices], _SHARED_Y[sample_indices])
    return tree, oob_mask


class RandomForestClassifier:
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
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if n_jobs == 0:
            raise ValueError("n_jobs cannot be 0")
        if oob_score and not bootstrap:
            raise ValueError("oob_score=True requires bootstrap=True")

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
        if self.classes_ is None:
            raise RuntimeError("Model has not been fitted yet: call fit(X, y) first.")
        return self.classes_

    def fit(self, X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
        X = np.asarray(X)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]

        master_rng = np.random.RandomState(self.random_state)
        seeds = master_rng.randint(0, _SEED_UPPER_BOUND, size=self.n_estimators)

        # Lightweight arguments payload without passing explicit X, y matrices inside loops
        job_args = [
            (
                self.max_depth,
                self.min_samples_split,
                self.max_features,
                self.criterion,
                self.bootstrap,
                int(seeds[i]),
            )
            for i in range(self.n_estimators)
        ]

        t0 = time.perf_counter()
        if self.n_jobs != 1:
            n_workers = None if self.n_jobs < 0 else self.n_jobs
            # Using initializer & initargs works perfectly across fork and spawn contexts
            with Pool(
                processes=n_workers, initializer=_init_worker, initargs=(X, y)
            ) as pool:
                results = pool.map(_fit_one_tree, job_args)
        else:
            _init_worker(X, y)
            results = [_fit_one_tree(a) for a in job_args]
        print(f"Forest fit took {time.perf_counter() - t0:.3f}s")

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
        self, X: np.ndarray, y: np.ndarray, oob_masks: List[np.ndarray]
    ) -> float:
        n_samples = X.shape[0]
        vote_counts = np.zeros((n_samples, self.n_classes_), dtype=np.int64)
        was_ever_oob = np.zeros(n_samples, dtype=bool)

        for tree, mask in zip(self.estimators_, oob_masks):
            if not np.any(mask):
                continue
            pred = tree.predict(X[mask])
            indices = np.flatnonzero(mask)

            encoded_preds = np.searchsorted(self._fitted_classes, pred)
            vote_counts[indices, encoded_preds] += 1
            was_ever_oob[mask] = True

        if not np.any(was_ever_oob):
            return float("nan")

        predictions = self._fitted_classes[np.argmax(vote_counts[was_ever_oob], axis=1)]
        return float(np.mean(predictions == y[was_ever_oob]))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.estimators_:
            raise RuntimeError("Model has not been fitted yet.")
        X = np.asarray(X)

        # Accumulate probabilities from all trees simultaneously
        probs = np.zeros((X.shape[0], self.n_classes_), dtype=np.float64)
        for tree in self.estimators_:
            probs += self._aligned_proba(tree, X)

        return probs / len(self.estimators_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Instead of calling predict() tree-by-tree and running searchsorted,
        # simply argmax the averaged class probabilities!
        probs = self.predict_proba(X)
        return self._fitted_classes[np.argmax(probs, axis=1)]

    def _aligned_proba(self, tree: DecisionTree, X: np.ndarray) -> np.ndarray:
        tree_proba = tree.predict_proba(X)
        if tree.classes_ is None:
            raise RuntimeError("DecisionTree classes_ is not available")
        if np.array_equal(tree.classes_, self._fitted_classes):
            return tree_proba

        aligned = np.zeros((X.shape[0], self.n_classes_))
        col_idx = np.searchsorted(self._fitted_classes, tree.classes_)
        aligned[:, col_idx] = tree_proba
        return aligned

    @property
    def oob_score_(self) -> float:
        if not self.oob_score or self._oob_score is None:
            raise AttributeError("OOB score is unavailable or model is un-fitted.")
        return self._oob_score

    @property
    def feature_importances_(self) -> np.ndarray:
        if self._feature_importances is None:
            raise AttributeError("Call fit() before accessing feature importances.")
        return self._feature_importances

    def __repr__(self) -> str:
        fitted = "fitted" if self.estimators_ else "not fitted"
        return f"RandomForestClassifier(n_estimators={self.n_estimators}, max_depth={self.max_depth}, {fitted})"
