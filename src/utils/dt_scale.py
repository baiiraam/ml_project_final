from __future__ import annotations
import time
from typing import Any, Optional, Union
import numpy as np


class DecisionTree:
    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        criterion: str = "gini",
        max_features: Optional[Union[int, str]] = None,
        random_state: Optional[int] = None,
    ) -> None:
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state

        self.tree_: Optional[dict[str, Any]] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_ = 0
        self._n_features = 0  # Backing variable renamed to avoid recursion collision

        self._rng: Optional[np.random.RandomState] = None
        self._depth = 0
        self._n_leaves = 0
        self._impurity_reductions: Optional[np.ndarray] = None

        # Statistics & Timers
        self._n_nodes = 0
        self._split_search_time = 0.0
        self._sort_time_total = 0.0
        self._scan_time_total = 0.0
        self._n_split_searches = 0
        self._n_threshold_checks = 0
        self._depth_stats: dict = {}

    def _feature_subset(self, n_features: int) -> np.ndarray:
        if self.max_features is None:
            return np.arange(n_features)
        elif isinstance(self.max_features, int):
            n_sub = min(n_features, self.max_features)
        elif self.max_features == "sqrt":
            n_sub = max(1, int(np.sqrt(n_features)))
        elif self.max_features in ("log2", "log"):
            n_sub = max(1, int(np.log2(n_features)))
        else:
            return np.arange(n_features)
        rng = self._rng

        if rng is None:
            raise RuntimeError("Random generator not initialized. Call fit() first.")

        return rng.choice(n_features, size=n_sub, replace=False)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "DecisionTree":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0] or X.shape[0] == 0:
            raise ValueError("Invalid shapes for X and y.")

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self._n_features = X.shape[1]
        self._y_encoded = np.searchsorted(self.classes_, y)
        self._rng = np.random.RandomState(self.random_state)

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0], dtype=np.float64)
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)

        self._depth, self._n_leaves, self._n_nodes = 0, 0, 0
        self._split_search_time = 0.0
        self._sort_time_total = 0.0
        self._scan_time_total = 0.0
        self._n_split_searches = 0
        self._n_threshold_checks = 0
        self._impurity_reductions = np.zeros(self._n_features, dtype=np.float64)
        self._depth_stats = {}

        self.tree_ = self._grow(X, y, self._y_encoded, sample_weight, depth=0)
        return self

    def feature_importances(self) -> np.ndarray:
        if self._impurity_reductions is None:
            raise ValueError("The decision tree has not been fitted yet.")
        total = self._impurity_reductions.sum()
        if total == 0:
            return np.zeros(self._n_features)
        return self._impurity_reductions / total

    def _grow(self, X, y, y_encoded, sample_weight, depth):
        node = {}
        self._n_nodes += 1
        n_samples = X.shape[0]

        self._depth_stats.setdefault(depth, [])
        self._depth_stats[depth].append(n_samples)
        total_weight = sample_weight.sum()

        class_counts = np.bincount(
            y_encoded, weights=sample_weight, minlength=self.n_classes_
        ).astype(np.float64)
        node["samples"] = n_samples
        node["value"] = class_counts
        node["impurity"] = self._impurity_from_counts(class_counts, total_weight)
        self._depth = max(self._depth, depth)

        if np.count_nonzero(class_counts) == 1 or n_samples < self.min_samples_split:
            self._n_leaves += 1
            return node
        if self.max_depth is not None and depth >= self.max_depth:
            self._n_leaves += 1
            return node

        t0 = time.perf_counter()
        best = self._best_split(X, y_encoded, sample_weight, total_weight)
        self._split_search_time += time.perf_counter() - t0

        if best is None:
            self._n_leaves += 1
            return node

        feature_idx, threshold, gain = best
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        if not np.any(left_mask) or not np.any(right_mask):
            self._n_leaves += 1
            return node

        self._impurity_reductions[feature_idx] += gain * total_weight
        node["feature_index"] = int(feature_idx)
        node["threshold"] = float(threshold)

        node["left"] = self._grow(
            X[left_mask],
            y[left_mask],
            y_encoded[left_mask],
            sample_weight[left_mask],
            depth + 1,
        )
        node["right"] = self._grow(
            X[right_mask],
            y[right_mask],
            y_encoded[right_mask],
            sample_weight[right_mask],
            depth + 1,
        )
        return node

    def _best_split(
        self,
        X: np.ndarray,
        y_encoded: np.ndarray,
        sample_weight: np.ndarray,
        total_weight: float,
    ):
        self._n_split_searches += 1
        features = self._feature_subset(self._n_features)

        parent_counts = np.bincount(
            y_encoded, weights=sample_weight, minlength=self.n_classes_
        ).astype(np.float64)
        parent_impurity = self._impurity_from_counts(parent_counts, total_weight)

        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        n_samples = len(y_encoded)
        oh_labels_template = np.zeros((n_samples, self.n_classes_), dtype=np.float64)

        for f_idx in features:
            t_sort0 = time.perf_counter()
            order = np.argsort(X[:, f_idx])
            sorted_col = X[order, f_idx]
            sorted_y = y_encoded[order]
            sorted_w = sample_weight[order]
            self._sort_time_total += time.perf_counter() - t_sort0

            t_scan0 = time.perf_counter()
            split_mask = sorted_col[:-1] != sorted_col[1:]
            if not np.any(split_mask):
                continue

            oh_labels = oh_labels_template.copy()
            oh_labels[np.arange(n_samples), sorted_y] = sorted_w

            left_counts_arr = np.cumsum(oh_labels, axis=0)[:-1]
            left_weights_arr = np.cumsum(sorted_w)[:-1]

            left_counts_arr = left_counts_arr[split_mask]
            left_weights_arr = left_weights_arr[split_mask]

            right_weights_arr = total_weight - left_weights_arr
            right_counts_arr = parent_counts - left_counts_arr

            self._n_threshold_checks += len(left_weights_arr)

            with np.errstate(divide="ignore", invalid="ignore"):
                p_left = left_counts_arr / left_weights_arr[:, np.newaxis]
                p_right = right_counts_arr / right_weights_arr[:, np.newaxis]

                if self.criterion == "gini":
                    left_imp = 1.0 - np.sum(p_left**2, axis=1)
                    right_imp = 1.0 - np.sum(p_right**2, axis=1)
                else:
                    left_imp = -np.sum(
                        np.where(
                            p_left > 0, p_left * np.log2(np.maximum(p_left, 1e-12)), 0.0
                        ),
                        axis=1,
                    )
                    right_imp = -np.sum(
                        np.where(
                            p_right > 0,
                            p_right * np.log2(np.maximum(p_right, 1e-12)),
                            0.0,
                        ),
                        axis=1,
                    )

            gains = (
                parent_impurity
                - (left_weights_arr / total_weight) * left_imp
                - (right_weights_arr / total_weight) * right_imp
            )
            self._scan_time_total += time.perf_counter() - t_scan0

            if len(gains) == 0:
                continue

            max_idx = np.argmax(gains)
            if gains[max_idx] > best_gain:
                best_gain = gains[max_idx]
                best_feature = f_idx
                actual_idx = np.where(split_mask)[0][max_idx]
                best_threshold = (
                    sorted_col[actual_idx] + sorted_col[actual_idx + 1]
                ) * 0.5

        if best_feature is None or best_threshold is None or best_gain <= 0:
            return None

        return int(best_feature), float(best_threshold), float(best_gain)

    def _impurity_from_counts(
        self, class_counts: np.ndarray, total_weight: float
    ) -> float:
        if total_weight == 0:
            return 0.0
        probs = class_counts / total_weight
        if self.criterion == "gini":
            return float(1.0 - np.dot(probs, probs))
        nonzero = probs > 0
        return float(-np.sum(probs[nonzero] * np.log2(probs[nonzero])))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.tree_ is None:
            raise ValueError("Model not fitted.")
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]

        # Pre-allocate the entire output array
        probabilities = np.zeros((n_samples, self.n_classes_), dtype=np.float64)

        def _traverse(node: dict, active_indices: np.ndarray) -> None:
            if len(active_indices) == 0:
                return

            if "feature_index" not in node:
                # We hit a leaf: compute probabilities for all samples assigned here
                total = node["value"].sum()
                prob = (
                    node["value"] / total
                    if total > 0
                    else np.ones(self.n_classes_) / self.n_classes_
                )
                probabilities[active_indices] = prob
                return

            # Vectorized splitting of indices at this node
            f_idx = node["feature_index"]
            thresh = node["threshold"]

            # Extract only the relevant feature values for active samples
            go_left = X[active_indices, f_idx] <= thresh

            _traverse(node["left"], active_indices[go_left])
            _traverse(node["right"], active_indices[~go_left])

        # Start traversal from the root with all indices
        _traverse(self.tree_, np.arange(n_samples, dtype=np.int64))
        return probabilities

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)

        if self.classes_ is None:
            raise ValueError("Model is not fitted yet")

        return self.classes_[np.argmax(probs, axis=1)]

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def n_leaves(self) -> int:
        return self._n_leaves

    @property
    def n_features(self) -> int:
        return self._n_features

    def __repr__(self) -> str:
        if self.tree_ is None:
            return "DecisionTree (not fitted)"
        return f"DecisionTree(depth={self._depth}, n_leaves={self._n_leaves}, criterion='{self.criterion}')"
