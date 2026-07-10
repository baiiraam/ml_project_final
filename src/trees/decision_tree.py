from __future__ import annotations
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
        self.n_classes_: int = 0
        self.n_features_: int = 0
        self.classes_: Optional[np.ndarray] = None
        self._rng: Optional[np.random.RandomState] = None
        self._depth: int = 0
        self._n_leaves: int = 0
        self._impurity_reductions: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> DecisionTree:
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        if X.shape[0] == 0:
            raise ValueError("X and y must not be empty")
        self.classes_ = np.unique(y)
        assert self.classes_ is not None
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        self._rng = np.random.RandomState(self.random_state)
        self._depth = 0
        self._n_leaves = 0
        self._impurity_reductions = np.zeros(
            self.n_features_,
            dtype=np.float64,
        )
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0], dtype=np.float64)
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            if sample_weight.ndim != 1:
                raise ValueError("sample_weight must be a 1D array")
            if len(sample_weight) != len(y):
                raise ValueError("sample_weight must have the same length as y")
            if np.any(sample_weight < 0):
                raise ValueError("sample_weight must be non-negative")
        self.tree_ = self._grow(X, y, sample_weight, depth=0)
        return self

    def _grow(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        depth: int,
    ) -> dict[str, Any]:
        assert self.classes_ is not None
        assert self._impurity_reductions is not None
        node: dict = {}
        n_samples = X.shape[0]
        total_weight = sample_weight.sum()
        class_counts = np.array([
            sample_weight[y == c].sum() for c in self.classes_
        ], dtype=np.float64)
        node["samples"] = n_samples
        node["value"] = class_counts
        if depth > self._depth:
            self._depth = depth
        pure = len(np.unique(y)) == 1
        no_split_possible = n_samples < 2 or np.all(X == X[0])
        if pure or no_split_possible:
            self._n_leaves += 1
            return node
        if self.max_depth is not None and depth >= self.max_depth:
            self._n_leaves += 1
            return node
        if n_samples < self.min_samples_split:
            self._n_leaves += 1
            return node
        best = self._best_split(X, y, sample_weight, total_weight)
        if best is None:
            self._n_leaves += 1
            return node
        feature_idx, threshold, gain = best
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            self._n_leaves += 1
            return node
        self._impurity_reductions[feature_idx] += gain * n_samples
        node["feature_index"] = int(feature_idx)
        node["threshold"] = float(threshold)
        node["left"] = self._grow(
            X[left_mask], y[left_mask], sample_weight[left_mask], depth + 1
        )
        node["right"] = self._grow(
            X[right_mask], y[right_mask], sample_weight[right_mask], depth + 1
        )
        return node

    def _best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        total_weight: float,
) -> Optional[tuple[int, float, float]]:
        assert self.classes_ is not None
        n_features = X.shape[1]
        features = self._feature_subset(n_features)
        parent_impurity = self._impurity(sample_weight, y, total_weight)
        best_gain: float = -np.inf
        best_feature: Optional[int] = None
        best_threshold: Optional[float] = None

        for f_idx in features:
            col = X[:, f_idx]
            order = np.argsort(col)
            sorted_col = col[order]
            sorted_y = y[order]
            sorted_w = sample_weight[order]
            n = len(sorted_col)
            class_indices = np.searchsorted(self.classes_, sorted_y)
            total_class_counts = np.bincount(
                class_indices,
                weights=sorted_w,
                minlength=self.n_classes_,
            ).astype(np.float64)

            left_counts = np.zeros(self.n_classes_, dtype=np.float64)
            left_w_sum = 0.0

            for i in range(n - 1):
                class_idx = class_indices[i]
                left_w_sum += sorted_w[i]
                left_counts[class_idx] += sorted_w[i]
                if sorted_col[i] == sorted_col[i + 1]:
                    continue
                right_w_sum = total_weight - left_w_sum
                if left_w_sum <= 0.0 or right_w_sum <= 0.0:
                    continue
                right_counts = total_class_counts - left_counts
                gain = (
                    parent_impurity
                    - (left_w_sum / total_weight)
                    * self._impurity_from_counts(left_counts, left_w_sum)
                    - (right_w_sum / total_weight)
                    * self._impurity_from_counts(right_counts, right_w_sum)
                )
                if gain > best_gain:
                    best_gain = gain
                    best_feature = f_idx
                    best_threshold = (sorted_col[i] + sorted_col[i + 1]) * 0.5

        if (best_feature is None or best_threshold is None or best_gain <= 0.0):
            return None
        return int(best_feature), float(best_threshold), float(best_gain)

    def _feature_subset(self, n_features: int) -> list:
        assert self._rng is not None
        if self.max_features is None:
            return list(range(n_features))
        if isinstance(self.max_features, int):
            if self.max_features <= 0 or self.max_features > n_features:
                raise ValueError(
                    f"max_features must be in (0, {n_features}], got {self.max_features}"
                )
            k = min(self.max_features, n_features)
            return self._rng.choice(n_features, k, replace=False).tolist()
        if isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                k = max(1, int(np.sqrt(n_features)))
            elif self.max_features == "log2":
                k = max(1, int(np.log2(n_features)))
            else:
                k = n_features
            return self._rng.choice(n_features, k, replace=False).tolist()
        return list(range(n_features))

    def _impurity(
        self,
        sample_weight: np.ndarray,
        y: np.ndarray,
        total_weight: float,
    ) -> float:
        assert self.classes_ is not None
        if total_weight == 0:
            return 0.0
        class_counts = np.array([
            sample_weight[y == c].sum() for c in self.classes_
        ], dtype=np.float64)
        return self._impurity_from_counts(class_counts, total_weight)

    def _impurity_from_counts(
        self,
        class_counts: np.ndarray,
        total_weight: float,
    ) -> float:
        if total_weight == 0:
            return 0.0
        probs = class_counts / total_weight
        if self.criterion == "gini":
            return float(1.0 - np.sum(probs ** 2))
        if self.criterion == "entropy":
            eps = 1e-12
            return float(-np.sum(probs * np.log2(probs + eps)))
        return 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.tree_ is None:
            raise ValueError("The decision tree has not been fitted yet.")
        return np.array([self._predict_row(x) for x in X])

    def _predict_row(self, x: np.ndarray) -> int:
        assert self.tree_ is not None
        assert self.classes_ is not None
        node = self.tree_
        while "feature_index" in node:
            if x[node["feature_index"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return int(self.classes_[np.argmax(node["value"])])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.tree_ is None:
            raise ValueError("The decision tree has not been fitted yet.")
        return np.array([self._predict_proba_row(x) for x in X])

    def _predict_proba_row(self, x: np.ndarray) -> np.ndarray:
        if self.tree_ is None:
            raise ValueError("The decision tree has not been fitted yet.")
        node = self.tree_
        while "feature_index" in node:
            if x[node["feature_index"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        total = node["value"].sum()
        if total == 0:
            return np.ones(self.n_classes_) / self.n_classes_
        return node["value"] / total

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def n_leaves(self) -> int:
        return self._n_leaves

    def feature_importances(self) -> np.ndarray:
        if self._impurity_reductions is None:
            raise ValueError("The decision tree has not been fitted yet.")
        total = self._impurity_reductions.sum()
        if total == 0:
            return np.zeros(self.n_features_)
        return self._impurity_reductions / total

    def _repr_tree(self, node: dict, depth: int = 0, prefix: str = "") -> str:
        indent = prefix
        if "feature_index" not in node:
            val = node["value"]
            total = val.sum()
            probs = val / total if total > 0 else val
            counts_str = ", ".join(f"{int(v)}" for v in val)
            probs_str = ", ".join(f"{p:.2f}" for p in probs)
            return (
                f"{indent}[{counts_str}] "
                f"probs=[{probs_str}] "
                f"samples={node['samples']}\n"
            )
        feat = node["feature_index"]
        thresh = node["threshold"]
        result = f"{indent}feature_{feat} <= {thresh:.4f}\n"
        new_prefix = prefix + "|   "
        if depth < 4:
            result += self._repr_tree(node["left"], depth + 1, new_prefix)
            result += self._repr_tree(node["right"], depth + 1, new_prefix)
        else:
            for side, child in [("left", node["left"]), ("right", node["right"])]:
                val = child["value"]
                total = val.sum()
                counts_str = ", ".join(f"{int(v)}" for v in val)
                result += f"{new_prefix}[{counts_str}]  samples={child['samples']}\n"
        return result

    def __repr__(self) -> str:
        if self.tree_ is None:
            return "DecisionTree (not fitted)"
        if self._depth > 4:
            return (
                f"DecisionTree(depth={self._depth}, n_leaves={self._n_leaves}, "
                f"criterion={self.criterion})"
            )
        return (
            f"DecisionTree(depth={self._depth}, n_leaves={self._n_leaves}, "
            f"criterion={self.criterion})\n{self._repr_tree(self.tree_)}"
        )
