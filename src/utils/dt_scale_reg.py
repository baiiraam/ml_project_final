# import numpy as np


# class RegressionNode:
#     def __init__(
#         self,
#         feature=None,
#         threshold=None,
#         left=None,
#         right=None,
#         value=None,
#         is_leaf=False,
#     ):
#         self.feature = feature
#         self.threshold = threshold
#         self.left = left
#         self.right = right
#         self.value = value  # Mean target value in this node
#         self.is_leaf = is_leaf


# class RegressionTreeScale:
#     def __init__(
#         self, max_depth=3, min_samples_split=2, max_features=None, random_state=None
#     ):
#         self.max_depth = max_depth
#         self.min_samples_split = min_samples_split
#         self.max_features = max_features
#         self.random_state = random_state
#         self.root = None
#         self.rng = np.random.default_rng(random_state)

#     def fit(self, X, y):
#         self.n_features = X.shape[1]
#         self.root = self._build_tree(X, y, depth=0)
#         return self

#     def _get_max_features(self):
#         if self.max_features is None:
#             return self.n_features
#         elif self.max_features == "sqrt":
#             return max(1, int(np.sqrt(self.n_features)))
#         elif self.max_features == "log2":
#             return max(1, int(np.log2(self.n_features)))
#         elif isinstance(self.max_features, float):
#             return max(1, int(self.max_features * self.n_features))
#         elif isinstance(self.max_features, int):
#             return min(self.n_features, self.max_features)
#         return self.n_features

#     def _build_tree(self, X, y, depth):
#         n_samples, n_features = X.shape

#         # Base cases
#         if (
#             depth >= self.max_depth
#             or n_samples < self.min_samples_split
#             or np.all(y == y[0])
#         ):
#             return RegressionNode(value=np.mean(y), is_leaf=True)

#         # Vectorized split finding
#         best_feat, best_thresh, best_val = self._find_best_split(X, y)
#         if best_feat is None:
#             return RegressionNode(value=np.mean(y), is_leaf=True)

#         left_idx = X[:, best_feat] <= best_thresh
#         right_idx = ~left_idx

#         # Prevent splitting if it results in empty children
#         if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
#             return RegressionNode(value=np.mean(y), is_leaf=True)

#         left_child = self._build_tree(X[left_idx], y[left_idx], depth + 1)
#         right_child = self._build_tree(X[right_idx], y[right_idx], depth + 1)

#         return RegressionNode(
#             feature=best_feat, threshold=best_thresh, left=left_child, right=right_child
#         )

#     def _find_best_split(self, X, y):
#         n_samples, n_features = X.shape
#         best_impurity_reduction = -1.0
#         best_feat, best_thresh = None, None

#         # Compute parent variance (MSE reduction targets)
#         parent_sum = np.sum(y)
#         parent_sum_sq = np.sum(y**2)
#         parent_mse = (parent_sum_sq / n_samples) - (parent_sum / n_samples) ** 2

#         # Feature Sub-sampling (Column Subsampling)
#         n_sub_features = self._get_max_features()
#         feature_indices = self.rng.choice(
#             n_features, size=n_sub_features, replace=False
#         )

#         for feat in feature_indices:
#             X_column = X[:, feat]
#             sort_idx = np.argsort(X_column)
#             X_sorted, y_sorted = X_column[sort_idx], y[sort_idx]

#             # Vectorized calculations of left and right split properties
#             sum_left = np.cumsum(y_sorted)[:-1]
#             sum_right = parent_sum - sum_left

#             sum_sq_left = np.cumsum(y_sorted**2)[:-1]
#             sum_sq_right = parent_sum_sq - sum_sq_left

#             n_l = np.arange(1, n_samples)
#             n_r = n_samples - n_l

#             # Variance (MSE) of left and right segments
#             mse_left = np.maximum(0.0, (sum_sq_left / n_l) - (sum_left / n_l) ** 2)
#             mse_right = np.maximum(0.0, (sum_sq_right / n_r) - (sum_right / n_r) ** 2)

#             # Combined child impurity
#             child_impurity = (n_l / n_samples) * mse_left + (
#                 n_r / n_samples
#             ) * mse_right
#             impurity_reduction = parent_mse - child_impurity

#             # Mask out non-split threshold boundaries (identical adjacent values)
#             split_mask = X_sorted[:-1] != X_sorted[1:]

#             if np.any(split_mask):
#                 valid_indices = np.where(split_mask)[0]
#                 if len(valid_indices) == 0:
#                     continue
#                 best_idx_in_feat = valid_indices[
#                     np.argmax(impurity_reduction[valid_indices])
#                 ]

#                 if impurity_reduction[best_idx_in_feat] > best_impurity_reduction:
#                     best_impurity_reduction = impurity_reduction[best_idx_in_feat]
#                     best_feat = feat
#                     # Split in the middle of adjacent distinct sorted values
#                     best_thresh = (
#                         X_sorted[best_idx_in_feat] + X_sorted[best_idx_in_feat + 1]
#                     ) / 2.0

#         return best_feat, best_thresh, best_impurity_reduction

#     def predict(self, X):
#         return np.array([self._predict_row(self.root, row) for row in X])

#     def _predict_row(self, node, row):
#         if node.is_leaf:
#             return node.value
#         if row[node.feature] <= node.threshold:
#             return self._predict_row(node.left, row)
#         return self._predict_row(node.right, row)

#     # Helper to retrieve leaf IDs for terminal-node updates
#     def apply(self, X):
#         return np.array(
#             [self._find_leaf_index(self.root, row, id_prefix="") for row in X]
#         )

#     def _find_leaf_index(self, node, row, id_prefix):
#         if node.is_leaf:
#             return id_prefix
#         if row[node.feature] <= node.threshold:
#             return self._find_leaf_index(node.left, row, id_prefix + "L")
#         return self._find_leaf_index(node.right, row, id_prefix + "R")















import time
import numpy as np


class RegressionNode:
    def __init__(
        self,
        feature=None,
        threshold=None,
        left=None,
        right=None,
        value=None,
        is_leaf=False,
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Mean target value in this node
        self.is_leaf = is_leaf


class RegressionTreeScale:
    """
    Highly optimized Regression Tree with vectorized MSE split-finding.
    """
    def __init__(
        self, max_depth=3, min_samples_split=2, max_features=None, random_state=None
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.root = None
        self.rng = np.random.default_rng(random_state)

        # Diagnostics & Profiling
        self.fit_time_ = 0.0
        self.predict_time_ = 0.0

    def fit(self, X, y):
        start_time = time.perf_counter()
        self.n_features = X.shape[1]
        self.root = self._build_tree(X, y, depth=0)
        self.fit_time_ = time.perf_counter() - start_time
        return self

    def _get_max_features(self):
        if self.max_features is None:
            return self.n_features
        elif self.max_features == "sqrt":
            return max(1, int(np.sqrt(self.n_features)))
        elif self.max_features == "log2":
            return max(1, int(np.log2(self.n_features)))
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * self.n_features))
        elif isinstance(self.max_features, int):
            return min(self.n_features, self.max_features)
        return self.n_features

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape

        # Base cases
        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or np.all(y == y[0])
        ):
            return RegressionNode(value=np.mean(y), is_leaf=True)

        # Vectorized split finding
        best_feat, best_thresh, best_val = self._find_best_split(X, y)
        if best_feat is None:
            return RegressionNode(value=np.mean(y), is_leaf=True)

        left_idx = X[:, best_feat] <= best_thresh
        right_idx = ~left_idx

        # Prevent splitting if it results in empty children
        if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
            return RegressionNode(value=np.mean(y), is_leaf=True)

        left_child = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx], y[right_idx], depth + 1)

        return RegressionNode(
            feature=best_feat, threshold=best_thresh, left=left_child, right=right_child
        )

    def _find_best_split(self, X, y):
        n_samples, n_features = X.shape
        best_impurity_reduction = -1.0
        best_feat, best_thresh = None, None

        # Compute parent variance (MSE reduction targets)
        parent_sum = np.sum(y)
        parent_sum_sq = np.sum(y**2)

        # Protect parent MSE from micro floating-point variances below zero
        parent_mse = max(0.0, (parent_sum_sq / n_samples) - (parent_sum / n_samples) ** 2)

        # Feature Sub-sampling (Column Subsampling)
        n_sub_features = self._get_max_features()
        feature_indices = self.rng.choice(
            n_features, size=n_sub_features, replace=False
        )

        for feat in feature_indices:
            X_column = X[:, feat]
            sort_idx = np.argsort(X_column)
            X_sorted, y_sorted = X_column[sort_idx], y[sort_idx]

            # Vectorized calculations of left and right split properties
            sum_left = np.cumsum(y_sorted)[:-1]
            sum_right = parent_sum - sum_left

            sum_sq_left = np.cumsum(y_sorted**2)[:-1]
            sum_sq_right = parent_sum_sq - sum_sq_left

            n_l = np.arange(1, n_samples)
            n_r = n_samples - n_l

            # Fix potential numerical stability in MSE calculations (avoid variance < 0)
            mse_left = np.maximum(0.0, (sum_sq_left / n_l) - (sum_left / n_l) ** 2)
            mse_right = np.maximum(0.0, (sum_sq_right / n_r) - (sum_right / n_r) ** 2)

            # Combined child impurity
            child_impurity = (n_l / n_samples) * mse_left + (
                n_r / n_samples
            ) * mse_right
            impurity_reduction = parent_mse - child_impurity

            # Mask out non-split threshold boundaries (identical adjacent values)
            split_mask = X_sorted[:-1] != X_sorted[1:]

            if np.any(split_mask):
                valid_indices = np.where(split_mask)[0]
                if len(valid_indices) == 0:
                    continue
                best_idx_in_feat = valid_indices[
                    np.argmax(impurity_reduction[valid_indices])
                ]

                if impurity_reduction[best_idx_in_feat] > best_impurity_reduction:
                    best_impurity_reduction = impurity_reduction[best_idx_in_feat]
                    best_feat = feat
                    # Split in the middle of adjacent distinct sorted values
                    best_thresh = (
                        X_sorted[best_idx_in_feat] + X_sorted[best_idx_in_feat + 1]
                    ) / 2.0

        return best_feat, best_thresh, best_impurity_reduction

    def predict(self, X):
        start_time = time.perf_counter()
        predictions = np.array([self._predict_row(self.root, row) for row in X])
        self.predict_time_ = time.perf_counter() - start_time
        return predictions

    def _predict_row(self, node, row):
        if node.is_leaf:
            return node.value
        if row[node.feature] <= node.threshold:
            return self._predict_row(node.left, row)
        return self._predict_row(node.right, row)

    # Helper to retrieve leaf IDs for terminal-node updates
    def apply(self, X):
        return np.array(
            [self._find_leaf_index(self.root, row, id_prefix="") for row in X]
        )

    def _find_leaf_index(self, node, row, id_prefix):
        if node.is_leaf:
            return id_prefix
        if row[node.feature] <= node.threshold:
            return self._find_leaf_index(node.left, row, id_prefix + "L")
        return self._find_leaf_index(node.right, row, id_prefix + "R")