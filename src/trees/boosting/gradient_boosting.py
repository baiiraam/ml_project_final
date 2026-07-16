import numpy as np
from src.utils.dt_scale_reg import (
    RegressionTreeScale,
)  # Importing our new fast regression tree


class FastBinaryGradientBoosting:
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        subsample=1.0,
        max_features=None,
        reg_lambda=1.0,
        random_state=None,
    ):
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if not (
            0.0 < learning_rate <= 1.0
        ):  # adjust range if your tests expect different bounds
            raise ValueError("learning_rate must be in (0, 1]")
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2")
        if not (0.0 < subsample <= 1.0):
            raise ValueError("subsample must be in (0, 1]")
        if reg_lambda < 0.0:
            raise ValueError("reg_lambda must be non-negative")

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.max_features = max_features
        self.reg_lambda = reg_lambda
        self.random_state = random_state

        self.trees = []
        self.raw_initial_val_ = None
        self.rng = np.random.default_rng(random_state)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if X.size == 0 or y.size == 0:
            raise ValueError("X and y must not be empty")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")

        n_samples, n_features = X.shape
        y = np.array(y, dtype=np.float64)

        # 1. Initialize prediction with log-odds
        p_mean = np.mean(y)
        p_mean = np.clip(p_mean, 1e-15, 1.0 - 1e-15)
        self.raw_initial_val_ = np.log(p_mean / (1.0 - p_mean))

        # Raw margin predictions F(X)
        F = np.full(n_samples, self.raw_initial_val_, dtype=np.float64)

        # Sigmoid probability activations p(x)
        p = 1.0 / (1.0 + np.exp(-F))

        for i in range(self.n_estimators):
            # 2. Calculate Negative Gradients (Residuals)
            residuals = y - p

            # 3. Stochastic Row Subsampling
            if self.subsample < 1.0:
                sample_size = int(self.subsample * n_samples)
                sub_indices = self.rng.choice(
                    n_samples, size=sample_size, replace=False
                )
                X_b, res_b = X[sub_indices], residuals[sub_indices]
            else:
                X_b, res_b = X, residuals

            # 4. Fit regression tree on (subsampled) residuals
            tree = RegressionTreeScale(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=self.rng.integers(0, 1000000)
                if self.random_state
                else None,
            )
            tree.fit(X_b, res_b)

            # 5. Newton-Raphson Terminal Node Value Adjustment
            # Find which leaf node identifier each row lands on
            leaf_ids = tree.apply(X)
            unique_leaves = np.unique(leaf_ids)

            # Re-solve leaf values using Newton-Raphson + L2 Regularization (lambda)
            leaf_value_mapping = {}
            for leaf in unique_leaves:
                mask = leaf_ids == leaf
                num = np.sum(residuals[mask])
                p_clipped = np.clip(p[mask], 1e-15, 1.0 - 1e-15)
                den = np.sum(p_clipped * (1.0 - p_clipped)) + self.reg_lambda

                # Leaf adjustment value
                gamma = num / den if den > 1e-10 else 0.0
                leaf_value_mapping[leaf] = gamma

            # Update the tree leaves directly so the tree predict outputs adjusted gammas
            self._update_tree_leaf_values(tree.root, "", leaf_value_mapping)

            # 6. Shrinkage: Update raw predictions F(X) and probabilities p(x)
            F += self.learning_rate * tree.predict(X)
            p = 1.0 / (1.0 + np.exp(-F))

            self.trees.append(tree)

        return self

    def _update_tree_leaf_values(self, node, current_id, leaf_value_mapping):
        """Recursively replaces tree node means with optimized Newton-Raphson step values."""
        if node.is_leaf:
            node.value = leaf_value_mapping.get(current_id, node.value)
            return
        self._update_tree_leaf_values(node.left, current_id + "L", leaf_value_mapping)
        self._update_tree_leaf_values(node.right, current_id + "R", leaf_value_mapping)

    def decision_function(self, X):
        """Returns raw margin scale predictions F(X)."""
        if self.raw_initial_val_ is None:
            raise ValueError("Model has not been fitted yet")

        F = np.full(X.shape[0], self.raw_initial_val_)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        return F

    def predict_proba(self, X):
        """Vectorized probability estimation."""
        if self.raw_initial_val_ is None:
            raise ValueError("Model has not been fitted yet")

        F = self.decision_function(X)
        p = 1.0 / (1.0 + np.exp(-F))
        return np.vstack([1.0 - p, p]).T

    def predict(self, X):
        """Predict binary output label [0, 1]."""
        prob = self.predict_proba(X)[:, 1]
        return (prob >= 0.5).astype(int)

    def staged_predict(self, X):
        """
        Yield predictions after each boosting stage.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input samples.

        Yields
        ------
        np.ndarray, shape (n_samples,)
            Predictions after each stage.
        """
        if not self.trees:
            raise ValueError("Model has not been fitted yet.")

        X = np.asarray(X, dtype=np.float64)
        F = np.full(X.shape[0], self.raw_initial_val_, dtype=np.float64)

        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
            p = 1.0 / (1.0 + np.exp(-np.clip(F, -500, 500)))
            yield (p >= 0.5).astype(int)

    @property
    def estimator_weights(self):
        if not self.trees:
            raise ValueError("Model has not been fitted yet.")
        return np.full(len(self.trees), self.learning_rate)
