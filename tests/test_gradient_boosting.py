"""
Unit tests for FastBinaryGradientBoosting (binary-only)

Tests cover:
1. Basic functionality (fit, predict, predict_proba)
2. Parameter validation
3. Staged predictions
4. Performance metrics
5. Edge cases
6. Reproducibility
"""

import re
import pytest
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from src.trees.boosting.gradient_boosting import FastBinaryGradientBoosting


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def binary_data() -> Tuple[np.ndarray, np.ndarray]:
    """Generates a simple, linearly separable binary classification dataset."""
    X = np.array(
        [
            [1.0, 2.0],
            [1.5, 1.8],
            [5.0, 8.0],
            [6.0, 9.0],
            [1.2, 1.5],
            [5.5, 7.8],
            [0.8, 1.1],
            [6.2, 8.5],
        ],
        dtype=np.float64,
    )
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    return X, y


@pytest.fixture
def binary_data_split(binary_data):
    """Split binary data into train/test."""
    X, y = binary_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


# =====================================================================
# Tests: Initialization & Argument Validation
# =====================================================================


def test_initialization_defaults() -> None:
    """Verifies that the default parameters are set correctly."""
    clf = FastBinaryGradientBoosting()
    assert clf.n_estimators == 100
    assert clf.learning_rate == 0.1
    assert clf.max_depth == 3
    assert clf.min_samples_split == 2
    assert clf.subsample == 1.0
    assert clf.reg_lambda == 1.0


@pytest.mark.parametrize(
    "params, expected_error",
    [
        ({"n_estimators": 0}, "n_estimators must be positive"),
        ({"n_estimators": -5}, "n_estimators must be positive"),
        ({"learning_rate": 0.0}, "learning_rate must be in"),
        ({"learning_rate": 1.5}, "learning_rate must be in"),
        ({"max_depth": 0}, "max_depth must be at least"),
        ({"min_samples_split": 1}, "min_samples_split must be at least"),
        ({"subsample": 0.0}, "subsample must be in"),
        ({"subsample": 1.5}, "subsample must be in"),
        ({"reg_lambda": -1.0}, "reg_lambda must be non-negative"),
    ],
)
def test_invalid_parameters(params: dict, expected_error: str) -> None:
    """Ensures improper parameters immediately raise descriptive ValueErrors."""
    with pytest.raises(ValueError, match=re.escape(expected_error)):
        FastBinaryGradientBoosting(**params)


def test_fit_input_validation() -> None:
    """Tests that fit() enforces matching shapes and non-empty inputs."""
    clf = FastBinaryGradientBoosting()
    X = np.array([[1, 2], [3, 4]])
    y_mismatched = np.array([1])

    with pytest.raises(ValueError, match="same number of samples"):
        clf.fit(X, y_mismatched)

    with pytest.raises(ValueError, match="X and y must not be empty"):
        clf.fit(np.array([]).reshape(0, 2), np.array([]))


# =====================================================================
# Tests: Binary Classification Performance & Mechanics
# =====================================================================


def test_binary_classification_fit(binary_data: Tuple[np.ndarray, np.ndarray]) -> None:
    """Verifies convergence and basic prediction on a simple binary problem."""
    X, y = binary_data
    clf = FastBinaryGradientBoosting(
        n_estimators=20, max_depth=3, learning_rate=0.5, random_state=42
    )
    clf.fit(X, y)

    # Internal state checks
    assert len(clf.trees) == 20
    assert clf.raw_initial_val_ is not None

    # Predictions
    preds = clf.predict(X)
    assert preds.shape == (X.shape[0],)

    # Check accuracy is reasonable (better than random)
    acc = accuracy_score(y, preds)
    assert acc > 0.60

    # Probabilities
    probs = clf.predict_proba(X)
    assert probs.shape == (X.shape[0], 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    np.testing.assert_allclose(np.sum(probs, axis=1), 1.0)


# =====================================================================
# Tests: Diagnostic Features (Staged Predictions)
# =====================================================================


def test_staged_predict(binary_data: Tuple[np.ndarray, np.ndarray]) -> None:
    """Validates that staged_predict successfully yields predictions for each stage."""
    X, y = binary_data
    clf = FastBinaryGradientBoosting(n_estimators=5, random_state=42)
    clf.fit(X, y)

    stages = list(clf.staged_predict(X))
    assert len(stages) == 5
    for stage_pred in stages:
        assert stage_pred.shape == (X.shape[0],)
        assert set(np.unique(stage_pred)) == {0, 1}


# =====================================================================
# Tests: Guardrails & API Contracts
# =====================================================================


def test_prediction_without_fit() -> None:
    """Verifies that calling prediction methods before fit() raises a clean ValueError."""
    clf = FastBinaryGradientBoosting()
    X = np.array([[1, 2]])

    with pytest.raises(ValueError, match="Model has not been fitted yet"):
        clf.predict(X)

    with pytest.raises(ValueError, match="Model has not been fitted yet"):
        clf.predict_proba(X)

    with pytest.raises(ValueError, match="Model has not been fitted yet"):
        list(clf.staged_predict(X))


# =====================================================================
# Tests: Edge Cases
# =====================================================================


def test_edge_cases() -> None:
    """Test edge cases: small dataset, single feature, perfect separation."""
    # Small dataset
    X, y = make_classification(
        n_samples=10, n_features=3, n_informative=2, n_redundant=0, random_state=42
    )
    clf = FastBinaryGradientBoosting(n_estimators=5, max_depth=2, random_state=42)
    clf.fit(X, y)
    assert len(clf.trees) == 5

    # Single feature
    X, y = make_classification(
        n_samples=100,
        n_features=1,
        n_informative=1,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42,
    )
    clf = FastBinaryGradientBoosting(n_estimators=5, random_state=42)
    clf.fit(X, y)
    assert clf.trees[0].root is not None

    # Perfectly separable
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    clf = FastBinaryGradientBoosting(n_estimators=5, random_state=42)
    clf.fit(X, y)
    acc = accuracy_score(y, clf.predict(X))
    assert acc > 0.95


# =====================================================================
# Tests: Reproducibility
# =====================================================================


def test_reproducibility() -> None:
    """Test that random_state ensures reproducible results."""
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)

    clf1 = FastBinaryGradientBoosting(n_estimators=10, random_state=42)
    clf2 = FastBinaryGradientBoosting(n_estimators=10, random_state=42)

    clf1.fit(X, y)
    clf2.fit(X, y)

    pred1 = clf1.predict(X)
    pred2 = clf2.predict(X)
    assert np.array_equal(pred1, pred2)


# =====================================================================
# Tests: Properties
# =====================================================================


def test_properties() -> None:
    """Test property access."""
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)

    clf = FastBinaryGradientBoosting(n_estimators=10, random_state=42)
    clf.fit(X, y)

    assert len(clf.estimator_weights) == 10
    assert np.all(clf.estimator_weights == clf.learning_rate)


# =====================================================================
# Run tests if executed directly
# =====================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
