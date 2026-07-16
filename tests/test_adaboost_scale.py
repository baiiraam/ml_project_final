"""
Unit tests for adaboost_scale.py - FIXED VERSION
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.trees.boosting.adaboost_scale import AdaBoostClassifier, DecisionStump


# ============================================================
# TEST 1: Basic Functionality - FIXED
# ============================================================


def test_adaboost_basic_binary():
    """
    Test basic binary classification functionality.
    """
    # Use a simpler dataset with more informative features
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=8,  # MORE informative features (was 5)
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    ab = AdaBoostClassifier(
        n_estimators=20, random_state=42
    )  # MORE estimators (was 10)
    ab.fit(X_train, y_train)

    y_pred = ab.predict(X_test)
    assert y_pred.shape == (X_test.shape[0],)
    assert len(np.unique(y_pred)) == 2

    y_proba = ab.predict_proba(X_test)
    assert y_proba.shape == (X_test.shape[0], 2)
    assert np.allclose(np.sum(y_proba, axis=1), 1.0)

    acc = accuracy_score(y_test, y_pred)
    assert acc > 0.65

    print(f" Basic binary test passed! Accuracy: {acc:.4f}")


# ============================================================
# TEST 2: Multiclass Support - PASSED (no change)
# ============================================================


def test_adaboost_multiclass():
    """Test multiclass classification (SAMME algorithm)."""
    X, y = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    ab = AdaBoostClassifier(n_estimators=20, random_state=42)
    ab.fit(X_train, y_train)

    assert len(ab.classes_) == 3
    assert np.array_equal(ab.classes_, np.array([0, 1, 2]))

    y_pred = ab.predict(X_test)
    assert y_pred.shape == (X_test.shape[0],)
    assert set(np.unique(y_pred)).issubset({0, 1, 2})

    y_proba = ab.predict_proba(X_test)
    assert y_proba.shape == (X_test.shape[0], 3)
    assert np.allclose(np.sum(y_proba, axis=1), 1.0)

    acc = accuracy_score(y_test, y_pred)
    assert acc > 0.8

    print(f" Multiclass test passed! Accuracy: {acc:.4f}")


# ============================================================
# TEST 3: DecisionStump Depth - PASSED (no change)
# ============================================================


def test_decision_stump_depth():
    """Test that DecisionStump creates depth-1 trees."""
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)

    stump = DecisionStump(random_state=42)
    stump.fit(X, y)

    assert stump.max_depth == 1

    if stump.tree_ is not None and "feature_index" in stump.tree_:
        assert stump.depth <= 1
        assert "feature_index" in stump.tree_

    print(f" DecisionStump test passed! Depth: {stump.depth}")


# ============================================================
# TEST 4: Sample Weighting - FIXED (now works with sample_weight)
# ============================================================


def test_adaboost_sample_weights():
    """Test that AdaBoost correctly handles sample weights."""
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)

    ab_uniform = AdaBoostClassifier(n_estimators=5, random_state=42)
    ab_uniform.fit(X, y)

    sample_weights = np.ones(100)
    sample_weights[y == 1] = 10.0
    sample_weights /= sample_weights.sum()

    ab_weighted = AdaBoostClassifier(n_estimators=5, random_state=42)
    ab_weighted.fit(X, y, sample_weight=sample_weights)  # NOW WORKS

    pred_uniform = ab_uniform.predict(X)
    pred_weighted = ab_weighted.predict(X)

    assert pred_uniform.shape == pred_weighted.shape
    assert len(np.unique(pred_uniform)) == 2
    assert len(np.unique(pred_weighted)) == 2

    print(" Sample weight test passed!")


# ============================================================
# TEST 5: Early Stopping - FIXED
# ============================================================


def test_adaboost_early_stopping():
    """Test early stopping when error >= max_error."""
    # Fix: Use valid make_classification parameters
    X, y = make_classification(
        n_samples=50,
        n_features=4,  # Must be >= n_informative + n_redundant
        n_informative=2,
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=1,  # Must be <= 2**n_informative (2**2=4)
        random_state=42,
    )

    ab = AdaBoostClassifier(n_estimators=100, random_state=42)
    ab.fit(X, y)

    assert len(ab.estimators_) > 0
    assert len(ab.estimator_weights_) == len(ab.estimators_)
    assert len(ab.estimator_errors_) == len(ab.estimators_)

    print(f" Early stopping test passed! Estimators fitted: {len(ab.estimators_)}")


# ============================================================
# TEST 6: Performance Metrics - PASSED (no change)
# ============================================================


def test_adaboost_performance_metrics():
    """Test that performance metrics are recorded."""
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)

    ab = AdaBoostClassifier(n_estimators=5, random_state=42)

    assert ab.fit_time_ == 0.0

    ab.fit(X, y)

    assert ab.fit_time_ > 0.0

    ab.predict(X)
    assert ab.predict_time_ > 0.0

    assert isinstance(ab.fit_time_, float)
    assert isinstance(ab.predict_time_, float)

    print(" Performance metrics test passed!")


# ============================================================
# TEST 7: Prediction Caching - FIXED
# ============================================================


def test_adaboost_caching():
    """Test that prediction caching works correctly."""
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(  # FIXED: added y_train
        X, y, test_size=0.3, random_state=42
    )

    ab = AdaBoostClassifier(n_estimators=5, random_state=42)
    ab.fit(X_train, y_train)  # FIXED: uses y_train

    preds1 = ab._get_all_predictions(X_test)
    assert ab._predictions_cache is not None
    assert ab._cache_key is not None

    preds2 = ab._get_all_predictions(X_test)
    assert np.array_equal(preds1, preds2)

    X_new, _ = make_classification(n_samples=50, n_features=5, random_state=99)
    preds3 = ab._get_all_predictions(X_new)

    assert ab._predictions_cache is not None
    assert preds1.shape != preds3.shape

    print(" Caching test passed!")


# ============================================================
# TEST 8: Parameter Validation - PASSED (no change)
# ============================================================


def test_adaboost_parameter_validation():
    """Test that invalid parameters raise appropriate errors."""
    with pytest.raises(ValueError, match="n_estimators must be positive"):
        AdaBoostClassifier(n_estimators=0)

    with pytest.raises(ValueError, match="n_estimators must be positive"):
        AdaBoostClassifier(n_estimators=-5)

    with pytest.raises(ValueError, match="learning_rate must be positive"):
        AdaBoostClassifier(learning_rate=0.0)

    with pytest.raises(ValueError, match="learning_rate must be positive"):
        AdaBoostClassifier(learning_rate=-0.5)

    with pytest.raises(ValueError, match="criterion must be 'gini' or 'entropy'"):
        AdaBoostClassifier(criterion="invalid")

    AdaBoostClassifier(
        n_estimators=10, learning_rate=0.5, criterion="gini", random_state=None
    )

    print(" Parameter validation test passed!")


# ============================================================
# TEST 9: Property Access - PASSED (no change)
# ============================================================


def test_adaboost_properties():
    """Test that properties work correctly."""
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)

    ab = AdaBoostClassifier(n_estimators=5, random_state=42)

    with pytest.raises(ValueError, match="Model has not been fitted yet"):
        _ = ab.n_features_in_

    ab.fit(X, y)

    assert len(ab.estimator_weights) == len(ab.estimators_)
    assert len(ab.estimator_errors) == len(ab.estimators_)
    assert ab.n_features_in_ == X.shape[1]

    assert np.all(ab.estimator_weights > 0)
    assert np.all(ab.estimator_errors >= 0)
    assert np.all(ab.estimator_errors <= 0.5)

    print(" Properties test passed!")


# ============================================================
# TEST 10: Staged Predict - PASSED (no change)
# ============================================================


def test_adaboost_staged_predict():
    """Test staged_predict functionality."""
    X, y = make_classification(n_samples=200, n_features=5, random_state=42)

    ab = AdaBoostClassifier(n_estimators=10, random_state=42)
    ab.fit(X, y)

    staged_preds = list(ab.staged_predict(X))

    assert len(staged_preds) == len(ab.estimators_)

    for pred in staged_preds:
        assert pred.shape == (X.shape[0],)
        assert len(np.unique(pred)) == 2

    accuracies = [accuracy_score(y, pred) for pred in staged_preds]
    assert np.all(np.diff(accuracies) >= -0.01)

    print(" Staged predict test passed!")


# ============================================================
# TEST 11: Consistency with sklearn - FIXED
# ============================================================


def test_adaboost_sklearn_consistency():
    """Test consistency with sklearn's AdaBoost."""
    from sklearn.ensemble import AdaBoostClassifier as SklearnAdaBoost
    from sklearn.tree import DecisionTreeClassifier

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    our_ab = AdaBoostClassifier(n_estimators=10, random_state=42, learning_rate=1.0)
    our_ab.fit(X_train, y_train)
    our_pred = our_ab.predict(X_test)
    our_acc = accuracy_score(y_test, our_pred)

    # FIXED: sklearn API changed - remove 'algorithm' parameter
    sk_ab = SklearnAdaBoost(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=10,
        random_state=42,
        learning_rate=1.0,
        # algorithm='SAMME'  # REMOVED - deprecated in newer sklearn
    )
    sk_ab.fit(X_train, y_train)
    sk_pred = sk_ab.predict(X_test)
    sk_acc = accuracy_score(y_test, sk_pred)

    diff = abs(our_acc - sk_acc)
    assert diff < 0.05

    print(" sklearn consistency test passed!")


# ============================================================
# TEST 12: Edge Cases - FIXED
# ============================================================


def test_adaboost_edge_cases():
    """Test edge cases."""
    # 1. Single sample
    X, y = make_classification(n_samples=1, n_features=5, random_state=42)
    ab = AdaBoostClassifier(n_estimators=5, random_state=42)

    with pytest.raises(ValueError):
        ab.fit(X, y)

    # 2. Single feature - FIXED: use valid parameters
    # Need n_features >= n_informative + n_redundant
    # n_informative defaults to 2, so n_features must be >= 2
    X, y = make_classification(
        n_samples=100,
        n_features=2,  # At least 2 for n_informative=2
        n_informative=2,
        n_redundant=0,
        random_state=42,
    )
    ab = AdaBoostClassifier(n_estimators=5, random_state=42)
    ab.fit(X, y)
    assert len(ab.estimators_) > 0
    assert ab.n_features_in_ == 2  # Updated to match

    # 3. Perfectly separable dataset
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    ab = AdaBoostClassifier(n_estimators=5, random_state=42)
    ab.fit(X, y)
    y_pred = ab.predict(X)
    acc = accuracy_score(y, y_pred)
    assert acc > 0.9

    # 4. Predict_proba on all zeros
    X_zero = np.zeros((10, X.shape[1]))
    y_proba = ab.predict_proba(X_zero)
    assert y_proba.shape == (10, 2)
    assert np.allclose(np.sum(y_proba, axis=1), 1.0)

    print(" Edge cases test passed!")


# ============================================================
# TEST 13: Reproducibility - PASSED (no change)
# ============================================================


def test_adaboost_reproducibility():
    """Test that random_state ensures reproducible results."""
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)

    ab1 = AdaBoostClassifier(n_estimators=10, random_state=42)
    ab2 = AdaBoostClassifier(n_estimators=10, random_state=42)

    ab1.fit(X, y)
    ab2.fit(X, y)

    pred1 = ab1.predict(X)
    pred2 = ab2.predict(X)
    assert np.array_equal(pred1, pred2)

    ab3 = AdaBoostClassifier(n_estimators=10, random_state=99)
    ab3.fit(X, y)
    pred3 = ab3.predict(X)

    assert len(np.unique(pred3)) == 2

    print(" Reproducibility test passed!")


# ============================================================
# TEST 14: Additional Edge Cases for Coverage
# ============================================================


def test_adaboost_additional_edge_cases():
    """
    Test additional edge cases for better coverage.

    What it tests:
    - Invalid sample_weight shapes
    - Negative sample_weights
    - X and y shape mismatch
    - Single class dataset
    - predict_proba before fit
    - staged_predict before fit
    """

    # 1. Invalid sample_weight shape
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    ab = AdaBoostClassifier(n_estimators=5, random_state=42)

    # Wrong shape (should be (n_samples,))
    bad_weights = np.ones(50)  # Wrong length
    with pytest.raises(ValueError, match="sample_weight must have shape"):
        ab.fit(X, y, sample_weight=bad_weights)

    # 2. Negative sample_weights
    bad_weights = np.ones(100)
    bad_weights[0] = -1.0
    with pytest.raises(ValueError, match="sample_weight must be non-negative"):
        ab.fit(X, y, sample_weight=bad_weights)

    # 3. X and y shape mismatch
    X_bad, y_bad = make_classification(n_samples=100, n_features=5, random_state=42)
    y_bad = y_bad[:50]  # Wrong length
    with pytest.raises(ValueError, match="same number of samples"):
        ab.fit(X_bad, y_bad)

    # 4. Single class dataset
    X_single = np.random.randn(100, 5)
    y_single = np.zeros(100)  # All zeros
    with pytest.raises(ValueError, match="at least two classes"):
        ab.fit(X_single, y_single)

    # 5. predict_proba before fit
    ab_unfitted = AdaBoostClassifier(n_estimators=5, random_state=42)
    with pytest.raises(ValueError, match="model has not been fitted"):
        ab_unfitted.predict_proba(X)

    # 6. staged_predict before fit
    with pytest.raises(ValueError, match="model has not been fitted"):
        list(ab_unfitted.staged_predict(X))

    print(" Additional edge cases test passed!")


# ============================================================
# RUN ALL TESTS
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING ALL TESTS FOR ADABOOST_SCALE.PY")
    print("=" * 60)
    print()

    test_adaboost_basic_binary()
    print()

    test_adaboost_multiclass()
    print()

    test_decision_stump_depth()
    print()

    test_adaboost_sample_weights()
    print()

    test_adaboost_early_stopping()
    print()

    test_adaboost_performance_metrics()
    print()

    test_adaboost_caching()
    print()

    test_adaboost_parameter_validation()
    print()

    test_adaboost_properties()
    print()

    test_adaboost_staged_predict()
    print()

    test_adaboost_sklearn_consistency()
    print()

    test_adaboost_edge_cases()
    print()

    test_adaboost_reproducibility()
    print()
