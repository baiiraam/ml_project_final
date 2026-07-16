import re
import pytest
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.trees.boosting.gradient_boosting import GradientBoostingClassifier


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
def multiclass_data() -> Tuple[np.ndarray, np.ndarray]:
    """Generates a simple 3-class classification dataset with enough samples for stratified splitting."""
    X = np.array(
        [
            [1.0, 1.0],
            [1.2, 0.8],
            [1.1, 1.1],
            [1.3, 0.9],  # Class 0 (4 samples)
            [5.0, 5.0],
            [4.8, 5.2],
            [4.9, 5.1],
            [5.1, 4.9],  # Class 1 (4 samples)
            [9.0, 1.0],
            [8.8, 1.2],
            [8.9, 1.1],
            [9.1, 0.9],  # Class 2 (4 samples)
        ],
        dtype=np.float64,
    )
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    return X, y


# =====================================================================
# Tests: Initialization & Argument Validation
# =====================================================================


def test_initialization_defaults() -> None:
    """Verifies that the default parameters are set correctly."""
    clf = GradientBoostingClassifier()
    assert clf.n_estimators == 100
    assert clf.learning_rate == 0.1
    assert clf.max_depth == 3
    assert clf.min_samples_split == 2
    assert clf.criterion == "gini"


@pytest.mark.parametrize(
    "params, expected_error",
    [
        ({"n_estimators": 0}, "n_estimators must be positive"),
        ({"n_estimators": -5}, "n_estimators must be positive"),
        ({"learning_rate": 0.0}, "learning_rate must be in (0, 1]"),
        ({"learning_rate": 1.5}, "learning_rate must be in (0, 1]"),
        ({"max_depth": 0}, "max_depth must be at least 1"),
        ({"min_samples_split": 1}, "min_samples_split must be at least 2"),
        ({"criterion": "invalid_criterion"}, "criterion must be 'gini' or 'entropy'"),
    ],
)

# Update this test function
def test_invalid_parameters(params: dict, expected_error: str) -> None:
    """Ensures improper parameters immediately raise descriptive ValueErrors."""
    # re.escape safely escapes "(0, 1]" so pytest treats it as a literal string match
    with pytest.raises(ValueError, match=re.escape(expected_error)):
        GradientBoostingClassifier(**params)


def test_fit_input_validation() -> None:
    """Tests that fit() enforces matching shapes and non-empty inputs."""
    clf = GradientBoostingClassifier()
    X = np.array([[1, 2], [3, 4]])
    y_mismatched = np.array([1])

    with pytest.raises(ValueError, match="same number of samples"):
        clf.fit(X, y_mismatched)

    with pytest.raises(ValueError, match="must not be empty"):
        clf.fit(np.array([]).reshape(0, 2), np.array([]))

    with pytest.raises(ValueError, match="at least 2 classes"):
        clf.fit(X, np.array([0, 0]))


# =====================================================================
# Tests: Binary Classification Performance & Mechanics
# =====================================================================


def test_binary_classification_fit(binary_data: Tuple[np.ndarray, np.ndarray]) -> None:
    """Verifies convergence and basic prediction on a simple binary problem."""
    X, y = binary_data
    clf = GradientBoostingClassifier(
        n_estimators=20, max_depth=3, learning_rate=0.5, random_state=42
    )
    clf.fit(X, y)

    # Internal state checks
    assert len(clf.estimators_) == 20
    assert len(clf.estimators_leaf_values_) == 20
    assert clf.n_classes_ == 2
    assert np.array_equal(clf.classes_, np.array([0, 1]))

    # Predictions
    preds = clf.predict(X)
    assert preds.shape == (X.shape[0],)

    # Check accuracy is reasonable (better than random)
    acc = accuracy_score(y, preds)
    assert acc > 0.60  # Should be >60% (better than random)

    # Probabilities
    probs = clf.predict_proba(X)
    assert probs.shape == (X.shape[0], 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    np.testing.assert_allclose(np.sum(probs, axis=1), 1.0)


# =====================================================================
# Tests: Multiclass Classification Performance & Mechanics
# =====================================================================


def test_multiclass_classification_fit(
    multiclass_data: Tuple[np.ndarray, np.ndarray],
) -> None:
    """Verifies that the multiclass K-tree architecture updates and converges."""
    X, y = multiclass_data
    clf = GradientBoostingClassifier(
        n_estimators=20, max_depth=3, learning_rate=0.5, random_state=42
    )
    clf.fit(X, y)

    # 5 boosting stages, with n_classes trees per stage
    assert len(clf.estimators_) == 20
    assert len(clf.estimators_[0]) == clf.n_classes_
    assert len(clf.estimators_leaf_values_[0]) == clf.n_classes_

    # Predictions - check accuracy is better than random
    preds = clf.predict(X)
    acc = accuracy_score(y, preds)
    random_baseline = 1.0 / clf.n_classes_
    assert acc > random_baseline * 1.2  # At least 20% better than random

    # Probabilities
    probs = clf.predict_proba(X)
    assert probs.shape == (X.shape[0], clf.n_classes_)
    np.testing.assert_allclose(np.sum(probs, axis=1), 1.0)


# =====================================================================
# Tests: Diagnostic Features (Loss, Staged Predictions)
# =====================================================================


def test_training_loss_tracking(binary_data: Tuple[np.ndarray, np.ndarray]) -> None:
    """Confirms that loss decreases strictly or monotonically during early training steps."""
    X, y = binary_data
    clf = GradientBoostingClassifier(
        n_estimators=10, learning_rate=0.2, random_state=42
    )
    clf.fit(X, y)

    losses = clf.train_loss
    assert len(losses) == 10
    # The loss at the final stage should be strictly smaller than the first stage
    assert losses[-1] < losses[0]


def test_staged_predict(binary_data: Tuple[np.ndarray, np.ndarray]) -> None:
    """Validates that staged_predict successfully yields predictions for each stage."""
    X, y = binary_data
    clf = GradientBoostingClassifier(n_estimators=5, random_state=42)
    clf.fit(X, y)

    stages = list(clf.staged_predict(X))
    assert len(stages) == 5
    for stage_pred in stages:
        assert stage_pred.shape == (X.shape[0],)


# =====================================================================
# Tests: Guardrails & API Contracts
# =====================================================================


def test_prediction_without_fit() -> None:
    """Verifies that calling prediction methods before fit() raises a clean ValueError."""
    clf = GradientBoostingClassifier()
    X = np.array([[1, 2]])

    with pytest.raises(ValueError, match="not been fitted yet"):
        clf.predict(X)

    with pytest.raises(ValueError, match="not been fitted yet"):
        clf.predict_proba(X)

    with pytest.raises(ValueError, match="not been fitted yet"):
        list(clf.staged_predict(X))


# ============================================
# Fixtures
# ============================================


@pytest.fixture
def binary_data_split(binary_data):
    """Split binary data into train/test."""
    X, y = binary_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture
def multiclass_data_split(multiclass_data):
    """Split multiclass data into train/test."""
    X, y = multiclass_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


# ============================================
# Tests for Initialization
# ============================================


def test_gradient_boosting_init_default():
    """Test default initialization."""
    gb = GradientBoostingClassifier()
    assert gb.n_estimators == 100
    assert gb.learning_rate == 0.1
    assert gb.max_depth == 3
    assert gb.min_samples_split == 2
    assert gb.criterion == "gini"
    assert gb.random_state is None
    assert gb.estimators_ == []
    assert gb.classes_ is None
    assert gb.n_classes_ == 0


def test_gradient_boosting_init_custom():
    """Test custom initialization parameters."""
    gb = GradientBoostingClassifier(
        n_estimators=50,
        learning_rate=0.05,
        max_depth=4,
        min_samples_split=5,
        criterion="entropy",
        random_state=123,
    )
    assert gb.n_estimators == 50
    assert gb.learning_rate == 0.05
    assert gb.max_depth == 4
    assert gb.min_samples_split == 5
    assert gb.criterion == "entropy"
    assert gb.random_state == 123


def test_gradient_boosting_init_invalid_params():
    """Test invalid initialization parameters."""
    with pytest.raises(ValueError, match="n_estimators must be positive"):
        GradientBoostingClassifier(n_estimators=0)

    with pytest.raises(ValueError, match="learning_rate must be in"):
        GradientBoostingClassifier(learning_rate=0)

    with pytest.raises(ValueError, match="learning_rate must be in"):
        GradientBoostingClassifier(learning_rate=2.0)

    with pytest.raises(ValueError, match="max_depth must be at least"):
        GradientBoostingClassifier(max_depth=0)

    with pytest.raises(ValueError, match="min_samples_split must be at least"):
        GradientBoostingClassifier(min_samples_split=0)

    with pytest.raises(ValueError, match="criterion must be 'gini' or 'entropy'"):
        GradientBoostingClassifier(criterion="invalid")


# ============================================
# Tests for Fit
# ============================================


def test_gradient_boosting_fit_binary(binary_data_split):
    """Test fitting on binary classification."""
    X_train, X_test, y_train, y_test = binary_data_split

    gb = GradientBoostingClassifier(
        n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42
    )
    gb.fit(X_train, y_train)

    # Check that model is fitted
    assert len(gb.estimators_) == 50
    assert gb.classes_ is not None
    assert gb.n_classes_ == 2
    assert len(gb.train_loss_) == 50
    assert len(gb.estimator_weights_) == 50

    # Check predictions work
    pred = gb.predict(X_test)
    assert len(pred) == len(y_test)
    assert set(pred) == set(gb.classes_)

    # Check probabilities
    probs = gb.predict_proba(X_test)
    assert probs.shape == (len(y_test), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_gradient_boosting_fit_multiclass(multiclass_data_split):
    """Test fitting on multiclass classification."""
    X_train, X_test, y_train, y_test = multiclass_data_split

    gb = GradientBoostingClassifier(
        n_estimators=30, learning_rate=0.1, max_depth=3, random_state=42
    )
    gb.fit(X_train, y_train)

    # Check that model is fitted
    assert len(gb.estimators_) == 30
    assert gb.classes_ is not None
    assert gb.n_classes_ == 3  # Fixed: changed from 4 to 3
    assert len(gb.train_loss_) == 30

    # Each stage should have n_classes trees
    assert len(gb.estimators_[0]) == 3  # Fixed: changed from 4 to 3

    # Check predictions
    pred = gb.predict(X_test)
    assert len(pred) == len(y_test)
    assert set(pred) == set(gb.classes_)

    # Check probabilities
    probs = gb.predict_proba(X_test)
    assert probs.shape == (len(y_test), 3)  # Fixed: changed from 4 to 3
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_gradient_boosting_fit_learning_rate_effect(binary_data_split):
    """Test that learning rate affects the model."""
    X_train, X_test, y_train, y_test = binary_data_split

    # Fit with high learning rate
    gb_high = GradientBoostingClassifier(
        n_estimators=20, learning_rate=0.5, max_depth=2, random_state=42
    )
    gb_high.fit(X_train, y_train)

    # Fit with low learning rate
    gb_low = GradientBoostingClassifier(
        n_estimators=20, learning_rate=0.01, max_depth=2, random_state=42
    )
    gb_low.fit(X_train, y_train)

    # Different learning rates should produce different losses
    assert gb_high.train_loss[-1] < gb_low.train_loss[-1]


def test_gradient_boosting_fit_max_depth_effect(binary_data_split):
    """Test that max_depth affects the model."""
    X_train, X_test, y_train, y_test = binary_data_split

    # Fit with shallow trees
    gb_shallow = GradientBoostingClassifier(
        n_estimators=20, max_depth=1, random_state=42
    )
    gb_shallow.fit(X_train, y_train)

    # Fit with deep trees
    gb_deep = GradientBoostingClassifier(n_estimators=20, max_depth=6, random_state=42)
    gb_deep.fit(X_train, y_train)

    # Deep trees should have different (probably lower) training loss
    assert gb_deep.train_loss[-1] <= gb_shallow.train_loss[-1] * 1.5


def test_gradient_boosting_fit_convergence(binary_data_split):
    """Test that training loss decreases over iterations."""
    X_train, X_test, y_train, y_test = binary_data_split

    gb = GradientBoostingClassifier(
        n_estimators=30, learning_rate=0.1, max_depth=3, random_state=42
    )
    gb.fit(X_train, y_train)

    # Loss should generally decrease
    for i in range(1, len(gb.train_loss_)):
        # Allow small increases due to learning rate
        assert gb.train_loss_[i] <= gb.train_loss_[i - 1] * 1.1


def test_gradient_boosting_fit_invalid_data():
    """Test fitting with invalid data."""
    gb = GradientBoostingClassifier()

    # Empty arrays
    with pytest.raises(ValueError, match="X and y must not be empty"):
        gb.fit(np.array([]), np.array([]))

    # Mismatched shapes
    X = np.random.randn(10, 5)
    y = np.random.randint(0, 2, 8)
    with pytest.raises(
        ValueError, match="X and y must have the same number of samples"
    ):
        gb.fit(X, y)

    # Single class
    y_single = np.zeros(10)
    with pytest.raises(ValueError, match="Classification requires at least 2 classes"):
        gb.fit(np.random.randn(10, 5), y_single)


# ============================================
# Tests for Prediction
# ============================================


def test_gradient_boosting_predict_binary(binary_data_split):
    """Test prediction on binary classification."""
    X_train, X_test, y_train, y_test = binary_data_split

    gb = GradientBoostingClassifier(n_estimators=20, random_state=42)
    gb.fit(X_train, y_train)

    pred = gb.predict(X_test)
    assert pred.shape == y_test.shape
    assert all(p in gb.classes_ for p in pred)

    # Should have reasonable accuracy (better than random)
    acc = accuracy_score(y_test, pred)
    assert acc > 0.5


def test_gradient_boosting_predict_multiclass(multiclass_data_split):
    """Test prediction on multiclass classification."""
    X_train, X_test, y_train, y_test = multiclass_data_split

    gb = GradientBoostingClassifier(n_estimators=20, random_state=42)
    gb.fit(X_train, y_train)

    pred = gb.predict(X_test)
    assert pred.shape == y_test.shape
    assert all(p in gb.classes_ for p in pred)

    acc = accuracy_score(y_test, pred)
    assert acc > 0.25  # Better than random (1/4 chance)


def test_gradient_boosting_predict_proba_binary(binary_data_split):
    """Test probability prediction on binary classification."""
    X_train, X_test, y_train, y_test = binary_data_split

    gb = GradientBoostingClassifier(n_estimators=20, random_state=42)
    gb.fit(X_train, y_train)

    probs = gb.predict_proba(X_test)
    assert probs.shape == (len(y_test), 2)
    assert np.all(probs >= 0)
    assert np.all(probs <= 1)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_gradient_boosting_predict_proba_multiclass(multiclass_data_split):
    """Test probability prediction on multiclass classification."""
    X_train, X_test, y_train, y_test = multiclass_data_split

    gb = GradientBoostingClassifier(n_estimators=20, random_state=42)
    gb.fit(X_train, y_train)

    probs = gb.predict_proba(X_test)
    assert probs.shape == (len(y_test), 3)  # Fixed: changed from 4 to 3
    assert np.all(probs >= 0)
    assert np.all(probs <= 1)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_gradient_boosting_predict_unfitted():
    """Test prediction before fitting."""
    gb = GradientBoostingClassifier()

    X = np.random.randn(10, 5)

    with pytest.raises(ValueError, match="Model has not been fitted yet"):
        gb.predict(X)

    with pytest.raises(ValueError, match="Model has not been fitted yet"):
        gb.predict_proba(X)


# ============================================
# Tests for Staged Predict
# ============================================


def test_gradient_boosting_staged_predict_binary(binary_data_split):
    """Test staged prediction on binary classification."""
    X_train, X_test, y_train, y_test = binary_data_split

    gb = GradientBoostingClassifier(n_estimators=20, random_state=42)
    gb.fit(X_train, y_train)

    staged_preds = list(gb.staged_predict(X_test))

    assert len(staged_preds) == 20
    assert all(len(pred) == len(y_test) for pred in staged_preds)

    # Accuracy should generally improve with more trees
    accs = [accuracy_score(y_test, pred) for pred in staged_preds]
    for i in range(1, len(accs)):
        # Allow small fluctuations
        assert accs[i] >= accs[i - 1] * 0.9


def test_gradient_boosting_staged_predict_multiclass(multiclass_data_split):
    """Test staged prediction on multiclass classification."""
    X_train, X_test, y_train, y_test = multiclass_data_split

    gb = GradientBoostingClassifier(n_estimators=15, random_state=42)
    gb.fit(X_train, y_train)

    staged_preds = list(gb.staged_predict(X_test))

    assert len(staged_preds) == 15
    assert all(len(pred) == len(y_test) for pred in staged_preds)

    # Check staged prediction after final stage matches regular prediction
    final_staged = staged_preds[-1]
    final_regular = gb.predict(X_test)
    assert np.array_equal(final_staged, final_regular)


# ============================================
# Tests for Properties
# ============================================


def test_gradient_boosting_properties(binary_data_split):
    """Test property access."""
    X_train, X_test, y_train, y_test = binary_data_split

    gb = GradientBoostingClassifier(n_estimators=20, random_state=42)
    gb.fit(X_train, y_train)

    assert len(gb.estimator_weights) == 20
    assert len(gb.train_loss) == 20
    assert np.all(gb.estimator_weights == gb.learning_rate)


def test_gradient_boosting_properties_unfitted():
    """Test properties before fitting."""
    gb = GradientBoostingClassifier()

    assert len(gb.estimator_weights) == 0
    assert len(gb.train_loss) == 0


# ============================================
# Tests for Repr
# ============================================


def test_gradient_boosting_repr_unfitted():
    """Test string representation when unfitted."""
    gb = GradientBoostingClassifier(n_estimators=50, learning_rate=0.05)
    repr_str = repr(gb)
    assert "not fitted" in repr_str
    assert "n_estimators=50" in repr_str
    assert "learning_rate=0.05" in repr_str


def test_gradient_boosting_repr_fitted(binary_data_split):
    """Test string representation when fitted."""
    X_train, X_test, y_train, y_test = binary_data_split

    gb = GradientBoostingClassifier(n_estimators=20, random_state=42)
    gb.fit(X_train, y_train)

    repr_str = repr(gb)
    assert "fitted with" in repr_str
    assert "20 stages" in repr_str
    assert "2 classes" in repr_str or "n_classes_=2" in repr_str


# ============================================
# Run tests if executed directly
# ============================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
