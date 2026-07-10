import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.trees.bagging.random_forest import RandomForestClassifier


# ==========================================================
# Basic fit test
# ==========================================================


def test_fit():

    X, y = make_classification(
        n_samples=200,
        n_features=5,
        n_classes=3,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )

    rf = RandomForestClassifier(n_estimators=10, random_state=42)

    rf.fit(X, y)

    assert len(rf.estimators_) == 10
    assert rf.n_features_ == 5
    assert rf.n_classes_ == 3


# ==========================================================
# Predict test
# ==========================================================


def test_predict():

    X, y = make_classification(n_samples=200, n_features=5, random_state=42)

    rf = RandomForestClassifier(n_estimators=10, random_state=42)

    rf.fit(X, y)

    pred = rf.predict(X)

    assert pred.shape == y.shape
    assert set(pred).issubset(set(y))


# ==========================================================
# Predict probability test
# ==========================================================


def test_predict_proba():

    X, y = make_classification(n_samples=100, n_features=5, random_state=42)

    rf = RandomForestClassifier(n_estimators=5, random_state=42)

    rf.fit(X, y)

    proba = rf.predict_proba(X[:10])

    assert proba.shape == (10, 2)

    assert np.allclose(proba.sum(axis=1), 1.0)


# ==========================================================
# Feature importance
# ==========================================================


def test_feature_importances():

    X, y = make_classification(n_samples=100, n_features=5, random_state=42)

    rf = RandomForestClassifier(n_estimators=5)

    rf.fit(X, y)

    imp = rf.feature_importances_

    assert len(imp) == 5

    assert np.isclose(imp.sum(), 1.0)


# ==========================================================
# OOB score
# ==========================================================


def test_oob_score():

    X, y = make_classification(n_samples=300, n_features=5, random_state=42)

    rf = RandomForestClassifier(
        n_estimators=20, bootstrap=True, oob_score=True, random_state=42
    )

    rf.fit(X, y)

    score = rf.oob_score_

    assert 0 <= score <= 1


# ==========================================================
# Predict before fit
# ==========================================================


def test_predict_before_fit():

    rf = RandomForestClassifier()

    with pytest.raises(Exception):
        rf.predict(np.array([[1, 2, 3]]))


# ==========================================================
# Aggregate feature importance internal
# ==========================================================


def test_aggregate_feature_importances():

    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    rf = RandomForestClassifier(n_estimators=5)
    rf.fit(X, y)
    result = rf._aggregate_feature_importances()
    assert result.shape == (5,)
    assert np.isclose(result.sum(), 1.0)


# ==========================================================
# aligned_proba
# ==========================================================
def test_aligned_proba():
    X, y = make_classification(
        n_samples=200,
        n_features=5,
        n_classes=3,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )

    rf = RandomForestClassifier(n_estimators=5, random_state=42)

    rf.fit(X, y)
    tree = rf.estimators_[0]

    proba = rf._aligned_proba(tree, X[:10])

    assert proba.shape == (10, 3)
    assert np.allclose(proba.sum(axis=1), 1)


# ==========================================================
# Internal OOB calculation
# ==========================================================
def test_compute_oob_score_internal():

    X, y = make_classification(n_samples=300, n_features=5, random_state=42)

    rf = RandomForestClassifier(
        n_estimators=20, bootstrap=True, oob_score=True, random_state=42
    )
    rf.fit(X, y)

    masks = []
    for i in range(len(rf.estimators_)):
        mask = np.ones(len(X), dtype=bool)
        masks.append(mask)

    score = rf._compute_oob_score(X, y, masks)
    assert isinstance(score, float)


# ==========================================================
# _fitted_classes property
# ==========================================================


def test_fitted_classes_error():
    rf = RandomForestClassifier()
    with pytest.raises(RuntimeError):
        rf._fitted_classes


def test_fitted_classes():
    X, y = make_classification(n_samples=50, n_features=4, random_state=42)
    rf = RandomForestClassifier(n_estimators=2)
    rf.fit(X, y)
    classes = rf._fitted_classes
    assert len(classes) == 2


# ==========================================================
# OOB property errors
# ==========================================================


def test_oob_score_property_error():
    rf = RandomForestClassifier(oob_score=False)

    with pytest.raises(AttributeError):
        rf.oob_score_


def test_feature_importance_before_fit():

    rf = RandomForestClassifier()
    with pytest.raises(AttributeError):
        rf.feature_importances_


# ==========================================================
# repr
# ==========================================================


def test_repr():
    rf = RandomForestClassifier(n_estimators=3)

    text = repr(rf)
    assert "not fitted" in text

    X, y = make_classification(n_samples=50, n_features=4, random_state=42)
    rf.fit(X, y)
    text = repr(rf)
    assert "n_trees=3" in text


# ==========================================================
# multiprocessing
# ==========================================================
def test_parallel_training():
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)

    rf = RandomForestClassifier(n_estimators=4, n_jobs=2, random_state=42)
    rf.fit(X, y)

    assert len(rf.estimators_) == 4


# ==========================================================
# bootstrap false
# ==========================================================


def test_bootstrap_false():

    X, y = make_classification(n_samples=100, n_features=5, random_state=42)

    rf = RandomForestClassifier(n_estimators=5, bootstrap=False)
    rf.fit(X, y)
    assert len(rf.estimators_) == 5


# ==========================================================
# max depth
# ==========================================================


def test_max_depth():
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)

    rf = RandomForestClassifier(n_estimators=5, max_depth=2)

    rf.fit(X, y)
    for tree in rf.estimators_:
        assert tree.depth <= 2


# ==========================================================
# predict classes
# ==========================================================


def test_predict_classes():

    X, y = make_classification(n_samples=100, n_features=5, random_state=42)

    rf = RandomForestClassifier(n_estimators=5)
    rf.fit(X, y)
    pred = rf.predict(X)
    assert len(pred) == 100
    assert set(pred).issubset(set(y))
