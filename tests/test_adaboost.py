# testing for adaboost
import numpy as np
import pytest

from src.trees.boosting.adaboost import AdaBoostClassifier, DecisionStump
from src.trees.decision_tree import DecisionTree

@pytest.fixture
def binary_data():
    rng = np.random.RandomState(42)
    X = rng.randn(200, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y

def test_decision_stump_is_subclass():
    stump = DecisionStump()
    assert isinstance(stump, DecisionTree)

def test_decision_stump_depth():
    stump = DecisionStump()
    assert stump.max_depth == 1

def test_adaboost_fit_predict(binary_data):
    X, y = binary_data
    ada = AdaBoostClassifier(n_estimators=10, random_state=42)
    ada.fit(X, y)
    pred = ada.predict(X)
    assert pred.shape == y.shape
    assert set(pred.tolist()).issubset({0, 1})

def test_adaboost_predict_proba(binary_data):
    X, y = binary_data
    ada = AdaBoostClassifier(n_estimators=10, random_state=42)
    ada.fit(X, y)
    proba = ada.predict_proba(X)
    assert proba.shape == (len(X), 2)
    np.testing.assert_almost_equal(proba.sum(axis=1), np.ones(len(X)))

def test_estimator_weights_property(binary_data):
    X, y = binary_data
    ada = AdaBoostClassifier(n_estimators=10, random_state=42)
    ada.fit(X, y)
    assert len(ada.estimator_weights) == len(ada.estimators_)
    assert np.all(ada.estimator_weights > 0)

def test_estimator_errors_property(binary_data):
    X, y = binary_data
    ada = AdaBoostClassifier(n_estimators=10, random_state=42)
    ada.fit(X, y)
    assert len(ada.estimator_errors) == len(ada.estimators_)
    assert np.all(ada.estimator_errors >= 0)

def test_staged_predict(binary_data):
    X, y = binary_data
    ada = AdaBoostClassifier(n_estimators=10, random_state=42)
    ada.fit(X, y)
    stages = list(ada.staged_predict(X))
    assert len(stages) == len(ada.estimators_)
    for stage_pred in stages:
        assert stage_pred.shape == y.shape

def test_n_estimators_1(binary_data):
    X, y = binary_data
    ada = AdaBoostClassifier(n_estimators=1, random_state=42)
    ada.fit(X, y)
    pred = ada.predict(X)
    assert pred.shape == y.shape

def test_n_estimators_200(binary_data):
    X, y = binary_data
    ada = AdaBoostClassifier(n_estimators=200, random_state=42)
    ada.fit(X, y)
    pred = ada.predict(X)
    assert pred.shape == y.shape
    acc = (pred == y).mean()
    assert acc > 0.7

def test_learning_rate_effect(binary_data):
    X, y = binary_data
    ada1 = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=42)
    ada1.fit(X, y)
    pred1 = ada1.predict(X)
    ada2 = AdaBoostClassifier(n_estimators=50, learning_rate=0.1, random_state=42)
    ada2.fit(X, y)
    pred2 = ada2.predict(X)

    assert not np.array_equal(pred1, pred2)

def test_against_sklearn(binary_data):
    from sklearn.ensemble import AdaBoostClassifier as SkAda
    from sklearn.tree import DecisionTreeClassifier
    X, y = binary_data
    ada = AdaBoostClassifier(n_estimators=50, random_state=42)
    ada.fit(X, y)
    pred = ada.predict(X)
    base = DecisionTreeClassifier(max_depth=1, random_state=42)
    sk_ada = SkAda(n_estimators=50, estimator=base, random_state=42)
    sk_ada.fit(X, y)
    sk_pred = sk_ada.predict(X)

    our_acc = (pred == y).mean()
    sk_acc = (sk_pred == y).mean()
    diff_pct = abs(our_acc - sk_acc) / max(sk_acc, 1e-10) * 100
    assert diff_pct <= 5.0, f"Difference too large: {diff_pct:.2f}%"

def test_stump_predict(binary_data):
    X, y = binary_data
    stump = DecisionStump(random_state=42)
    stump.fit(X, y)
    pred = stump.predict(X)
    assert pred.shape == y.shape

def test_stump_predict_proba(binary_data):
    X, y = binary_data
    stump = DecisionStump(random_state=42)
    stump.fit(X, y)
    proba = stump.predict_proba(X)
    assert proba.shape == (len(X), 2)

def test_repr():
    ada = AdaBoostClassifier(n_estimators=10)
    assert "AdaBoostClassifier" in repr(ada)

def test_fit_returns_self(binary_data):
    X, y = binary_data
    ada = AdaBoostClassifier(random_state=42)
    returned = ada.fit(X, y)
    assert returned is ada
