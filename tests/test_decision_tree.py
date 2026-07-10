# Tests – placeholder stub
import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier
from src.trees.decision_tree import DecisionTree

@pytest.fixture
def xor_data():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 2)
    y = ((X[:, 0] > 0) != (X[:, 1] > 0)).astype(int)
    return X, y

@pytest.fixture
def simple_data():
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0, 0, 1, 1, 1])
    return X, y

def test_predict_basic(simple_data):
    X, y = simple_data
    dt = DecisionTree(max_depth=3, random_state=42)
    dt.fit(X, y)
    pred = dt.predict(X)
    assert pred.shape == y.shape
    assert set(pred.tolist()).issubset({0, 1})

def test_predict_proba(simple_data):
    X, y = simple_data
    dt = DecisionTree(max_depth=3, random_state=42)
    dt.fit(X, y)
    proba = dt.predict_proba(X)
    assert proba.shape == (len(X), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)

def test_predict_proba_matches_predict(simple_data):
    X, y = simple_data
    dt = DecisionTree(random_state=42)
    dt.fit(X, y)
    pred = dt.predict(X)
    proba = dt.predict_proba(X)
    np.testing.assert_array_equal(pred, np.argmax(proba, axis=1))

def test_depth_property(simple_data):
    X, y = simple_data
    dt = DecisionTree(max_depth=3, random_state=42)
    dt.fit(X, y)
    assert dt.depth <= 3

def test_n_leaves(simple_data):
    X, y = simple_data
    dt = DecisionTree(max_depth=3, random_state=42)
    dt.fit(X, y)
    assert dt.n_leaves >= 1

def test_feature_importances(simple_data):
    X, y = simple_data
    dt = DecisionTree(max_depth=3, random_state=42)
    dt.fit(X, y)
    fi = dt.feature_importances()
    assert fi.shape == (X.shape[1],)
    np.testing.assert_allclose(fi.sum(), 1.0)

def test_feature_importances_zero():
    X = np.ones((20, 2))
    y = np.zeros(20, dtype=int)
    dt = DecisionTree(random_state=42)
    dt.fit(X, y)
    np.testing.assert_array_equal(
        dt.feature_importances(),
        np.zeros(2),
    )

def test_max_depth_zero():
    X = np.random.RandomState(42).randn(50, 3)
    y = (X[:, 0] > 0).astype(int)
    dt = DecisionTree(max_depth=0, random_state=42)
    dt.fit(X, y)
    assert dt.depth == 0

def test_all_same_label():
    X = np.random.RandomState(42).randn(50, 3)
    y = np.zeros(50, dtype=int)
    dt = DecisionTree(random_state=42)
    dt.fit(X, y)
    pred = dt.predict(X)
    assert np.all(pred == 0)
    assert dt.n_leaves == 1

def test_constant_features():
    X = np.ones((50, 3))
    y = np.random.RandomState(42).randint(0, 2, 50)
    dt = DecisionTree(random_state=42)
    dt.fit(X, y)
    pred = dt.predict(X)
    assert pred.shape == (50,)
    assert dt.n_leaves == 1

def test_single_feature():
    X = np.random.RandomState(42).randn(50, 1)
    y = (X[:, 0] > 0).astype(int)
    dt = DecisionTree(max_depth=3, random_state=42)
    dt.fit(X, y)
    pred = dt.predict(X)
    assert pred.shape == (50,)

def test_min_samples_split():
    X = np.random.RandomState(42).randn(50, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    dt = DecisionTree(
        min_samples_split=100,
        random_state=42,
    )
    dt.fit(X, y)
    assert dt.depth == 0
    assert dt.n_leaves == 1

def test_criterion_entropy():
    X = np.random.RandomState(42).randn(50, 2)
    y = (X[:, 0] > 0).astype(int)
    dt = DecisionTree(
        criterion="entropy",
        random_state=42,
    )
    dt.fit(X, y)
    pred = dt.predict(X)
    assert pred.shape == (50,)

def test_max_features_sqrt():
    X = np.random.RandomState(42).randn(100, 6)
    y = (X[:, 0] > 0).astype(int)
    dt = DecisionTree(
        max_features="sqrt",
        random_state=42,
    )
    dt.fit(X, y)
    assert dt.predict(X).shape == (100,)

def test_max_features_log2():
    X = np.random.RandomState(42).randn(100, 8)
    y = (X[:, 0] > 0).astype(int)
    dt = DecisionTree(
        max_features="log2",
        random_state=42,
    )
    dt.fit(X, y)
    assert dt.predict(X).shape == (100,)

def test_max_features_integer():
    X = np.random.RandomState(42).randn(100, 5)
    y = (X[:, 0] > 0).astype(int)
    dt = DecisionTree(
        max_features=2,
        random_state=42,
    )
    dt.fit(X, y)
    assert dt.predict(X).shape == (100,)

def test_against_sklearn(simple_data):
    X, y = simple_data
    ours = DecisionTree(max_depth=3, random_state=42)
    ours.fit(X, y)
    sk = DecisionTreeClassifier(max_depth=3, random_state=42)
    sk.fit(X, y)
    our_acc = (ours.predict(X) == y).mean()
    sk_acc = (sk.predict(X) == y).mean()
    assert abs(our_acc - sk_acc) <= 0.02

def test_depth_against_sklearn():
    rng = np.random.RandomState(42)
    X = rng.randn(200, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    ours = DecisionTree(max_depth=4, random_state=42)
    ours.fit(X, y)
    sk = DecisionTreeClassifier(max_depth=4, random_state=42)
    sk.fit(X, y)
    assert abs(ours.depth - sk.get_depth()) <= 1

def test_feature_importances_against_sklearn():
    rng = np.random.RandomState(42)
    X = rng.randn(300, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    ours = DecisionTree(random_state=42)
    ours.fit(X, y)
    sk = DecisionTreeClassifier(random_state=42)
    sk.fit(X, y)
    np.testing.assert_allclose(
        np.round(ours.feature_importances(), 5),
        np.round(sk.feature_importances_, 5),
        atol=0.02,
    )

def test_xor(xor_data):
    X, y = xor_data
    dt = DecisionTree(max_depth=5, random_state=42)
    dt.fit(X, y)
    acc = (dt.predict(X) == y).mean()
    assert acc > 0.8

def test_sample_weight():
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0, 0, 0, 1, 1])
    w = np.array([1.0, 1.0, 1.0, 2.0, 2.0])
    dt = DecisionTree(max_depth=3, random_state=42)
    dt.fit(X, y, sample_weight=w)
    assert dt.predict(X).shape == (5,)

def test_random_state_reproducibility():
    X = np.random.RandomState(42).randn(100, 3)
    y = (X[:, 0] > 0).astype(int)
    dt1 = DecisionTree(random_state=123)
    dt2 = DecisionTree(random_state=123)
    dt1.fit(X, y)
    dt2.fit(X, y)
    np.testing.assert_array_equal(
        dt1.predict(X),
        dt2.predict(X),
    )

def test_repr_not_fitted():
    dt = DecisionTree()
    assert "not fitted" in repr(dt)

def test_repr_fitted(simple_data):
    X, y = simple_data
    dt = DecisionTree(max_depth=3, random_state=42)
    dt.fit(X, y)
    rep = repr(dt)
    assert "DecisionTree" in rep
    assert "depth" in rep
