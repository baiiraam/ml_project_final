import pytest
import numpy as np
from src.trees.decision_tree import DecisionTree
from src.trees.bagging.random_forest import RandomForestClassifier
from src.trees.boosting.adaboost import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
from src.metrics.evaluation import accuracy_calculation, f1_score, auc_roc


@pytest.fixture
def binary_dataset():
    X = np.array(
        [
            [0, 1],
            [1, 1],
            [1, 0],
            [0, 0],
            [2, 1],
            [2, 2],
        ],
        dtype=np.float32,
    )

    y = np.array([0, 1, 1, 0, 1, 0])

    return X, y


@pytest.fixture
def multiclass_dataset():
    X = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 0],
            [3, 1],
            [4, 2],
            [5, 0],
        ],
        dtype=np.float32,
    )

    y = np.array([0, 1, 2, 0, 1, 2])

    return X, y


def test_accuracy():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    result = accuracy_calculation(y_true, y_pred)

    assert result == 0.75


def test_macro_f1():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    result = f1_score(y_true, y_pred, mean="macro")

    assert 0 <= result <= 1


def test_auc_range():

    y_true = np.array([0, 1, 1, 0])

    probabilities = np.array([[0.9, 0.1], [0.2, 0.8], [0.1, 0.9], [0.7, 0.3]])

    result = auc_roc(y_true, probabilities)

    assert 0 <= result <= 1


def test_decision_tree_fit_predict(binary_dataset):

    X, y = binary_dataset

    model = DecisionTree(random_state=42)

    model.fit(X, y)

    pred = model.predict(X)

    assert len(pred) == len(y)
    assert set(pred).issubset(set(y))


def test_random_forest(binary_dataset):

    X, y = binary_dataset

    model = RandomForestClassifier(n_estimators=5, random_state=42)

    model.fit(X, y)

    pred = model.predict(X)

    assert pred.shape == y.shape


def test_adaboost(binary_dataset):

    X, y = binary_dataset

    model = AdaBoostClassifier(n_estimators=5, random_state=42)

    model.fit(X, y)

    pred = model.predict(X)

    assert len(pred) == len(y)


def test_predict_probability_sum(binary_dataset):

    from src.trees.bagging.random_forest import RandomForestClassifier

    X, y = binary_dataset

    model = RandomForestClassifier(n_estimators=10, random_state=42)

    model.fit(X, y)

    probs = model.predict_proba(X)

    assert probs.shape[0] == len(y)

    sums = probs.sum(axis=1)

    np.testing.assert_allclose(sums, np.ones(len(y)), atol=1e-5)


def test_stratified_split(binary_dataset):

    X, y = binary_dataset

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    folds = list(skf.split(X, y))

    assert len(folds) == 3

    for train_idx, test_idx in folds:
        assert len(set(train_idx) & set(test_idx)) == 0
