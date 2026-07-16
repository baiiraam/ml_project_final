import os
import sys

# noqa: E402
sys.path.insert(0, os.path.join(os.getcwd(), ".."))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from sklearn.tree import DecisionTreeClassifier as SkDT  # noqa: E402

from src.metrics.evaluation import accuracy_calculation, auc_roc, f1_score  # noqa: E402
from src.trees.boosting.adaboost import DecisionStump  # noqa: E402
from src.trees.decision_tree import DecisionTree  # noqa: E402
from src.utils.preprocessing import standardize, train_test_split  # noqa: E402


@pytest.fixture
def synthetic_data():
    """Generate a small synthetic dataset for rapid unit testing.

    Returns:
        tuple: A tuple containing:
            - X (numpy.ndarray): Feature matrix of shape (100, 5).
            - y (numpy.ndarray): Binary target labels of shape (100,).
    """
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, size=100)
    return X, y


def test_models_pipeline(synthetic_data):
    """Verify training, prediction, and metric computation pipelines.

    Ensures that custom DecisionTree and DecisionStump predictions remain
    consistent and checks if the custom DecisionTree accuracy strictly
    aligns within a 10% margin compared to the scikit-learn baseline.

    Args:
        synthetic_data (tuple): Tuple of features and target array.
    """
    X_data, y_data = synthetic_data

    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42
    )

    X_train_s, X_test_s = standardize(X_train, X_test)

    dt = DecisionTree(random_state=42)
    dt.fit(X_train_s, y_train)
    pred = dt.predict(X_test_s)
    proba = dt.predict_proba(X_test_s)

    acc_dt = accuracy_calculation(y_test, pred)
    f1_dt = f1_score(y_test, pred, mean="macro")
    auc_dt = auc_roc(y_test, proba)

    assert 0.0 <= acc_dt <= 1.0
    assert 0.0 <= f1_dt <= 1.0
    assert 0.0 <= auc_dt <= 1.0

    stump = DecisionStump(random_state=42)
    stump.fit(X_train_s, y_train)
    pred_s = stump.predict(X_test_s)
    proba_s = stump.predict_proba(X_test_s)

    assert len(pred_s) == len(y_test)
    assert len(proba_s) == len(y_test)

    sk_dt = SkDT(random_state=42)
    sk_dt.fit(X_train_s, y_train)
    sk_pred = sk_dt.predict(X_test_s)
    acc_sk = accuracy_calculation(y_test, sk_pred)

    diff_pct = abs(acc_dt - acc_sk) / max(acc_sk, 1e-10) * 100

    assert diff_pct <= 10.0, (
        f"Accuracy discrepancy vs sklearn exceeds threshold: {diff_pct:.2f}%"
    )
