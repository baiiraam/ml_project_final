import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix as sk_confusion_matrix,
    f1_score as sk_f1_score,
    roc_auc_score,
)

from src.metrics.evaluation import (
    accuracy_calculation,
    f1_score,
    confusion_matrix,
    auc_roc,
)


def test_accuracy():
    np.random.seed(42)
    n = 300
    y_true = np.random.randint(0, 2, size=n)
    y_pred = y_true.copy()
    noise = np.random.rand(n) < 0.2
    y_pred[noise] = 1 - y_pred[noise]
    scratch = accuracy_calculation(y_true, y_pred)
    sklearn = accuracy_score(y_true, y_pred)
    assert np.isclose(scratch, sklearn)


def test_f1_macro_binary_dataset():
    np.random.seed(42)
    n = 300
    y_true = np.random.randint(0, 2, size=n)
    y_pred = y_true.copy()
    noise = np.random.rand(n) < 0.2
    y_pred[noise] = 1 - y_pred[noise]
    scratch = f1_score(y_true, y_pred, mean="macro")
    sklearn = sk_f1_score(y_true, y_pred, average="macro")
    assert np.isclose(scratch, sklearn)


def test_f1_binary():
    np.random.seed(42)
    n = 300
    y_true = np.random.randint(0, 2, size=n)
    y_pred = y_true.copy()
    noise = np.random.rand(n) < 0.2
    y_pred[noise] = 1 - y_pred[noise]
    scratch = f1_score(y_true, y_pred, mean="binary")
    sklearn = sk_f1_score(y_true, y_pred, average="binary")
    assert np.isclose(scratch, sklearn)


def test_f1_multiclass():
    np.random.seed(42)
    n = 500
    y_true = np.random.randint(0, 4, size=n)
    y_pred = y_true.copy()
    idx = np.random.choice(n, size=int(0.25 * n), replace=False)
    y_pred[idx] = np.random.randint(0, 4, size=len(idx))
    scratch = f1_score(y_true, y_pred, mean="macro")
    sklearn = sk_f1_score(y_true, y_pred, average="macro")
    assert np.isclose(scratch, sklearn)


def test_confusion_matrix():
    np.random.seed(42)
    n = 300
    y_true = np.random.randint(0, 3, size=n)
    y_pred = np.random.randint(0, 3, size=n)
    scratch = confusion_matrix(y_true, y_pred)
    sklearn = sk_confusion_matrix(y_true, y_pred)
    assert np.array_equal(scratch, sklearn)


def test_auc_roc():
    np.random.seed(42)
    n = 300
    y_true = np.random.randint(0, 2, size=n)
    y_score = np.random.rand(n)
    scratch = auc_roc(y_true, y_score)
    sklearn = roc_auc_score(y_true, y_score)
    assert np.isclose(scratch, sklearn, atol=1e-10)
