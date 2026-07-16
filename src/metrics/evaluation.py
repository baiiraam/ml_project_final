# Evaluation metrics – placeholder stub
import numpy as np


def accuracy_calculation(y_true, y_pred):
    """
    Calculate classification accuracy.
    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    Returns
    -------
    float
        Accuracy score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(y_true == y_pred)


def f1_score(y_true, y_pred, mean="macro"):
    """
    Calculate F1 score.
    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    mean : str, default="macro"
        "macro" or "binary"
    Returns
    -------
    float
        F1 score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(y_true)
    f1_scores = []
    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        f1_scores.append(f1)
    if mean == "macro":
        return np.mean(f1_scores)
    elif mean == "binary":
        return f1_scores[-1]
    else:
        raise ValueError("mean must be 'macro' or 'binary'")


def confusion_matrix(y_true, y_pred):
    """
    Compute confusion matrix.
    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    Returns
    -------
    np.ndarray
        Confusion matrix.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(np.concatenate((y_true, y_pred)))
    n_classes = len(classes)
    label_to_index = {label: idx for idx, label in enumerate(classes)}
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        i = label_to_index[true]
        j = label_to_index[pred]
        matrix[i, j] += 1
    return matrix


def _binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = np.sum(pos)
    n_neg = np.sum(neg)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Both positive and negative samples are required.")
    # Stable sort
    order = np.argsort(y_score, kind="mergesort")
    scores = y_score[order]
    # Average ranks for tied scores
    ranks = np.empty(len(scores), dtype=float)
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and scores[j] == scores[i]:
            j += 1
        avg_rank = (i + j - 1) / 2 + 1
        ranks[i:j] = avg_rank
        i = j
    final_ranks = np.empty_like(ranks)
    final_ranks[order] = ranks
    sum_pos = np.sum(final_ranks[pos])
    auc = (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def auc_roc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    if y_pred_proba.ndim == 1:
        return _binary_auc(y_true, y_pred_proba)
    if y_pred_proba.shape[1] == 2:
        return _binary_auc(y_true, y_pred_proba[:, 1])
    classes = np.unique(y_true)
    aucs = []
    for i, cls in enumerate(classes):
        binary_true = (y_true == cls).astype(int)
        aucs.append(_binary_auc(binary_true, y_pred_proba[:, i]))
    return float(np.mean(aucs))
