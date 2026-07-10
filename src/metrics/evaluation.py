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
    label_to_index = {
        label: idx
        for idx, label in enumerate(classes)
    }
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        i = label_to_index[true]
        j = label_to_index[pred]
        matrix[i, j] += 1
    return matrix

def auc_roc(y_true, y_score):
    """
    Compute ROC AUC for binary classification.

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels (0/1).
    y_score : np.ndarray
        Predicted probabilities for positive class.

    Returns
    -------
    float
        ROC AUC score.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    order = np.argsort(-y_score)
    y_true = y_true[order]
    positives = np.sum(y_true == 1)
    negatives = np.sum(y_true == 0)
    if positives == 0 or negatives == 0:
        raise ValueError("Both classes must exist.")
    tpr = [0.0]
    fpr = [0.0]
    tp = 0
    fp = 0
    for label in y_true:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / positives)
        fpr.append(fp / negatives)
    tpr.append(1.0)
    fpr.append(1.0)
    auc = np.trapezoid(tpr, fpr)
    return auc