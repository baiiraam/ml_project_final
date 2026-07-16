# Evaluation Metrics

The `metrics` directory contains custom implementations of classification evaluation metrics used throughout the project.

## Directory Contents

| File            | Description                                                                            |
| --------------- | -------------------------------------------------------------------------------------- |
| `evaluation.py` | Implements accuracy, F1-score, confusion matrix, and AUC-ROC calculations using NumPy. |

## Overview

The metrics in `evaluation.py` are implemented from scratch without relying on metric functions from scikit-learn.

The module supports both binary and multiclass classification and is used to evaluate the custom Decision Tree, AdaBoost, and Random Forest implementations.

Available functions:

* `accuracy_calculation`
* `f1_score`
* `confusion_matrix`
* `auc_roc`
* `_binary_auc`

## Import

```python
from src.metrics.evaluation import (
    accuracy_calculation,
    auc_roc,
    confusion_matrix,
    f1_score,
)
```

---

# Accuracy

Function:

```python
accuracy_calculation(y_true, y_pred)
```

Calculates the proportion of correctly predicted samples.

## Formula

```text
           Number of correct predictions
Accuracy = -----------------------------
              Total number of samples
```

## Parameters

| Parameter | Description                                  |
| --------- | -------------------------------------------- |
| `y_true`  | Array containing the true class labels.      |
| `y_pred`  | Array containing the predicted class labels. |

Both inputs are converted to NumPy arrays before the calculation.

## Returns

A floating-point value between `0` and `1`.

* `1.0` means every prediction is correct.
* `0.0` means no predictions are correct.

## Example

```python
import numpy as np

from src.metrics.evaluation import accuracy_calculation

y_true = np.array([0, 1, 1, 0, 1])
y_pred = np.array([0, 1, 0, 0, 1])

accuracy = accuracy_calculation(y_true, y_pred)

print(accuracy)
```

Output:

```text
0.8
```

Four of the five predictions are correct, so the accuracy is `0.8`.

---

# F1-Score

Function:

```python
f1_score(y_true, y_pred, mean="macro")
```

Calculates the F1-score using class-specific precision and recall.

## Precision

Precision measures how many samples predicted as a particular class actually belong to that class.

```text
                  True Positives
Precision = ---------------------------
            True Positives + False Positives
```

## Recall

Recall measures how many samples belonging to a class were identified correctly.

```text
               True Positives
Recall = ---------------------------
         True Positives + False Negatives
```

## F1 Formula

The F1-score is the harmonic mean of precision and recall.

```text
                  Precision × Recall
F1 = 2 × --------------------------------
               Precision + Recall
```

When both precision and recall are zero, the implementation returns an F1-score of `0.0` for that class.

## Parameters

| Parameter | Description                                                                                       |
| --------- | ------------------------------------------------------------------------------------------------- |
| `y_true`  | Array containing the true class labels.                                                           |
| `y_pred`  | Array containing the predicted class labels.                                                      |
| `mean`    | Determines how class-level F1-scores are returned. Supported values are `"macro"` and `"binary"`. |

## Macro F1

With:

```python
mean="macro"
```

the F1-score is calculated separately for every class and then averaged:

```text
           F1₁ + F1₂ + ... + F1ₖ
Macro F1 = ----------------------
                      k
```

Each class receives equal importance regardless of how many samples it contains.

Macro F1 is useful for:

* Multiclass classification
* Imbalanced datasets
* Comparing performance across all classes equally

## Binary F1

With:

```python
mean="binary"
```

the function returns the F1-score of the last class in the sorted unique labels found in `y_true`.

For conventional binary labels:

```text
[0, 1]
```

class `1` is treated as the positive class.

## Returns

A floating-point F1-score between `0` and `1`.

## Example: Macro F1

```python
import numpy as np

from src.metrics.evaluation import f1_score

y_true = np.array([0, 0, 1, 1, 2, 2])
y_pred = np.array([0, 1, 1, 1, 2, 0])

score = f1_score(y_true, y_pred, mean="macro")

print(score)
```

## Example: Binary F1

```python
import numpy as np

from src.metrics.evaluation import f1_score

y_true = np.array([0, 1, 1, 0, 1])
y_pred = np.array([0, 1, 0, 0, 1])

score = f1_score(y_true, y_pred, mean="binary")

print(score)
```

## Invalid Averaging Mode

Only `"macro"` and `"binary"` are accepted.

For example:

```python
f1_score(y_true, y_pred, mean="weighted")
```

raises:

```text
ValueError: mean must be 'macro' or 'binary'
```

---

# Confusion Matrix

Function:

```python
confusion_matrix(y_true, y_pred)
```

Constructs a confusion matrix from the true and predicted labels.

A confusion matrix shows how often every true class is predicted as each available class.

## Matrix Structure

* Rows represent true labels.
* Columns represent predicted labels.
* Diagonal values represent correct predictions.
* Off-diagonal values represent classification errors.

For binary classification, the structure is:

```text
                     Predicted
                  Class 0  Class 1
True Class 0        TN       FP
True Class 1        FN       TP
```

where:

* `TN` is true negatives.
* `FP` is false positives.
* `FN` is false negatives.
* `TP` is true positives.

## Class Ordering

The function determines classes using:

```python
np.unique(np.concatenate((y_true, y_pred)))
```

This ensures that labels appearing only in the predictions are also included.

The resulting rows and columns follow NumPy's sorted unique-label order.

## Parameters

| Parameter | Description                                  |
| --------- | -------------------------------------------- |
| `y_true`  | Array containing the true class labels.      |
| `y_pred`  | Array containing the predicted class labels. |

## Returns

An integer NumPy array with shape:

```text
(number of classes, number of classes)
```

## Example

```python
import numpy as np

from src.metrics.evaluation import confusion_matrix

y_true = np.array([0, 0, 1, 1, 2, 2])
y_pred = np.array([0, 1, 1, 1, 2, 0])

matrix = confusion_matrix(y_true, y_pred)

print(matrix)
```

Output:

```text
[[1 1 0]
 [0 2 0]
 [1 0 1]]
```

Interpretation:

* One class-0 sample was correctly predicted as class 0.
* One class-0 sample was incorrectly predicted as class 1.
* Both class-1 samples were classified correctly.
* One class-2 sample was predicted correctly.
* One class-2 sample was incorrectly predicted as class 0.

---

# AUC-ROC

Public function:

```python
auc_roc(y_true, y_pred_proba)
```

Calculates the Area Under the Receiver Operating Characteristic Curve.

The function supports:

* Binary classification with one-dimensional scores
* Binary classification with two probability columns
* Multiclass classification using one-vs-rest macro averaging

## Interpretation

AUC-ROC measures how effectively a model ranks positive samples above negative samples.

Typical interpretations are:

|       AUC | Interpretation               |
| --------: | ---------------------------- |
|     `1.0` | Perfect ranking              |
| `0.9–1.0` | Excellent discrimination     |
| `0.8–0.9` | Good discrimination          |
| `0.7–0.8` | Fair discrimination          |
|     `0.5` | Equivalent to random ranking |
|   `< 0.5` | Ranking is worse than random |

AUC evaluates ranking quality rather than classification accuracy at a fixed threshold.

---

## Binary AUC with One-Dimensional Scores

When `y_pred_proba` is one-dimensional, it is interpreted as the score or probability of the positive class.

```python
import numpy as np

from src.metrics.evaluation import auc_roc

y_true = np.array([0, 0, 1, 1])
positive_scores = np.array([0.1, 0.4, 0.35, 0.8])

auc = auc_roc(y_true, positive_scores)

print(auc)
```

---

## Binary AUC with Two Probability Columns

When the probability array contains two columns, the second column is used:

```python
y_pred_proba[:, 1]
```

The expected structure is:

```text
Column 0: probability of class 0
Column 1: probability of class 1
```

Example:

```python
import numpy as np

from src.metrics.evaluation import auc_roc

y_true = np.array([0, 0, 1, 1])

probabilities = np.array([
    [0.90, 0.10],
    [0.60, 0.40],
    [0.65, 0.35],
    [0.20, 0.80],
])

auc = auc_roc(y_true, probabilities)

print(auc)
```

---

## Multiclass AUC

For more than two probability columns, the function uses the one-vs-rest strategy.

For each class:

1. The selected class is treated as positive.
2. Every other class is treated as negative.
3. Binary AUC is calculated using that class's probability column.
4. The class-level AUC values are averaged.

```text
                      AUC₁ + AUC₂ + ... + AUCₖ
Multiclass AUC = --------------------------------
                                  k
```

This produces a macro-averaged multiclass AUC.

Example:

```python
import numpy as np

from src.metrics.evaluation import auc_roc

y_true = np.array([0, 1, 2, 0, 1, 2])

probabilities = np.array([
    [0.80, 0.10, 0.10],
    [0.10, 0.75, 0.15],
    [0.05, 0.15, 0.80],
    [0.60, 0.25, 0.15],
    [0.20, 0.65, 0.15],
    [0.10, 0.20, 0.70],
])

auc = auc_roc(y_true, probabilities)

print(auc)
```

The probability columns must correspond to the sorted classes returned by:

```python
np.unique(y_true)
```

For labels `[0, 1, 2]`, the expected column order is:

```text
Column 0 → class 0
Column 1 → class 1
Column 2 → class 2
```

---

# Internal Binary AUC Calculation

Helper function:

```python
_binary_auc(y_true, y_score)
```

This private helper calculates binary AUC using score ranks.

It is used internally by `auc_roc` and is not normally called directly by other project modules.

## Rank-Based Calculation

The implementation:

1. Sorts prediction scores using a stable merge sort.
2. Assigns ranks to sorted scores.
3. Gives tied scores their average rank.
4. Sums the ranks assigned to positive samples.
5. Calculates AUC using the Mann–Whitney rank statistic.

The formula is:

```text
             Sum of positive ranks - n₊(n₊ + 1) / 2
AUC = ----------------------------------------------------
                            n₊ × n₋
```

where:

* `n₊` is the number of positive samples.
* `n₋` is the number of negative samples.

## Stable Sorting

Scores are sorted using:

```python
np.argsort(y_score, kind="mergesort")
```

Merge sort is stable, which provides predictable handling of equal scores.

## Tied Scores

Samples with identical prediction scores receive their average rank.

This avoids incorrectly favouring one sample over another when their model scores are equal.

## Required Classes

Binary AUC requires at least:

* One positive sample
* One negative sample

Otherwise, the function raises:

```text
ValueError: Both positive and negative samples are required.
```

This can occur when a test set or cross-validation fold contains only one class.

---

# Complete Example

```python
import numpy as np

from src.metrics.evaluation import (
    accuracy_calculation,
    auc_roc,
    confusion_matrix,
    f1_score,
)

y_true = np.array([0, 1, 2, 0, 1, 2])
y_pred = np.array([0, 1, 1, 0, 2, 2])

y_pred_proba = np.array([
    [0.80, 0.10, 0.10],
    [0.10, 0.75, 0.15],
    [0.15, 0.55, 0.30],
    [0.70, 0.20, 0.10],
    [0.20, 0.35, 0.45],
    [0.10, 0.15, 0.75],
])

accuracy = accuracy_calculation(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, mean="macro")
matrix = confusion_matrix(y_true, y_pred)
auc = auc_roc(y_true, y_pred_proba)

print("Accuracy:", accuracy)
print("Macro F1:", macro_f1)
print("Confusion matrix:")
print(matrix)
print("AUC-ROC:", auc)
```

---

# Function Summary

| Function               | Purpose                                            | Binary Support | Multiclass Support |
| ---------------------- | -------------------------------------------------- | :------------: | :----------------: |
| `accuracy_calculation` | Calculates the proportion of correct predictions   |       Yes      |         Yes        |
| `f1_score`             | Calculates binary or macro-averaged F1-score       |       Yes      |         Yes        |
| `confusion_matrix`     | Counts true-label and predicted-label combinations |       Yes      |         Yes        |
| `auc_roc`              | Calculates binary or macro one-vs-rest AUC         |       Yes      |         Yes        |
| `_binary_auc`          | Internal rank-based binary AUC calculation         |       Yes      |         No         |

# Expected Directory Structure

```text
src/
└── metrics/
    ├── __init__.py
    ├── evaluation.py
    └── README.md
```

# Dependency

The module depends only on NumPy:

```text
numpy
```

Install it with:

```bash
pip install numpy
```

Alternatively, install all project dependencies:

```bash
pip install -r requirements.txt
```

# Current Behavior and Assumptions

The current implementation makes the following assumptions:

* `y_true` and `y_pred` contain the same number of samples.
* Binary AUC labels are represented by `0` and `1`.
* In two-column binary probabilities, column `1` represents the positive class.
* In multiclass probabilities, columns follow the sorted order of classes in `y_true`.
* Binary F1 uses the last class in the sorted unique true labels as the positive class.
* Macro F1 averages all classes present in `y_true`.
* Inputs are one-dimensional label arrays or can be safely converted into the expected NumPy representation.

These conventions must be followed by model implementations when producing predictions and probability matrices.
