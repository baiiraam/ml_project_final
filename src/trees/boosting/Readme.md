# Boosting Ensemble (`boosting/`)

This directory contains the **Boosting-based ensemble learning implementation** developed for the **ML Project Final** by **Madagascar Penguins**.

The main purpose of this module is to implement **AdaBoost from scratch** using Decision Stumps as weak learners and analyze the boosting approach compared with bagging methods such as Random Forest.

Boosting improves model performance by training weak learners sequentially, where each new learner focuses more on previously misclassified samples.

---

# Overview

## What is Boosting?

Boosting is an ensemble learning strategy that combines multiple weak models into a stronger predictive model.

Unlike bagging, where models are trained independently, boosting trains models sequentially.

The general workflow:

1. Train a weak learner on the training data.
2. Evaluate its mistakes.
3. Increase the importance of incorrectly classified samples.
4. Train the next learner with updated sample weights.
5. Combine all learners using weighted voting.

The main goal of boosting is to reduce model bias and improve predictive performance.

---

# AdaBoost Classifier

The main implementation in this module is:

```
AdaBoostClassifier
```

The implementation follows the AdaBoost SAMME formulation for classification.

The weak learners are:

```
DecisionStump
```

A Decision Stump is a decision tree with maximum depth equal to 1.

---

# Decision Stump

A Decision Stump is a simplified decision tree that performs only one split.

Implementation:

```python
DecisionStump(DecisionTree)
```

Characteristics:

- Maximum depth = 1
- Single feature split
- Simple weak learner
- Used as the base estimator for AdaBoost

Although individual stumps have limited predictive power, combining many stumps creates a strong classifier.

---

# AdaBoost Training Process

The algorithm starts by assigning equal weights to all training samples.

For each boosting iteration:

## 1. Train Weak Learner

A Decision Stump is trained using the current sample weights.

Samples with higher weights have more influence during training.

---

## 2. Calculate Weighted Error

The weighted classification error is calculated:

```
error = sum(sample_weight × incorrect_predictions)
```

A lower error indicates a stronger weak learner.

---

## 3. Calculate Estimator Weight

Each stump receives a weight based on its performance.

Better performing stumps receive higher influence during prediction.

The estimator weight controls how much each weak learner contributes to the final decision.

---

## 4. Update Sample Weights

Incorrectly classified samples receive higher weights.

This forces the next weak learner to focus more on difficult examples.

Correctly classified samples become less influential.

---

# Final Prediction

The final prediction is obtained using weighted voting:

```
Final prediction =
weighted combination of all Decision Stump predictions
```

Each stump contributes according to its learned estimator weight.

---

# Probability Prediction

The implementation supports probability estimation.

The process:

1. Collect weighted predictions from all estimators.
2. Convert weighted votes into normalized probabilities.
3. Return class probability distribution.

Available through:

```python
model.predict_proba(X)
```

---

# Staged Prediction

AdaBoost supports monitoring model performance after each boosting iteration.

Method:

```python
staged_predict(X)
```

Returns predictions after:

- first estimator,
- second estimator,
- third estimator,
- ...
- final estimator.

This functionality is useful for:

- tracking training progress,
- detecting overfitting,
- creating error curves.

---

# Model Parameters

| Parameter | Description |
|---|---|
| `n_estimators` | Number of boosting rounds |
| `learning_rate` | Contribution of each weak learner |
| `criterion` | Decision tree split criterion (`gini` or `entropy`) |
| `random_state` | Controls reproducibility |

Example:

```python
AdaBoostClassifier(
    n_estimators=100,
    learning_rate=1.0,
    criterion="gini",
    random_state=42
)
```

---

# Bias Reduction Perspective

AdaBoost mainly improves performance by reducing model bias.

A single Decision Stump:

- has high bias,
- cannot capture complex decision boundaries.

AdaBoost:

- combines many weak learners,
- gradually corrects previous mistakes,
- creates a stronger classifier.

However, boosting can become sensitive to noisy data because misclassified samples receive increasing weights.

---

# Comparison with Bagging

The project compares two ensemble strategies:

| Method | Strategy | Main Effect |
|---|---|---|
| AdaBoost | Sequential learning | Reduces bias |
| Random Forest | Independent bootstrap models | Reduces variance |

The main research question:

> Under what conditions does boosting outperform bagging?

is investigated using this implementation.

---

# Additional Features

## SAMMER Extension

The project includes support for real-valued boosting concepts as an advanced extension.

SAMME.R improves probability-based multi-class boosting by using class probability estimates instead of only discrete predictions.

---

# Testing

The implementation is tested using:

```bash
pytest --cov
```

Testing includes:

- weak learner training,
- estimator weight calculation,
- error calculation,
- sample weight updates,
- staged predictions,
- reproducibility with fixed random seeds.

---

# File Structure

```
boosting/
│
└── adaboost.py
```

Dependencies:

```
src/trees/decision_tree.py
```

The AdaBoost implementation uses the custom Decision Tree implementation and does not rely on:

```python
sklearn.ensemble.AdaBoostClassifier
```

as the primary algorithm.

---

# Related Experiments

This module is evaluated through:

- AdaBoost estimator scaling experiments.
- Training and testing error analysis.
- Noise robustness experiments.
- Cross-validation comparison.
- Bias-variance analysis.

These experiments are used to understand the strengths and limitations of boosting methods.

---

# Team

**Team:** Madagascar Penguins  
**Project:** ML Project Final  
**Course:** Machine Learning – Spring 2026  
**Institution:** AI Academy, National AI Center