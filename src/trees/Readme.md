# Tree-Based Models (`trees/`)



The purpose of this module is to implement decision tree based algorithms from scratch and study the difference between two ensemble learning strategies:

- **Decision Tree** → A hierarchical model that makes predictions by recursively splitting data based on feature values to create simple decision rules on datasets.
- **Boosting** → Sequential improvement of weak learners by focusing on previously misclassified samples.
- **Bagging** → Variance reduction by combining multiple independently trained models using bootstrap sampling.

The implementations follow the principles of CART-based decision trees and ensemble learning methods described in the course material.

---

# Directory Structure

```text
trees/
│
├── __init__.py
├── decision_tree.py
│
├── boosting/
|   ├── __init__.py
│   └── adaboost.py
│
└── bagging/
    ├── __init.py
    └── random_forest.py
```

---

# Implemented Algorithms

## Decision Tree Classifier

File:


decision_tree.py


The Decision Tree implementation follows the CART (Classification and Regression Trees) methodology.

Supported functionality:

- Binary classification
- Continuous feature splitting
- Gini impurity criterion
- Entropy criterion
- Recursive tree construction
- Probability prediction
- Feature importance calculation
- Tree structure inspection

---

## Splitting Strategy

The tree searches for optimal splits by maximizing impurity reduction.

Supported criteria:

### Gini Impurity

Measures node impurity based on class distribution.

### Gini Impurity

Measures node impurity based on the distribution of classes.

$$
Gini = 1 - \sum_{c=1}^{C} p_c^2
$$

where:
- $p_{c}$ — Probability of class \(c\)
- $C$ — number of classes


---

### Entropy

Measures the uncertainty or randomness in the class distribution.

$$
Entropy = -\sum_{c=1}^{C} p_c \log_2(p_c+\epsilon)
$$

where:
- $p_{c}$ — Probability of class \(c\)
- $C$ — number of classes
- $\epsilon$ — small constant added to avoid \(\log(0)\)

---

# Tree Stopping Conditions

Tree growth stops when:

- Maximum depth is reached
- Minimum number of samples for splitting is not satisfied
- Node contains only one class
- No valid split improves impurity

---

# Decision Stump

Decision Stumps are depth-1 decision trees used as weak learners.

They are primarily used in:


boosting/


for AdaBoost training.

A stump performs a single feature-based split and provides a simple classifier that can be improved through boosting.

---

# Boosting Module

Location:


trees/boosting/


This module contains AdaBoost implementations.

## AdaBoost Classifier

AdaBoost combines multiple weak learners sequentially.

The implementation includes:

- weighted sample training,
- estimator error calculation,
- estimator weight calculation,
- sequential learner updates,
- weighted voting prediction.

The algorithm follows the SAMME boosting framework.

Additional support:

- SAMMER extension
- estimator weight tracking
- staged predictions

---

# Bagging Module

Location:


trees/bagging/


This module contains Random Forest implementation.

## Random Forest Classifier

Random Forest combines multiple decision trees using:

- bootstrap sampling,
- feature randomization,
- majority voting.

Implemented components:

- bootstrap dataset generation,
- multiple tree training,
- feature sub-sampling,
- probability averaging,
- feature importance aggregation,
- out-of-bag evaluation.

---

# Random Forest Workflow

For each tree:

1. Create a bootstrap sample from the training data.
2. Train a Decision Tree on the sampled dataset.
3. Randomly select features during splitting.
4. Store the trained tree.

During prediction:

- Each tree produces a prediction.
- Final output is obtained through majority voting.

---

# Model Comparison Purpose

The tree module is designed to answer the main research question of the project:

> Under what conditions does boosting outperform bagging, and vice versa?

The implemented models allow comparison of:

| Model | Learning Strategy | Main Effect |
|---|---|---|
| Decision Tree | Single learner | Baseline model |
| AdaBoost | Sequential ensemble | Bias reduction |
| Random Forest | Parallel ensemble | Variance reduction |

---

# Testing

Tree-based models are validated using automated tests:

```bash
pytest --cov

Testing includes:

correctness checks,
comparison with sklearn baselines,
edge case handling,
deterministic behavior with random seeds.

The implementations are compared against sklearn models only for verification purposes.

The following sklearn models are not used as implementations:

DecisionTreeClassifier
AdaBoostClassifier
RandomForestClassifier
Design Goals

The module follows these principles:

From Scratch Implementation

All main algorithmic components are manually implemented to demonstrate understanding of:

tree construction,
impurity optimization,
ensemble learning,
bias-variance trade-offs.
Reproducibility

All stochastic components support:

random_state
deterministic experiments
repeatable evaluation
Modular Architecture

The separation between:

base learner,
boosting,
bagging,

allows independent testing and future extensions.

Related Experiments

Tree-based models are evaluated through:

baseline comparison,
ensemble scaling experiments,
5-fold cross-validation,
noise robustness analysis,
bias-variance decomposition.

All experiment results contribute to the final comparison between boosting and bagging methods.