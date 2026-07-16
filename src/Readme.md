# Source Code (`src/`)

This directory contains the core implementation of the **ML Project Final** developed by **Madagascar Penguins** for the Machine Learning course at **AI Academy, National AI Center (Spring 2026)**.

The `src/` package contains all machine learning algorithm implementations, experiment pipelines, evaluation utilities, preprocessing components, and supporting modules required for the project.

The main goal of this implementation is to build machine learning algorithms from scratch and perform an empirical comparison between two ensemble learning philosophies:

- **Boosting** → AdaBoost with Decision Stumps
- **Bagging** → Random Forest with bootstrap aggregation

The project also includes an unsupervised learning pipeline to analyze dataset structure before and after classification.

---

## Folder Structure

```text
src/
│
├── trees/                     # Decision tree based algorithms
│   ├── __init__.py
│   ├── decision_tree.py
│   ├── bagging/               # Bagging implementations
│   │   ├── __init__.py
|   │   └── random_forest.py
│   └── boosting/              # Boosting implementations
│   │   ├── __init__.py
|   │   └── adaboost.py
│
├── unsupervised/              # Unsupervised learning algorithms
│   ├── pca.py                 # Principal Component Analysis
│   ├── kmeans.py              # K-Means clustering
│   └── dbscan.py              # DBSCAN clustering
│
├── experiments/               # Experiment scripts and comparisons
│   ├── experiment_baseline.py
│   ├── experiment_head_to_head.py
│   └── run_all.py
│
├── metrics/                   # Evaluation metrics
│   └── evaluation.py
│
└── utils/                     # Utility functions
    ├── preprocessing.py
    └── unsupervised_helper.py
```
---


# Modules Overview

## `trees/`

Contains supervised learning algorithms based on decision trees and ensemble methods.

Implemented algorithms:

- Decision Tree Classifier
- Decision Stump
- AdaBoost
- Random Forest

The implementations follow classical machine learning algorithms without using:

- `sklearn.tree.DecisionTreeClassifier`
- `sklearn.ensemble.AdaBoostClassifier`
- `sklearn.ensemble.RandomForestClassifier`

as primary implementations.

Scikit-learn models may only be used as comparison baselines and tests.

---

## `unsupervised/`

Contains implementations of unsupervised learning algorithms developed from scratch.

Included algorithms:

- Principal Component Analysis (PCA)
- K-Means Clustering
- DBSCAN

The module is used for:

- dimensionality reduction,
- cluster discovery,
- visualization,
- comparison between natural data structure and classification performance.

Additional analysis includes:

- PCA explained variance analysis
- K-Means elbow method
- DBSCAN k-distance analysis
- ARI-based cluster evaluation
- t-SNE visualization bonus analysis

---

## `experiments/`

Contains experiment definitions and execution pipelines.

Current experiment modules include:

### Baseline Experiments

Comparison between:

- Custom Decision Tree
- Decision Stump
- Scikit-learn Decision Tree baseline

Metrics:

- Accuracy
- Macro F1-score
- AUC-ROC

---

### Ensemble Comparison

The project investigates:

**AdaBoost**

- effect of increasing number of estimators,
- training/test error behavior,
- sensitivity to noise.

**Random Forest**

- effect of tree count,
- effect of maximum depth,
- variance reduction through bagging.

---

### Head-to-Head Evaluation

The experimental design includes:

- 5-fold cross-validation,
- fixed-resource comparison,
- mean and standard deviation reporting.

Models compared:

- Single Decision Tree
- AdaBoost
- Random Forest
- Scikit-learn Random Forest baseline

---

### Reproducible Pipeline

The project is designed around a unified execution pipeline:

```bash
python src/experiments/run_all.py

This pipeline will generate all required experiment results and visualizations.

(Current development status: experiment orchestration is being finalized.)

metrics/

Contains evaluation functions used throughout experiments.

Supported evaluation includes:

Accuracy
Macro F1-score
ROC-AUC
Classification reports
Cross-validation result summaries

The module provides a consistent evaluation interface across different models.

utils/

Contains shared helper functions.

preprocessing.py

Responsible for:

data preprocessing,
feature preparation,
scaling,
train/test preparation.
unsupervised_helper.py

Contains helper functions for:

clustering analysis,
visualization preparation,
dimensionality reduction workflows.
Development Principles

The source code follows these principles:

From-Scratch Implementation

Core machine learning algorithms are implemented manually to understand:

mathematical foundations,
optimization procedures,
algorithmic trade-offs.
Reproducibility

All randomized algorithms support:

fixed random seeds,
deterministic execution,
repeatable experiments.
Object-Oriented Design

Implementations follow:

reusable classes,
clear interfaces,
modular architecture,
separation between algorithms and experiments.
Testing and Quality Assurance

The project includes automated testing using:

pytest --cov

Testing covers:

algorithm correctness,
edge cases,
reproducibility,
evaluation utilities.
Additional Features

The project includes optional advanced components:

SAMME.R AdaBoost Extension

An extended AdaBoost variant supporting real-valued boosting predictions.

t-SNE Visualization

Additional dimensionality reduction experiments comparing:

PCA projections,
t-SNE embeddings,
clustering structures.
GitHub Actions CI

Automated workflow support for:

testing,
validation,
continuous integration.
Interactive Analysis Notebook

Interactive notebook experiments are included for exploring:

model behavior,
parameter effects,
visualization results.
Team

Team Name: Madagascar Penguins

Project: ML Project Final

Course: Machine Learning – Spring 2026

Institution: AI Academy, National AI Center