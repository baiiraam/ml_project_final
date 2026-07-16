# Notebooks Directory

This directory contains Jupyter notebooks developed for experiments, analysis, and evaluation in the ML final project.

## Overview

The notebooks cover different stages of the machine learning workflow, including:

* Baseline model experiments
* Cross-validation comparison
* Bias-variance analysis
* Noise robustness evaluation
* Unsupervised learning analysis

---

## Notebook Descriptions

### 1. `experiments_baseline.ipynb`

**Purpose:**
Establishes baseline machine learning models and evaluates their initial performance.

**Includes:**

* Dataset loading
* Data preprocessing
* Baseline model training
* Evaluation metrics
* Initial performance comparison

---

### 2. `Head-to-Head_5-fold_CV.ipynb`

**Purpose:**
Compares models using 5-fold cross-validation.

**Includes:**

* K-fold validation setup
* Model-to-model comparison
* Mean and standard deviation of scores
* Generalization performance analysis

---

### 3. `Bias_Variance_Decompostion.ipynb`

**Purpose:**
Analyzes the bias-variance trade-off of machine learning models.

**Includes:**

* Training and validation error analysis
* Model complexity evaluation
* Underfitting and overfitting investigation
* Bias-variance decomposition experiments

---

### 4. `Noise_Robustness.ipynb`

**Purpose:**
Evaluates how models behave under noisy data conditions.

**Includes:**

* Noise injection experiments
* Performance degradation analysis
* Robustness comparison between models

---

### 5. `Unsupervised_Analysis.ipynb`

**Purpose:**
Explores datasets using unsupervised learning techniques.

**Includes:**

* Dimensionality reduction
* Clustering experiments
* Pattern discovery
* Visualization of feature spaces

---

## Running the Notebooks

From the project root:

```bash
jupyter notebook
```

or:

```bash
jupyter lab
```

Required dependencies should be installed before running experiments.

---

## Output Files

Generated plots and visualizations should be saved in:

```
../figures/
```

Datasets are loaded from:

```
../data/
```

---

