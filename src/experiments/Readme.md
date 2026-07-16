# Experiments

This directory contains three experiment scripts used to evaluate the custom machine learning implementations in the project.Also,when we run run_all.py file it will save all outputs of experiments into the figures folder.

The experiments cover:

* Baseline validation of the custom Decision Tree
* Head-to-head supervised model comparison using 5-fold cross-validation
* Automated unsupervised learning analysis using PCA, K-Means, and DBSCAN

## Directory Contents

| File                         | Purpose                                                                                                                   |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `experiment_baseline.py`     | Compares the custom Decision Tree and Decision Stump with scikit-learn's Decision Tree.                                   |
| `experiment_head_to_head.py` | Compares Decision Tree, AdaBoost, Random Forest, and scikit-learn Random Forest using stratified 5-fold cross-validation. |
| `run_all.py`                 | Runs the complete unsupervised analysis pipeline for PCA, K-Means, and DBSCAN.                                            |

---

## Datasets

The experiment scripts use four datasets:

| Dataset       | Task                      | Number of Classes |
| ------------- | ------------------------- | ----------------: |
| Breast Cancer | Binary classification     |                 2 |
| Adult Income  | Binary classification     |                 2 |
| Covertype     | Multiclass classification |                 7 |
| MNIST         | Multiclass classification |                10 |

The datasets are loaded through the project's preprocessing utilities or directly from scikit-learn, OpenML, and the UCI Machine Learning Repository.

---

# 1. Baseline Experiment

File:

```text
experiment_baseline.py
```

The baseline experiment validates the custom tree implementation by comparing three models:

* Custom `DecisionTree`
* Custom `DecisionStump`
* scikit-learn `DecisionTreeClassifier`

The experiment runs on all four datasets.

## Workflow

For each dataset, the script:

1. Loads the dataset.
2. Prints its shape, classes, and class distribution.
3. Splits the data into training and testing sets.
4. Standardizes the features.
5. Trains the three models.
6. Calculates evaluation metrics.
7. Compares the custom Decision Tree accuracy with scikit-learn.
8. Generates a comparison figure.

The train-test split uses:

```python
test_size=0.2
random_state=42
```

## Evaluation Metrics

The following custom metrics are calculated:

* Accuracy
* Macro F1-score
* AUC-ROC

The metrics are imported from:

```text
src/metrics/evaluation.py
```

## Correctness Check

The custom Decision Tree accuracy is compared with the scikit-learn Decision Tree accuracy.

The relative accuracy difference is calculated as:

```text
|custom accuracy - sklearn accuracy|
------------------------------------ × 100
        sklearn accuracy
```

The comparison result is marked as:

* `PASS` when the difference is at most 2%
* `FAIL` when the difference is greater than 2%

## Generated Figure

The script creates:

```text
figures/baseline_comparison.png
```

The figure contains three bar charts comparing:

* Accuracy
* Macro F1-score
* AUC-ROC

for all three models across the four datasets.

## Run

From the project root:

```bash
python experiments/experiment_baseline.py
```

---

# 2. Head-to-Head Supervised Comparison

File:

```text
experiment_head_to_head.py
```

This experiment performs a more comprehensive comparison of the supervised learning models.

The following models are evaluated:

* Custom single `DecisionTree`
* Custom `AdaBoostClassifier`
* Custom `RandomForestClassifier`
* scikit-learn `RandomForestClassifier`

AdaBoost and both Random Forest implementations use:

```python
n_estimators=100
```

## Dataset Preparation

The experiment uses:

* The complete Breast Cancer dataset
* The cleaned Adult Income dataset
* The first 1,000 Covertype samples
* The first 1,000 MNIST samples

The reduced Covertype and MNIST subsets help keep the execution time manageable for custom implementations.

The Covertype labels are converted to zero-based class indices.

## Cross-Validation

The experiment uses stratified 5-fold cross-validation:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)
```

Stratification preserves the class distribution across folds.

For every dataset, each model is trained and evaluated once per fold.

This produces:

```text
4 datasets × 4 models × 5 folds = 80 experiment results
```

## Parallel Execution

The experiment uses `joblib` to run model-fold combinations in parallel.

The worker count is selected automatically:

```python
N_JOBS = max(1, min(cpu_count - 1, 6))
```

This configuration:

* Keeps at least one CPU available
* Uses no more than six parallel workers
* Prevents excessive resource usage

The custom and scikit-learn Random Forest models use `n_jobs=1` internally because parallelism is already applied at the experiment level.

This avoids nested parallel execution.

## Evaluation Metrics

For each fold, the script records:

* Dataset
* Fold number
* Model
* Accuracy
* Macro F1-score
* AUC-ROC

The results are stored in a pandas DataFrame named:

```python
fold_results_df
```

## Summary Table

Results are grouped by dataset and model.

Each metric is reported in the format:

```text
mean ± sample standard deviation
```

For example:

```text
0.9534 ± 0.0128
```

The summary table contains:

* Dataset
* Model
* Accuracy
* Macro F1
* AUC-ROC

## Generated Figures

The experiment creates both box plots and violin plots.

For every dataset, it generates visual comparisons of:

* Accuracy
* Macro F1-score
* AUC-ROC

Example output files:

```text
figures/head_to_head_box_breast_cancer.png
figures/head_to_head_violin_breast_cancer.png
figures/head_to_head_box_adult_income.png
figures/head_to_head_violin_adult_income.png
figures/head_to_head_box_covertype.png
figures/head_to_head_violin_covertype.png
figures/head_to_head_box_mnist.png
figures/head_to_head_violin_mnist.png
```

## Run

From the project root:

```bash
python experiments/experiment_head_to_head.py
```

---

# 3. Unsupervised Experiment Runner

File:

```text
run_all.py
```

This script runs the complete unsupervised learning analysis.

It evaluates the project's custom implementations and helper functions for:

* PCA
* K-Means
* DBSCAN

The main experiment functions are imported from:

```text
src/utils/unsupervised_helper.py
```

## Dataset Loading

The script loads:

### Breast Cancer

Loaded from scikit-learn.

```python
load_breast_cancer()
```

Maximum number of analyzed samples:

```text
3,000
```

### Adult Income

Loaded from the UCI Machine Learning Repository.

Several categorical columns are removed before analysis:

* `workclass`
* `education`
* `marital-status`
* `occupation`
* `relationship`
* `race`
* `sex`
* `native-country`

Maximum number of analyzed samples:

```text
3,000
```

### Covertype

Loaded from the UCI Machine Learning Repository.

Binary indicator columns related to wilderness areas and soil types are removed.

Maximum number of analyzed samples:

```text
3,000
```

### MNIST

Loaded from OpenML.

Maximum number of analyzed samples:

```text
1,000
```

## Analysis Pipeline

For each dataset, the script performs the following steps.

### Step 1: K-Means Analysis

The script runs:

```python
kmeans_task(...)
```

The K-Means results contain the Adjusted Rand Index for different values of `k`.

The best number of clusters is selected using:

```python
best_k = int(
    kmeans_results.loc[kmeans_results["ARI"].idxmax(), "k"]
)
```

Therefore, the selected value of `k` is the one with the highest ARI score.

### Step 2: DBSCAN Epsilon Search

The script runs:

```python
find_best_dbscan_eps(...)
```

This function evaluates multiple `eps` values and returns the best DBSCAN configuration.

The selected epsilon value is extracted using:

```python
best_eps = float(best_dbscan["eps"])
```

### Step 3: PCA Analysis

The script runs:

```python
pca_task(...)
```

The selected K-Means cluster count and DBSCAN epsilon are passed into the PCA analysis:

```python
n_clusters=best_k
dbscan_eps=best_eps
```

This allows the reduced PCA representation to be analyzed using both clustering methods.

### Step 4: DBSCAN Analysis

The script runs:

```python
dbscan_task(...)
```

using the best epsilon value found during the parameter search.

### Step 5: Best-Epsilon Visualization

The script runs:

```python
plot_dbscan_best_eps(...)
```

to visualize the final DBSCAN result using the selected epsilon value.

## Automatic Figure Saving

The script uses the non-interactive Matplotlib backend:

```python
matplotlib.use("Agg")
```

This allows it to run without opening graphical windows.

The normal `plt.show()` function is replaced with a custom function that automatically saves all active figures.

Figures are stored in:

```text
figures/
```

Generated names include the dataset, experiment type, and figure number.

Examples:

```text
figures/Breast_Cancer_kmeans_1.png
figures/Breast_Cancer_dbscan_eps_search_1.png
figures/Breast_Cancer_pca_1.png
figures/Breast_Cancer_dbscan_1.png
figures/Breast_Cancer_dbscan_best_eps_1.png
```

The same naming format is used for Adult Income, Covertype, and MNIST.

## Run

From the project root:

```bash
python experiments/run_all.py
```

---

# Requirements

The three scripts depend on the following main packages:

```text
numpy
pandas
matplotlib
scikit-learn
joblib
ucimlrepo
```

Install the project dependencies using:

```bash
pip install -r requirements.txt
```

---

# Expected Project Structure

The scripts assume a structure similar to:

```text
project-root/
├── experiments/
│   ├── experiment_baseline.py
│   ├── experiment_head_to_head.py
│   └── run_all.py
├── figures/
├── src/
│   ├── metrics/
│   │   └── evaluation.py
│   ├── trees/
│   │   ├── decision_tree.py
│   │   ├── bagging/
│   │   │   └── random_forest.py
│   │   └── boosting/
│   │       └── adaboost.py
│   └── utils/
│       ├── preprocessing.py
│       └── unsupervised_helper.py
├── requirements.txt
└── README.md
```

---

# Reproducibility

The supervised experiments use:

```python
RANDOM_STATE = 42
```

NumPy is also initialized with the same seed:

```python
np.random.seed(42)
```

This improves reproducibility for:

* Data splitting
* Cross-validation
* Decision Tree training
* AdaBoost training
* Random Forest training

Some small variation may still occur depending on:

* Operating system
* Python version
* Package versions
* Parallel execution behavior

---

# Output Summary

| Script                       | Main Output                                                     |
| ---------------------------- | --------------------------------------------------------------- |
| `experiment_baseline.py`     | Printed baseline metrics and `baseline_comparison.png`          |
| `experiment_head_to_head.py` | Fold-level results, summary tables, box plots, and violin plots |
| `run_all.py`                 | PCA, K-Means, DBSCAN, epsilon-search, and clustering figures    |
