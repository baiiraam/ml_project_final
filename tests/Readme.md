# Test Suite

The `tests` directory contains unit, integration, regression, and workflow tests for the custom machine learning library.

The test suite validates:

* Custom evaluation metrics
* Decision Tree behavior
* AdaBoost and Decision Stump behavior
* Random Forest functionality
* PCA, K-Means, and DBSCAN implementations
* Bias-variance utilities
* Unsupervised experiment helpers
* Baseline and head-to-head experiment pipelines
* AdaBoost scaling utilities
* Automatic experiment execution and figure generation

The tests are primarily written with `pytest`, NumPy assertions, synthetic datasets, and selected scikit-learn reference implementations.

---

## Test Files

| File                                | Tested Component                                          |
| ----------------------------------- | --------------------------------------------------------- |
| `test_evaluation.py`                | Accuracy, F1-score, confusion matrix, and AUC-ROC         |
| `test_decision_tree.py`             | Custom Decision Tree                                      |
| `test_adaboost.py`                  | AdaBoost and Decision Stump                               |
| `test_random_forest.py`             | Custom Random Forest                                      |
| `test_pca.py`                       | Principal Component Analysis                              |
| `test_kmeans.py`                    | K-Means clustering                                        |
| `test_dbscan.py`                    | DBSCAN clustering                                         |
| `test_bias_variance.py`             | Bias-variance decomposition helpers                       |
| `test_unsupervised_helper.py`       | PCA, K-Means, and DBSCAN utility workflows                |
| `test_experiment_baseline.py`       | Baseline supervised experiment pipeline                   |
| `test_experiments_head_to_head.py`  | Head-to-head model comparison components                  |
| `test_adaboost_experiment_utils.py` | AdaBoost scaling experiment utilities                     |
| `test_run_all.py`                   | Full unsupervised experiment runner and figure generation |

---

# Evaluation Metric Tests

File:

```text
test_evaluation.py
```

These tests compare the custom metric implementations against scikit-learn.

The following functions are tested:

* `accuracy_calculation`
* `f1_score`
* `confusion_matrix`
* `auc_roc`

The tests generate reproducible binary and multiclass labels using NumPy and verify that the custom outputs match:

* `sklearn.metrics.accuracy_score`
* `sklearn.metrics.f1_score`
* `sklearn.metrics.confusion_matrix`
* `sklearn.metrics.roc_auc_score`

The test suite covers:

* Binary accuracy
* Binary F1-score
* Macro F1-score
* Multiclass macro F1-score
* Multiclass confusion matrices
* Rank-based binary AUC

Numerical comparisons use `np.isclose`, while confusion matrices are compared using exact array equality.

---

# Decision Tree Tests

File:

```text
test_decision_tree.py
```

The Decision Tree test suite validates both normal behavior and edge cases.

Main functionality tested:

* Model fitting
* Label prediction
* Probability prediction
* Probability normalization
* Tree depth
* Number of leaves
* Feature importance
* Gini and entropy criteria
* Sample weights
* Reproducibility
* Model representation

The tests verify that:

* `predict()` returns labels with the correct shape.
* `predict_proba()` returns one probability per class.
* Every probability row sums to one.
* Predicted classes agree with the maximum probability.
* Tree depth respects `max_depth`.
* `min_samples_split` prevents invalid splits.
* Constant features produce a single leaf.
* A dataset containing one class is handled correctly.
* `max_features` supports `"sqrt"`, `"log2"`, and integer values.
* Feature importances sum to one when splits exist.
* Feature importances are zero when no valid split exists.

Several regression tests compare the custom implementation with scikit-learn's `DecisionTreeClassifier`.

The comparisons include:

* Training accuracy
* Tree depth
* Feature importance values

The XOR test confirms that the implementation can learn a nonlinear decision boundary when sufficient depth is available.

---

# AdaBoost Tests

File:

```text
test_adaboost.py
```

The AdaBoost tests validate the ensemble and its weak learner.

The test suite checks that `DecisionStump`:

* Inherits from the custom `DecisionTree`
* Uses `max_depth=1`
* Supports prediction
* Supports probability prediction

The `AdaBoostClassifier` tests cover:

* Fitting and prediction
* Probability output
* Probability normalization
* Estimator weights
* Estimator errors
* Staged prediction
* Learning-rate behavior
* One-estimator ensembles
* Large ensembles
* `fit()` returning `self`
* Informative `repr()` output

The staged prediction test verifies that one prediction array is produced after every trained weak learner.

The tests also check that:

```python
len(model.estimator_weights) == len(model.estimators_)
```

and that estimator weights are positive.

A comparison with scikit-learn's AdaBoost implementation verifies that the custom model remains within an acceptable accuracy difference.

---

# Random Forest Tests

File:

```text
test_random_forest.py
```

The Random Forest tests cover binary and multiclass classification.

The test suite validates:

* Number of trained trees
* Feature count
* Number of classes
* Prediction shape
* Probability shape
* Probability normalization
* Feature importance aggregation
* Out-of-bag scoring
* Parallel training
* Bootstrap and non-bootstrap modes
* Maximum tree depth
* Fitted class handling
* Model representation

The tests verify internal helper methods such as:

```python
_aggregate_feature_importances()
_aligned_proba()
_compute_oob_score()
```

The aligned-probability test is especially important for multiclass learning because individual trees may observe different class subsets during bootstrap sampling.

Property error tests confirm that:

* `predict()` cannot be used before fitting.
* `feature_importances_` is unavailable before fitting.
* `oob_score_` is unavailable when OOB scoring is disabled.
* `_fitted_classes` raises an error before model training.

Parallel training is checked using:

```python
RandomForestClassifier(
    n_estimators=4,
    n_jobs=2,
)
```

---

# PCA Tests

File:

```text
test_pca.py
```

The PCA tests use synthetic correlated features so that the expected principal-component structure is meaningful.

The tests validate:

* Output shape after transformation
* Component matrix shape
* Mean vector shape
* Explained variance ratio shape
* Non-negative explained variance
* Descending component importance
* Orthogonality of principal components
* Inverse transformation
* Invalid component counts
* Transforming before fitting

Principal-component orthonormality is checked using:

```python
components @ components.T
```

and comparing the result with an identity matrix.

The explained variance ratios are also verified to sum to at most one.

---

# K-Means Tests

File:

```text
test_kmeans.py
```

The K-Means tests generate three clearly separated synthetic clusters.

The implementation is expected to recover these clusters with an Adjusted Rand Index greater than:

```text
0.95
```

The tests cover:

* Cluster detection
* Centroid shape
* Non-negative inertia
* Prediction for unseen samples
* `fit_predict()` consistency
* Reproducibility
* Invalid cluster counts
* Prediction before fitting

Reproducibility is verified by training two models with the same random seed and comparing:

* Cluster labels
* Centroid coordinates

The tests also confirm that `n_clusters` cannot be zero or greater than the number of available samples.

---

# DBSCAN Tests

File:

```text
test_dbscan.py
```

The DBSCAN tests use two dense synthetic clusters and several distant noise points.

The tests verify that DBSCAN:

* Detects two clusters
* Identifies noise points
* Produces a high Adjusted Rand Index
* Stores core sample indices
* Stores the number of detected clusters
* Returns the same labels through `fit_predict()` and `labels_`
* Marks all points as noise when `eps` is extremely small
* Rejects invalid `eps`
* Rejects invalid `min_samples`

Noise samples are expected to receive the standard DBSCAN label:

```text
-1
```

---

# Bias-Variance Tests

File:

```text
test_bias_variance.py
```

These tests validate the complete bias-variance helper module.

The test suite covers:

* Bootstrap sampling
* Bootstrap reproducibility
* Preservation of all target classes
* Positive-class probability extraction
* Non-standard class ordering
* Exact bias and variance values
* Total-error decomposition
* Zero-variance predictions
* Binary-only validation
* Full experiment output

Exact-value tests verify:

```text
Total Error = Bias² + Variance
```

to a very small numerical tolerance.

Dummy AdaBoost and Random Forest models are used to isolate the experiment workflow from the actual model implementations.

This keeps the full experiment test:

* Fast
* Deterministic
* Independent of model complexity

The returned DataFrame is expected to contain one row for AdaBoost and one row for Random Forest.

---

# Unsupervised Helper Tests

File:

```text
test_unsupervised_helper.py
```

These tests validate the reusable workflows in `unsupervised_helper.py`.

The test suite covers:

* Numeric conversion
* Missing-value handling
* Removal of all-NaN columns
* Standardization
* Deterministic subsampling
* K-Means evaluation
* DBSCAN evaluation
* DBSCAN epsilon search
* PCA workflow execution
* Final DBSCAN visualization

Matplotlib display is disabled during tests using:

```python
plt.show = lambda *args, **kwargs: None
```

This prevents graphical windows from opening in automated environments.

The K-Means helper is expected to return a DataFrame containing:

```text
k, inertia, ARI
```

The DBSCAN epsilon-search helper must return:

```text
eps, clusters, ARI, noise_fraction
```

The best row must correspond to the maximum ARI value.

---

# Baseline Experiment Tests

File:

```text
test_experiment_baseline.py
```

This test validates the baseline supervised learning pipeline using a small synthetic binary dataset.

The workflow includes:

1. Train-test splitting
2. Standardization
3. Custom Decision Tree training
4. Decision Stump training
5. Scikit-learn Decision Tree training
6. Metric calculation
7. Accuracy comparison

The following metrics are validated:

* Accuracy
* Macro F1-score
* AUC-ROC

Each metric must remain between zero and one.

The custom Decision Tree accuracy is compared with the scikit-learn baseline using a maximum accepted difference of 10%.

---

# Head-to-Head Experiment Tests

File:

```text
test_experiments_head_to_head.py
```

This file provides lightweight tests for the main components used by the head-to-head experiment.

It checks:

* Custom metric calculations
* Decision Tree training
* Random Forest training
* AdaBoost training
* Probability normalization
* Stratified cross-validation

Binary and multiclass fixtures are provided for reusable test data.

The stratified split test verifies that:

* The requested number of folds is created.
* Training and testing indices do not overlap.
* The split can be used safely for model evaluation.

---

# AdaBoost Scaling Utility Tests

File:

```text
test_adaboost_experiment_utils.py
```

This suite validates the utilities used by the AdaBoost scaling experiments.

Tested functions include:

* `get_memory_usage`
* `run_adaboost_scaling_staged`
* `compare_with_sklearn`
* `save_dataset_results`
* `save_staged_predictions`
* `train_dataset_staged`
* `run_all_staged`

The tests cover memory calculation for:

* NumPy arrays
* pandas DataFrames
* pandas Series
* Unsupported objects
* Empty objects

The staged experiment tests verify that result lists have matching lengths and contain:

* Estimator counts
* Training accuracy
* Test accuracy
* Training F1-score
* Test F1-score

File-writing functions are tested with `unittest.mock.patch`, avoiding unnecessary disk operations.

Mocks are also used for:

* `StandardScaler`
* `train_test_split`
* AdaBoost models
* pandas DataFrames
* CSV output
* scikit-learn comparison models

---

# Full Experiment Runner Tests

File:

```text
test_run_all.py
```

These tests validate the automatic unsupervised experiment runner.

The test suite checks:

* Safe file-name generation
* Matplotlib figure saving
* Figure closing
* Task wrapper return values
* Mocked dataset loading
* Full workflow execution
* Required visualization generation

The real dataset loader is temporarily replaced with a small synthetic dataset.

This ensures that the test does not:

* Download external datasets
* Depend on internet access
* Consume excessive memory
* Require long runtimes

After execution, the test verifies that figures exist for:

* K-Means
* DBSCAN epsilon search
* PCA
* DBSCAN
* Best-epsilon DBSCAN visualization

Every generated PNG must exist and have a non-zero file size.

---

# Test Categories

The test suite includes several testing approaches.

## Unit Tests

Unit tests validate individual functions or methods in isolation.

Examples:

* Metric calculations
* PCA transformation
* K-Means parameter validation
* DBSCAN noise detection
* Bias-variance calculation

## Integration Tests

Integration tests validate multiple components working together.

Examples:

* Baseline model pipeline
* AdaBoost scaling workflow
* Full unsupervised experiment runner
* Bias-variance experiment output

## Regression Tests

Regression tests compare custom implementations with trusted scikit-learn behavior.

Examples:

* Decision Tree accuracy
* Decision Tree depth
* Feature importance
* AdaBoost accuracy
* Evaluation metrics

## Edge-Case Tests

Edge cases include:

* Constant features
* Single-class targets
* Empty clusters
* Invalid parameters
* Prediction before fitting
* Extremely small DBSCAN epsilon
* Disabled OOB scoring
* Zero prediction variance

## Mock-Based Tests

Mocking is used where real execution would be slow or involve filesystem operations.

Mocked components include:

* Model classes
* Dataset loading
* CSV writing
* DataFrames
* Standardization
* Train-test splitting

---

# Running the Tests

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Install pytest if it is not already available:

```bash
pip install pytest
```

Run the complete test suite from the project root:

```bash
pytest
```

Run with detailed output:

```bash
pytest -v
```

Run only the tests directory:

```bash
pytest tests/
```

Run one test file:

```bash
pytest tests/test_decision_tree.py -v
```

Run one specific test:

```bash
pytest tests/test_decision_tree.py::test_predict_basic -v
```

Stop after the first failure:

```bash
pytest -x
```

Show printed output during execution:

```bash
pytest -s
```

Run tests in quiet mode:

```bash
pytest -q
```

---

# Coverage

Code coverage can be measured using `pytest-cov`.

Install it with:

```bash
pip install pytest-cov
```

Run coverage for the `src` package:

```bash
pytest --cov=src --cov-report=term-missing
```

Generate an HTML coverage report:

```bash
pytest --cov=src --cov-report=html
```

The report will be generated in:

```text
htmlcov/index.html
```

---

# Expected Project Structure

```text
project-root/
├── src/
│   ├── experiments/
│   ├── metrics/
│   ├── trees/
│   ├── unsupervised/
│   └── utils/
├── tests/
│   ├── test_adaboost.py
│   ├── test_adaboost_experiment_utils.py
│   ├── test_bias_variance.py
│   ├── test_dbscan.py
│   ├── test_decision_tree.py
│   ├── test_evaluation.py
│   ├── test_experiment_baseline.py
│   ├── test_experiments_head_to_head.py
│   ├── test_kmeans.py
│   ├── test_pca.py
│   ├── test_random_forest.py
│   ├── test_run_all.py
│   ├── test_unsupervised_helper.py
│   └── README.md
├── figures/
├── requirements.txt
└── pyproject.toml
```

---

# Reproducibility

Most synthetic datasets use a fixed seed:

```python
random_state = 42
```

or:

```python
np.random.default_rng(42)
```

This ensures reproducible:

* Dataset generation
* Model initialization
* Bootstrap sampling
* Label generation
* Cluster generation
* Train-test splitting
* Cross-validation

Tests involving file generation clean or recreate the `figures` directory before execution.

---

# Test Design Principles

The suite follows these principles:

* Keep unit tests small and deterministic.
* Use synthetic data instead of downloading datasets.
* Compare custom implementations with established references.
* Test both successful and invalid workflows.
* Avoid graphical interaction during automated tests.
* Mock expensive models and filesystem operations.
* Validate shapes, ranges, fitted attributes, and numerical identities.
* Use fixed random seeds for reproducibility.

Together, these tests provide broad coverage of the machine learning models, experiment utilities, evaluation functions, and end-to-end project workflows.
