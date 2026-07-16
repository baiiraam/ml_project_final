# Utility Modules

The `utils` directory contains reusable helper functions used across the project for dataset loading, preprocessing, memory optimization, supervised robustness experiments, bias-variance analysis, and unsupervised learning evaluation.

These utilities reduce code duplication and provide a consistent workflow for the experiment scripts.

## Directory Contents

| File                      | Main Purpose                                                                              |
| ------------------------- | ----------------------------------------------------------------------------------------- |
| `preprocessing.py`        | Dataset loading, caching, train-test splitting, standardization, and memory optimization. |
| `noise_helper.py`         | Label-noise generation and robustness experiments for AdaBoost and Random Forest.         |
| `bias_variance_helper.py` | Bootstrap-based bias-variance decomposition for binary classification.                    |
| `unsupervised_helper.py`  | PCA, K-Means, and DBSCAN experiment workflows, parameter search, and visualization.       |

---

# 1. Preprocessing Utilities

File:

```text
preprocessing.py
```

This module provides data-loading and preprocessing functions for the four datasets used in the project:

* Breast Cancer
* Adult Income
* Covertype
* MNIST

It also supports local Parquet caching and automatic memory optimization.

## Project Root Detection

The function

```python
get_project_root()
```

detects the repository root using the location of `preprocessing.py`.

This allows datasets to be stored consistently inside:

```text
project-root/data/
```

regardless of the directory from which the script is executed.

## Standardization

```python
standardize(X_train, X_test)
```

standardizes the training and test sets using scikit-learn's `StandardScaler`.

The scaler is fitted only on the training data:

```python
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

This prevents information from the test set from leaking into the training process.

## Custom Train-Test Split

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=None,
)
```

creates a randomized split using NumPy.

The function:

1. Creates sample indices.
2. Shuffles them using `random_state`.
3. Separates the indices according to `test_size`.
4. Returns training and testing arrays.

Returned values:

```text
X_train, X_test, y_train, y_test
```

## Memory Optimization

The module contains three memory-management helpers.

### DataFrame Optimization

```python
optimize_dataframe_memory(df)
```

attempts to reduce DataFrame memory usage by:

* Converting numeric object columns when possible
* Downcasting integer columns
* Downcasting `float64` columns
* Skipping excluded columns
* Printing memory reduction statistics

### NumPy Array Optimization

```python
optimize_numpy_array(arr)
```

selects a smaller numeric dtype based on the minimum and maximum values.

For example:

* Small non-negative integers may become `uint8`
* Medium-sized integers may become `int16` or `int32`
* Floating-point arrays are normally converted to `float32`

### Series Optimization

```python
optimize_series_memory(series)
```

downcasts numeric pandas Series while preserving non-numeric values.

## Dataset Caching

The internal helper

```python
_load_parquet_or_fetch(...)
```

first checks whether a dataset already exists as a Parquet file.

If the file exists, it is loaded locally.

Otherwise, the dataset is:

1. Downloaded
2. Optimized
3. Combined into one DataFrame
4. Saved in Parquet format
5. Returned to the caller

This prevents repeated dataset downloads and reduces startup time for future experiments.

## Dataset Loaders

### Breast Cancer

```python
load_breast_cancer_data()
```

loads the Breast Cancer Wisconsin dataset from scikit-learn.

Returns:

```text
X_bc, y_bc, df_bc
```

### Adult Income

```python
load_adult_income_data(
    drop_categorical=True,
)
```

loads the Adult Income dataset from the UCI repository.

The target labels are cleaned by removing trailing periods.

When `drop_categorical=True`, categorical columns such as occupation, education, race, sex, and native country are removed.

Returns:

```text
X_adult, y_adult, df_adult
```

### Covertype

```python
load_covertype_data(
    drop_categorical=True,
)
```

loads the Covertype dataset from the UCI repository.

When categorical indicators are removed, the following groups are excluded:

* Wilderness-area binary columns
* Soil-type binary columns

Returns:

```text
X_cover, y_cover, df_cover
```

### MNIST

```python
load_mnist_data(
    return_numpy=False,
)
```

loads MNIST from OpenML and stores pixel columns as:

```text
pixel_0, pixel_1, ..., pixel_783
```

When `return_numpy=True`, feature and target values are returned as NumPy arrays.

Returns:

```text
X_mnist, y_mnist, df_mnist
```

---

# 2. Label Noise Utilities

File:

```text
noise_helper.py
```

This module evaluates how AdaBoost and Random Forest behave when the training labels contain artificial noise.

## Flipping Labels

```python
flip_labels(
    y,
    noise_fraction,
    random_state=42,
)
```

randomly selects a percentage of labels and replaces each selected label with another valid class.

The replacement class is always different from the original class.

The function returns:

```text
corrupted_labels, flipped_indices
```

Validation includes:

* `noise_fraction` must be between `0` and `1`
* At least two classes must be present

## Stratified Subsampling

```python
stratified_subsample(
    X,
    y,
    max_samples,
    random_state=42,
)
```

reduces large datasets while preserving class proportions.

If the dataset is already smaller than `max_samples`, it is returned unchanged.

## Noise Robustness Experiment

```python
run_noise_experiment(
    dataset_name,
    X,
    y,
    noise_levels,
    n_estimators=100,
    test_size=0.25,
    max_samples=None,
    random_state=42,
)
```

runs the full label-noise experiment.

For each noise level:

1. The dataset is optionally subsampled.
2. A stratified train-test split is created.
3. Training labels are corrupted.
4. AdaBoost is trained.
5. Random Forest is trained.
6. Accuracy is measured on the clean test set.
7. Results are stored in a pandas DataFrame.

The test labels are never corrupted.

This ensures that the measured degradation reflects the effect of noisy training data.

## Result Columns

The returned DataFrame includes:

| Column                      | Description                              |
| --------------------------- | ---------------------------------------- |
| `dataset`                   | Dataset name                             |
| `noise_fraction`            | Requested label-noise level              |
| `actual_noise_fraction`     | Real fraction of changed labels          |
| `train_samples`             | Number of training samples               |
| `test_samples`              | Number of test samples                   |
| `number_of_classes`         | Number of target classes                 |
| `flipped_labels`            | Number of modified labels                |
| `adaboost_accuracy`         | AdaBoost test accuracy                   |
| `random_forest_accuracy`    | Random Forest test accuracy              |
| `trained_stumps`            | Number of trained AdaBoost weak learners |
| `adaboost_degradation`      | Accuracy loss relative to zero noise     |
| `random_forest_degradation` | Accuracy loss relative to zero noise     |

## Degradation Curves

```python
plot_degradation_curves(results)
```

plots the accuracy degradation of AdaBoost and Random Forest as the noise level increases.

Each dataset receives a separate plot.

## Sensitivity Summary

```python
create_sensitivity_summary(results)
```

calculates the mean degradation for the standard noise levels:

```text
5%, 10%, and 20%
```

The output identifies which model is more sensitive for each dataset:

* AdaBoost
* Random Forest
* Equal

---

# 3. Bias-Variance Utilities

File:

```text
bias_variance_helper.py
```

This module performs bootstrap-based bias-variance decomposition for AdaBoost and Random Forest.

The current implementation supports binary classification only.

## Bootstrap Sampling

```python
generate_bootstrap_sample(
    X_train,
    y_train,
    random_state=None,
)
```

draws a bootstrap sample with replacement.

Sampling continues until all classes in the original training data are represented in the bootstrap sample.

This prevents binary models from receiving a sample containing only one class.

## Positive-Class Probabilities

```python
get_positive_class_probabilities(
    model,
    X,
    positive_class,
)
```

extracts the probability column corresponding to the selected positive class.

The function uses:

```python
model.classes_
```

instead of assuming that the positive class is always stored in the second column.

## Bias-Variance Calculation

```python
calculate_bias_variance(
    predictions,
    y_test,
)
```

expects probability predictions from multiple trained models.

If the predictions have shape:

```text
(number of bootstrap models, number of test samples)
```

the function calculates:

### Bias Squared

```text
Bias² = mean((mean prediction - true target)²)
```

### Variance

```text
Variance = mean((prediction - mean prediction)²)
```

### Total Error

```text
Total Error = mean((prediction - true target)²)
```

The function also verifies the decomposition:

```text
Total Error ≈ Bias² + Variance
```

and returns the absolute numerical difference.

## Full Experiment

```python
run_bias_variance_experiment(
    X_train,
    y_train,
    X_test,
    y_test,
    n_bootstrap=100,
    n_estimators=100,
    random_state=42,
    rf_n_jobs=-1,
)
```

performs the complete experiment.

For every bootstrap iteration:

1. A bootstrap dataset is generated.
2. AdaBoost is trained.
3. Random Forest is trained.
4. Positive-class probabilities are stored.
5. Classification accuracy is recorded.

After all iterations, the function calculates bias, variance, total error, mean accuracy, and ensemble accuracy.

## Returned Results

The function returns a DataFrame containing one row for AdaBoost and one for Random Forest.

Columns include:

* `model`
* `bias_squared`
* `variance`
* `total_error`
* `bias_plus_variance`
* `difference`
* `mean_accuracy`
* `ensemble_accuracy`

## Plotting

```python
plot_bias_variance(results)
```

creates a bar chart comparing:

* Bias squared
* Variance

for AdaBoost and Random Forest.

---

# 4. Unsupervised Experiment Utilities

File:

```text
unsupervised_helper.py
```

This module provides reusable workflows for PCA, K-Means, and DBSCAN analysis.

It combines the custom unsupervised models with preprocessing, evaluation metrics, and visualizations.

## Data Preparation

The internal function

```python
_prepare_unsupervised_data(
    X,
    y,
    max_points,
    drop_all_nan_columns=True,
)
```

performs the following steps:

1. Converts `X` to a DataFrame
2. Converts `y` to a one-dimensional Series
3. Converts feature values to numeric form
4. Replaces invalid values with `NaN`
5. Optionally removes completely empty columns
6. Fills missing values using column medians
7. Randomly samples large datasets
8. Standardizes features using `StandardScaler`

The sampling uses a fixed seed of `42`.

## PCA Task

```python
pca_task(
    dataset_name,
    X,
    y,
    n_clusters,
    dbscan_eps,
    dbscan_min_samples=5,
    max_points=5000,
)
```

performs a complete PCA-based analysis.

It:

* Fits PCA using all available components
* Calculates cumulative explained variance
* Produces a scree plot
* Projects the dataset into two dimensions
* Applies K-Means in PCA space
* Applies DBSCAN in PCA space
* Compares true, K-Means, and DBSCAN labels
* Reports variance explained by the first two components
* Reports the number of components required to preserve at least 90% variance

## K-Means Task

```python
kmeans_task(
    dataset_name,
    X,
    y,
    max_points=5000,
    n_init=10,
)
```

evaluates values of:

```text
k = 1, 2, ..., 10
```

For every `k`, the custom K-Means implementation is trained multiple times with different seeds.

The best run is selected using the smallest inertia.

The function then calculates:

* Inertia
* Adjusted Rand Index

It also generates an elbow-method plot and reports the `k` with the best ARI.

## DBSCAN Task

```python
dbscan_task(
    dataset_name,
    X,
    y,
    eps,
    min_samples=5,
    max_points=3000,
)
```

evaluates DBSCAN using a selected epsilon value.

It generates a k-distance plot and reports:

* Number of clusters
* Adjusted Rand Index
* Noise fraction
* Selected `eps`
* Selected `min_samples`

## DBSCAN Epsilon Search

```python
find_best_dbscan_eps(
    dataset_name,
    X,
    y,
    min_samples=5,
    max_points=3000,
)
```

automatically creates candidate `eps` values from percentiles of the sorted k-nearest-neighbor distances.

For each candidate, the function evaluates:

* Number of clusters
* ARI
* Noise fraction

The row with the highest ARI is selected as the best configuration.

## Final DBSCAN Visualization

```python
plot_dbscan_best_eps(
    dataset_name,
    X,
    y,
    best_eps,
    min_samples=5,
    max_points=3000,
)
```

runs DBSCAN with the selected epsilon value and visualizes the clusters in a two-dimensional PCA representation.

Noise points remain represented by the label:

```text
-1
```

---

# Dependencies

The utility modules use the following packages:

```text
numpy
pandas
matplotlib
scikit-learn
ucimlrepo
IPython
pyarrow
```

Install all project dependencies with:

```bash
pip install -r requirements.txt
```

Parquet caching requires a compatible engine such as:

```bash
pip install pyarrow
```

---

# Expected Directory Structure

```text
project-root/
├── data/
├── figures/
├── experiments/
└── src/
    ├── metrics/
    │   └── evaluation.py
    ├── trees/
    │   ├── bagging/
    │   │   └── random_forest.py
    │   └── boosting/
    │       └── adaboost.py
    ├── unsupervised/
    │   ├── pca.py
    │   ├── kmeans.py
    │   └── dbscan.py
    └── utils/
        ├── preprocessing.py
        ├── noise_helper.py
        ├── bias_variance_helper.py
        ├── unsupervised_helper.py
        └── README.md
```

## Reproducibility

Most random operations use:

```python
random_state=42
```

This includes:

* Dataset splitting
* Subsampling
* Label corruption
* Bootstrap sampling
* K-Means restarts
* AdaBoost training
* Random Forest training

Using a fixed seed improves consistency across repeated experiment runs.
