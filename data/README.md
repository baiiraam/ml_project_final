# Dataset Directory

This folder contains the datasets used for the ML final project.

## 1. Breast Cancer Wisconsin (Diagnostic)

**File:** `wdbc.data`

**Source:** UCI Machine Learning Repository

**Description:**
A binary classification dataset used to predict whether a breast tumor is malignant or benign based on features computed from digitized images of breast mass samples.

**Task:**

* Classification
* Target: Malignant / Benign

---

## 2. Adult Income

**Files:**

* `adult.data`
* `adult.test`

**Source:** UCI Machine Learning Repository

**Description:**
A classification dataset that predicts whether a person's annual income exceeds $50K based on demographic and employment-related attributes.

**Task:**

* Binary classification
* Target:

  * `<=50K`
  * `>50K`

---

## 3. Covertype

**Files:**

* `covtype.data.gz`
* `covtype.data` (extracted version)

**Source:** UCI Machine Learning Repository

**Description:**
A multi-class classification dataset used to predict forest cover type from cartographic variables.

**Task:**

* Multi-class classification
* Target: Forest cover type classes

---

## 4. MNIST (2-Class Subset)

**Loading method:**
MNIST is not stored as a static file in this directory.

It is downloaded automatically using Scikit-learn:

```python
from sklearn.datasets import fetch_openml

X, y = fetch_openml(
    'mnist_784',
    version=1,
    return_X_y=True,
    as_frame=False
)
```

**Description:**
A handwritten digit recognition dataset. The project uses a two-class subset for binary classification experiments.

**Task:**

* Image classification
* Target: Selected digit classes

---

## Dataset Download

Datasets can be downloaded by running:

```bash
bash download_data.sh
```

The script will create this directory structure:

```
data/
├── README.md
├── wdbc.data
├── adult.data
├── adult.test
├── covtype.data.gz
└── covtype.data
```

## Notes

* Dataset files are not modified after download.
* Preprocessing and feature engineering are handled in the project code.
* Original datasets are provided by the UCI Machine Learning Repository.
