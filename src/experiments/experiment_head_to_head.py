# ruff: noqa: E402
# === 1. IMPORTS & SETUP ===
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier as SkRF
from sklearn.model_selection import StratifiedKFold

# Path Ayarları
project_root = Path.cwd().parent
sys.path.insert(0, str(project_root))

# Custom Modul Importları
from src.metrics.evaluation import accuracy_calculation, auc_roc, f1_score
from src.trees.bagging.random_forest import RandomForestClassifier
from src.trees.boosting.adaboost import AdaBoostClassifier
from src.trees.decision_tree import DecisionTree
from src.utils.preprocessing import (
    load_adult_income_data,
    load_breast_cancer,
    load_covertype_data,
    load_mnist_data,
)

# Konfiqurasiya
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

matplotlib.rcParams["figure.dpi"] = 120
os.makedirs("../figures", exist_ok=True)

# === 2. DATASET LOADING & PREPROCESSING ===

# Breast Cancer
data_bc = load_breast_cancer()
X_bc = np.ascontiguousarray(data_bc.data, dtype=np.float32)
y_bc = np.ascontiguousarray(data_bc.target, dtype=np.int64).ravel()

# Adult Income
X_adult, y_adult, _ = load_adult_income_data()
adult_labels = y_adult.iloc[:, 0] if hasattr(y_adult, "columns") else y_adult
adult_labels = adult_labels.astype(str).str.strip().str.rstrip(".")
adult_target = adult_labels.map({"<=50K": 0, ">50K": 1})
adult_valid_mask = adult_target.notna().to_numpy()

X_adult_clean = np.ascontiguousarray(X_adult, dtype=np.float32)[adult_valid_mask]
y_adult_clean = np.ascontiguousarray(
    adult_target[adult_valid_mask].to_numpy(), dtype=np.int64
)

# Covertype (İlk 1000 nümunə)
A, b, _ = load_covertype_data()
X_cover = np.ascontiguousarray(A[:1000], dtype=np.float32)
y_cover = np.ascontiguousarray(b.values[:1000], dtype=np.int64).ravel()
y_cover = y_cover - y_cover.min()  # 0-6 formatına çevirmə

# MNIST (İlk 1000 nümunə)
M, n, _ = load_mnist_data()
X_mnist = np.ascontiguousarray(M[:1000], dtype=np.float32)
y_mnist = np.ascontiguousarray(n.values[:1000], dtype=np.int64).ravel()

# Datasetlərin lüğəti
datasets = {
    "Breast Cancer": (X_bc, y_bc),
    "Adult Income": (X_adult_clean, y_adult_clean),
    "Covertype": (X_cover, y_cover),
    "MNIST": (X_mnist, y_mnist),
}

# Dataset info çapı
for name, (X_d, y_d) in datasets.items():
    print(f"\n{name} initialized -> X: {X_d.shape}, y: {y_d.shape}")
    print(f"Classes: {np.unique(y_d)} | Distribution: {np.bincount(y_d)}")

# === 3. MODEL EVALUATION FUNCTIONS ===


def calculate_metrics(model, X_test, y_test):
    y_test = np.asarray(y_test).ravel()
    predictions = np.asarray(model.predict(X_test)).ravel()
    probabilities = np.asarray(model.predict_proba(X_test), dtype=np.float64)

    if probabilities.ndim != 2 or probabilities.shape[0] != y_test.shape[0]:
        raise ValueError("predict_proba() çıxışı düzgün ölçüdə deyil.")

    return {
        "accuracy": float(accuracy_calculation(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, mean="macro")),
        "auc_roc": float(auc_roc(y_test, probabilities)),
    }


def run_model_on_fold(
    dataset_name, fold, train_idx, test_idx, X_data, y_data, model_name
):
    X_train, X_test = X_data[train_idx], X_data[test_idx]
    y_train, y_test = y_data[train_idx], y_data[test_idx]
    model_seed = RANDOM_STATE + fold

    if model_name == "Single Tree":
        model = DecisionTree(random_state=model_seed)
    elif model_name == "AdaBoost":
        model = AdaBoostClassifier(n_estimators=100, random_state=model_seed)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=100, random_state=model_seed, n_jobs=1
        )
    elif model_name == "sklearn RF":
        model = SkRF(n_estimators=100, random_state=model_seed, n_jobs=1)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.fit(X_train, y_train)
    metrics = calculate_metrics(model, X_test, y_test)

    return {
        "dataset": dataset_name,
        "fold": fold,
        "model": model_name,
        **metrics,
    }


# === 4. PARALLEL CROSS-VALIDATION RUNNER ===

MODEL_NAMES = ("Single Tree", "AdaBoost", "Random Forest", "sklearn RF")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

cpu_count = os.cpu_count() or 2
N_JOBS = max(1, min(cpu_count - 1, 6))
print(f"\nDetected CPUs: {cpu_count} | Parallel workers: {N_JOBS}")

all_fold_results = []

for dataset_name, (X_data, y_data) in datasets.items():
    classes, counts = np.unique(y_data, return_counts=True)
    if counts.min() < 5:
        raise ValueError(f"{dataset_name}: Hər class-da ən azı 5 nümunə olmalıdır.")

    print(f"\nProcessing: {dataset_name} " + "=" * 40)

    folds = list(skf.split(X_data, y_data))
    tasks = [
        (fold, train_idx, test_idx, model_name)
        for fold, (train_idx, test_idx) in enumerate(folds, start=1)
        for model_name in MODEL_NAMES
    ]

    dataset_results = Parallel(
        n_jobs=N_JOBS,
        backend="loky",
        max_nbytes="20M",
        mmap_mode="r",
        batch_size=1,
        pre_dispatch=N_JOBS,
        verbose=5,
    )(
        delayed(run_model_on_fold)(
            dataset_name=dataset_name,
            fold=fold,
            train_idx=train_idx,
            test_idx=test_idx,
            X_data=X_data,
            y_data=y_data,
            model_name=model_name,
        )
        for (fold, train_idx, test_idx, model_name) in tasks
    )

    all_fold_results.extend(dataset_results)

# DataFrame qurulması və sıralanması
fold_results_df = pd.DataFrame(all_fold_results)
model_order = {name: idx for idx, name in enumerate(MODEL_NAMES)}
fold_results_df["model_order"] = fold_results_df["model"].map(model_order)
fold_results_df = (
    fold_results_df.sort_values(by=["dataset", "model_order", "fold"])
    .drop(columns="model_order")
    .reset_index(drop=True)
)

print(f"\nTəcrübələr tamamlandı. Toplam nəticə sayı: {len(fold_results_df)}")

# === 5. SUMMARY TABLES GENERATION ===

summary_rows = []
METRIC_COLUMNS = ["accuracy", "macro_f1", "auc_roc"]

for (dataset_name, model_name), group in fold_results_df.groupby(
    ["dataset", "model"], sort=False
):
    row = {"Dataset": dataset_name, "Model": model_name}
    for col in METRIC_COLUMNS:
        vals = group[col].to_numpy(dtype=float)
        row[col] = f"{np.mean(vals):.4f} ± {np.std(vals, ddof=1):.4f}"
    summary_rows.append(row)

summary_table = pd.DataFrame(summary_rows)
summary_table["model_order"] = summary_table["Model"].map(model_order)
summary_table = (
    summary_table.sort_values(by=["Dataset", "model_order"])
    .drop(columns="model_order")
    .rename(
        columns={
            "accuracy": "Accuracy",
            "macro_f1": "Macro F1",
            "auc_roc": "AUC-ROC",
        }
    )
    .reset_index(drop=True)
)

# Dataset üzrə nəticələri ekrana çıxarmaq
for dataset_name in summary_table["Dataset"].unique():
    print(f"\nNəticələr: {dataset_name} " + "-" * 30)
    print(
        summary_table[summary_table["Dataset"] == dataset_name].drop(columns="Dataset")
    )

# === 6. PLOTTING FUNCTIONS ===

PLOT_COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
metrics_info = [
    ("accuracy", "Accuracy"),
    ("macro_f1", "Macro F1"),
    ("auc_roc", "AUC-ROC"),
]


def plot_results(df, plot_type="box"):
    """Plot_type olaraq 'box' və ya 'violin' qəbul edir və qrafikləri yadda saxlayır."""
    for dataset_name in df["dataset"].unique():
        sub_df = df[df["dataset"] == dataset_name]
        models = sub_df["model"].unique().tolist()

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))

        for ax, (metric_col, metric_title) in zip(axes, metrics_info):
            data = [
                sub_df[sub_df["model"] == m][metric_col].astype(float).values
                for m in models
            ]

            if plot_type == "box":
                bp = ax.boxplot(data, tick_labels=models, patch_artist=True)
                for patch, color in zip(bp["boxes"], PLOT_COLORS):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
            elif plot_type == "violin":
                vp = ax.violinplot(
                    data, showmeans=True, showmedians=True, showextrema=True
                )
                for body, color in zip(vp["bodies"], PLOT_COLORS):
                    body.set_facecolor(color)
                    body.set_edgecolor("black")
                    body.set_alpha(0.7)
                ax.set_xticks(range(1, len(models) + 1))
                ax.set_xticklabels(models)

            ax.set_title(metric_title, fontsize=12, fontweight="bold")
            ax.set_ylabel("Score")
            ax.grid(axis="y", alpha=0.3)
            ax.tick_params(axis="x", rotation=25)

        fig.suptitle(
            f"Head-to-Head ({plot_type.capitalize()}): {dataset_name} (5-fold CV)",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        safe_name = dataset_name.lower().replace(" ", "_")
        save_path = f"../figures/head_to_head_{plot_type}_{safe_name}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"{plot_type.capitalize()} plot saved to: {save_path}")


# Qrafikləri çək və saxla
plot_results(fold_results_df, plot_type="box")
plot_results(fold_results_df, plot_type="violin")
