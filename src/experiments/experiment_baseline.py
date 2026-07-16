import os
import sys

# Ruff linterinin sys.path manipulyasiyasından sonrakı importlara irad bildirməməsi üçün
sys.path.insert(0, os.path.join(os.getcwd(), ".."))

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.tree import DecisionTreeClassifier as SkDT  # noqa: E402

from src.metrics.evaluation import accuracy_calculation, auc_roc, f1_score  # noqa: E402
from src.trees.boosting.adaboost import DecisionStump  # noqa: E402
from src.trees.decision_tree import DecisionTree  # noqa: E402
from src.utils.preprocessing import (  # noqa: E402
    load_adult_income_data,
    load_breast_cancer,
    load_covertype_data,
    load_mnist_data,
    standardize,
    train_test_split,
)

matplotlib.use("Agg")
matplotlib.rcParams["figure.dpi"] = 120

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
data = load_breast_cancer()
X, y = data.data, data.target
print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
print(f"Classes: {np.unique(y)} ({len(np.unique(y))} classes)")
print(f"Class distribution: {np.bincount(y)}")

X_adult, y_adult, df_adult = load_adult_income_data()
Z = X_adult
t = (
    y_adult.str.strip()
    .map(
        {
            "<=50K": 0,
            ">50K": 1,
        }
    )
    .values
)
print(f"Samples: {Z.shape[0]}, Features: {Z.shape[1]}")
print(f"Classes: {np.unique(t)} ({len(np.unique(t))} classes)")
print(f"Class distribution: {np.bincount(t)}")

A, b, df_cover = load_covertype_data()
b = b.values
print(f"Samples: {A.shape[0]}, Features: {A.shape[1]}")
print(f"Classes: {np.unique(b)} ({len(np.unique(b))} classes)")
print(f"Class distribution: {np.bincount(b)}")

M, n, df_mnist = load_mnist_data()
n = n.values
print(f"Samples: {M.shape[0]}, Features: {M.shape[1]}")
print(f"Classes: {np.unique(n)} ({len(np.unique(n))} classes)")
print(f"Class distribution: {np.bincount(n)}")

datasets = {
    "Breast Cancer": (np.asarray(X), np.asarray(y)),
    "Adult Income": (np.asarray(Z), np.asarray(t)),
    "Covertype": (np.asarray(A), np.asarray(b).ravel()),
    "MNIST": (np.asarray(M), np.asarray(n)),
}

all_baseline_results = {}

for name, (X_data, y_data) in datasets.items():
    print(f"\n{'=' * 60}")
    print(name)
    print(f"{'=' * 60}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=RANDOM_STATE
    )

    X_train_s, X_test_s = standardize(X_train, X_test)

    results = {}

    dt = DecisionTree(random_state=RANDOM_STATE)
    dt.fit(X_train_s, y_train)

    pred = dt.predict(X_test_s)
    proba = dt.predict_proba(X_test_s)

    results["DT"] = {
        "acc": accuracy_calculation(y_test, pred),
        "f1": f1_score(y_test, pred, mean="macro"),
        "auc": auc_roc(y_test, proba),
    }

    stump = DecisionStump(random_state=RANDOM_STATE)
    stump.fit(X_train_s, y_train)

    pred_s = stump.predict(X_test_s)
    proba_s = stump.predict_proba(X_test_s)

    results["Stump"] = {
        "acc": accuracy_calculation(y_test, pred_s),
        "f1": f1_score(y_test, pred_s, mean="macro"),
        "auc": auc_roc(y_test, proba_s),
    }

    sk_dt = SkDT(random_state=RANDOM_STATE)
    sk_dt.fit(X_train_s, y_train)

    sk_pred = sk_dt.predict(X_test_s)
    sk_proba = sk_dt.predict_proba(X_test_s)

    results["sklearn DT"] = {
        "acc": accuracy_calculation(y_test, sk_pred),
        "f1": f1_score(y_test, sk_pred, mean="macro"),
        "auc": auc_roc(y_test, sk_proba),
    }

    print(
        f"DecisionTree (ours): "
        f"Acc={results['DT']['acc']:.4f}, "
        f"F1={results['DT']['f1']:.4f}, "
        f"AUC={results['DT']['auc']:.4f}"
    )

    print(
        f"DecisionStump:      "
        f"Acc={results['Stump']['acc']:.4f}, "
        f"F1={results['Stump']['f1']:.4f}, "
        f"AUC={results['Stump']['auc']:.4f}"
    )

    print(
        f"sklearn Tree:       "
        f"Acc={results['sklearn DT']['acc']:.4f}, "
        f"F1={results['sklearn DT']['f1']:.4f}, "
        f"AUC={results['sklearn DT']['auc']:.4f}"
    )

    diff_pct = (
        abs(results["DT"]["acc"] - results["sklearn DT"]["acc"])
        / max(results["sklearn DT"]["acc"], 1e-10)
        * 100
    )

    status = "PASS" if diff_pct <= 2 else "FAIL"

    print(f"Accuracy difference vs sklearn: {diff_pct:.2f}% ({status})")
    all_baseline_results[name] = results


def plot_baseline_results(results_dict):
    datasets_list = list(results_dict.keys())
    models = ["DT", "Stump", "sklearn DT"]
    metrics = ["acc", "f1", "auc"]
    metric_names = ["Accuracy", "F1-Score", "AUC-ROC"]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 16), sharex=False)

    x = np.arange(len(datasets_list))
    width = 0.25

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        for m_idx, model in enumerate(models):
            means = [results_dict[dataset][model][metric] for dataset in datasets_list]
            rects = ax.bar(
                x + (m_idx - 1) * width,
                means,
                width,
                label=model,
                color=colors[m_idx],
                alpha=0.85,
            )
            ax.bar_label(rects, padding=3, fmt="%.2f", fontsize=9)

        ax.set_ylabel(metric_names[idx], fontsize=12, fontweight="bold")
        ax.set_title(
            f"Models Comparison based on {metric_names[idx]}",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(datasets_list, fontsize=11)
        ax.set_ylim(0, 1.15)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        if idx == 0:
            ax.legend(loc="upper right", fontsize=11)

    plt.tight_layout()

    figures_dir = os.path.join(os.getcwd(), "..", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    save_path = os.path.join(figures_dir, "baseline_comparison.png")
    plt.savefig(save_path)
    plt.close()


plot_baseline_results(all_baseline_results)
