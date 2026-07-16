from __future__ import annotations
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.trees.bagging.random_forest import RandomForestClassifier
from src.trees.boosting.adaboost import AdaBoostClassifier


def flip_labels(
    y: np.ndarray,
    noise_fraction: float,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y).ravel()

    if not 0.0 <= noise_fraction <= 1.0:
        raise ValueError("noise_fraction must be between 0 and 1")

    classes = np.unique(y)

    if len(classes) < 2:
        raise ValueError("At least two classes are required")

    rng = np.random.RandomState(random_state)
    y_corrupted = y.copy()

    n_flip = int(round(noise_fraction * len(y)))

    if n_flip == 0:
        return y_corrupted, np.array([], dtype=np.int64)

    flip_indices = rng.choice(
        len(y),
        size=n_flip,
        replace=False,
    )

    for index in flip_indices:
        alternative_classes = classes[classes != y_corrupted[index]]

        y_corrupted[index] = rng.choice(alternative_classes)

    return y_corrupted, flip_indices


def stratified_subsample(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: Optional[int],
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if max_samples is None or len(y) <= max_samples:
        return X, y

    X_subset, _, y_subset, _ = train_test_split(
        X,
        y,
        train_size=max_samples,
        random_state=random_state,
        stratify=y,
    )

    return X_subset, y_subset


def run_noise_experiment(
    dataset_name: str,
    X: np.ndarray,
    y: np.ndarray,
    noise_levels: list[float],
    n_estimators: int = 100,
    test_size: float = 0.25,
    max_samples: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel()

    X, y = stratified_subsample(
        X=X,
        y=y,
        max_samples=max_samples,
        random_state=random_state,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    rows = []

    for eta in noise_levels:
        print(f"\n{dataset_name} | eta={eta:.2f}")

        y_corrupted, flipped_indices = flip_labels(
            y_train,
            noise_fraction=eta,
            random_state=random_state,
        )

        adaboost = AdaBoostClassifier(
            n_estimators=n_estimators,
            learning_rate=1.0,
            criterion="gini",
            random_state=random_state,
        )

        random_forest = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            max_features="sqrt",
            min_samples_split=2,
            criterion="gini",
            bootstrap=True,
            oob_score=False,
            n_jobs=-1,
            random_state=random_state,
        )

        adaboost.fit(X_train, y_corrupted)
        random_forest.fit(X_train, y_corrupted)

        adaboost_predictions = adaboost.predict(X_test)
        random_forest_predictions = random_forest.predict(X_test)

        adaboost_accuracy = accuracy_score(
            y_test,
            adaboost_predictions,
        )

        random_forest_accuracy = accuracy_score(
            y_test,
            random_forest_predictions,
        )

        rows.append(
            {
                "dataset": dataset_name,
                "noise_fraction": eta,
                "actual_noise_fraction": float(np.mean(y_corrupted != y_train)),
                "train_samples": len(y_train),
                "test_samples": len(y_test),
                "number_of_classes": len(np.unique(y)),
                "flipped_labels": len(flipped_indices),
                "adaboost_accuracy": adaboost_accuracy,
                "random_forest_accuracy": random_forest_accuracy,
                "trained_stumps": len(adaboost.estimators_),
            }
        )

        print(f"Flipped labels: {len(flipped_indices)}")
        print(f"AdaBoost accuracy: {adaboost_accuracy:.4f}")
        print(f"Random Forest accuracy: {random_forest_accuracy:.4f}")

    result = pd.DataFrame(rows)

    adaboost_baseline = result.loc[
        result["noise_fraction"] == 0.0,
        "adaboost_accuracy",
    ].iloc[0]

    random_forest_baseline = result.loc[
        result["noise_fraction"] == 0.0,
        "random_forest_accuracy",
    ].iloc[0]

    result["adaboost_degradation"] = adaboost_baseline - result["adaboost_accuracy"]

    result["random_forest_degradation"] = (
        random_forest_baseline - result["random_forest_accuracy"]
    )

    return result


def plot_degradation_curves(
    results: pd.DataFrame,
) -> None:
    for dataset_name in results["dataset"].unique():
        subset = results[results["dataset"] == dataset_name].sort_values(
            "noise_fraction"
        )

        plt.figure(figsize=(8, 5))

        plt.plot(
            subset["noise_fraction"],
            subset["adaboost_degradation"],
            marker="o",
            linewidth=2,
            label="AdaBoost",
        )

        plt.plot(
            subset["noise_fraction"],
            subset["random_forest_degradation"],
            marker="s",
            linewidth=2,
            label="Random Forest",
        )

        plt.xlabel("Training label noise fraction (η)")
        plt.ylabel("Accuracy degradation")
        plt.title(f"Accuracy Degradation — {dataset_name}")

        plt.xticks(
            subset["noise_fraction"],
            [f"{eta:.0%}" for eta in subset["noise_fraction"]],
        )

        plt.axhline(0, linewidth=1)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


def create_sensitivity_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        results[results["noise_fraction"].isin([0.05, 0.10, 0.20])]
        .groupby("dataset")[
            [
                "adaboost_degradation",
                "random_forest_degradation",
            ]
        ]
        .mean()
        .reset_index()
        .rename(
            columns={
                "adaboost_degradation": "adaboost_mean_degradation",
                "random_forest_degradation": "random_forest_mean_degradation",
            }
        )
    )

    summary["more_sensitive_model"] = np.where(
        summary["adaboost_mean_degradation"]
        > summary["random_forest_mean_degradation"],
        "AdaBoost",
        np.where(
            summary["random_forest_mean_degradation"]
            > summary["adaboost_mean_degradation"],
            "Random Forest",
            "Equal",
        ),
    )

    return summary
