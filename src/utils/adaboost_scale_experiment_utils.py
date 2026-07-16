"""
AdaBoost Scaling Experiment Utilities

This module provides utility functions for running AdaBoost scaling experiments
with the optimized adaboost_scale.py implementation.

Functions:
- get_memory_usage: Get memory usage of numpy/pandas arrays
- run_all_staged: Run AdaBoost scaling experiment using staged_predict
- plot_adaboost_scaling: Generate accuracy vs n_estimators plots
- compare_with_sklearn: Compare with sklearn's AdaBoost
"""

import time
import csv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.trees.boosting.adaboost_scale import AdaBoostClassifier


def get_memory_usage(arr):
    """
    Get memory usage of an array in MB.

    Supports both numpy arrays and pandas DataFrame/Series.

    Parameters
    ----------
    arr : numpy.ndarray or pandas.DataFrame/Series
        Array to measure memory usage

    Returns
    -------
    float
        Memory usage in MB
    """
    if hasattr(arr, "memory_usage"):
        mem = arr.memory_usage(deep=True)
        return mem.sum() / 1024**2 if hasattr(mem, "sum") else mem / 1024**2
    else:
        return arr.nbytes / 1024**2


def run_all_staged(
    datasets, max_estimators=200, step=5, random_state=42, test_size=0.2
):
    """
    Run AdaBoost scaling experiment on all datasets using staged_predict.

    This function trains AdaBoost once with max_estimators, then uses
    staged_predict to record performance at each specified step.

    Parameters
    ----------
    datasets : dict
        Dictionary of datasets {name: (X, y)}
    max_estimators : int, default=200
        Maximum number of estimators to train
    step : int, default=5
        Step size for recording results
    random_state : int, default=42
        Random seed for reproducibility
    test_size : float, default=0.2
        Train/test split ratio

    Returns
    -------
    results : dict
        Dictionary of results for each dataset with keys:
        - n_estimators: list of estimator counts
        - train_acc: training accuracy at each step
        - test_acc: test accuracy at each step
        - train_f1: training F1 at each step
        - test_f1: test F1 at each step
        - test_auc: test AUC at each step
        - time: cumulative training time at each step
        - total_train_time: total training time
    trained_models : dict
        Dictionary of trained models for each dataset
    total_time : float
        Total execution time in seconds
    """
    results = {}
    trained_models = {}

    # Create list of n_estimators to record
    n_estimators_list = [1] + list(range(step, max_estimators + 1, step))

    print(f"\nRunning AdaBoost scaling with n_estimators: {n_estimators_list}")
    print(f"Total recording points: {len(n_estimators_list)}")
    print("=" * 60)

    total_start_time = time.perf_counter()

    for name, (X, y) in datasets.items():
        print(f"\n{'=' * 50}")
        print(f"Processing {name}...")
        print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train AdaBoost with max estimators
        print("  Training AdaBoost with max estimators...")
        start_time = time.perf_counter()
        ab = AdaBoostClassifier(
            n_estimators=max_estimators,
            learning_rate=0.1,
            criterion="gini",
            random_state=random_state,
        )
        ab.fit(X_train_scaled, y_train)
        train_time = time.perf_counter() - start_time
        print(f"  Training complete: {train_time:.2f}s")

        # Get staged predictions
        print("  Extracting staged predictions...")
        train_stages = list(ab.staged_predict(X_train_scaled))
        test_stages = list(ab.staged_predict(X_test_scaled))

        # Record results at each specified n_estimators
        train_acc = []
        test_acc = []
        train_f1 = []
        test_f1 = []
        test_auc = []
        times = []

        # Get final probabilities for AUC
        final_proba = ab.predict_proba(X_test_scaled)
        n_classes = len(np.unique(y_test))
        if n_classes == 2:
            final_auc = roc_auc_score(y_test, final_proba[:, 1])
        else:
            final_auc = roc_auc_score(
                y_test, final_proba, multi_class="ovr", average="weighted"
            )

        for n_est in n_estimators_list:
            idx = n_est - 1
            if idx >= len(train_stages):
                idx = len(train_stages) - 1

            train_pred = train_stages[idx]
            test_pred = test_stages[idx]

            train_acc.append(accuracy_score(y_train, train_pred))
            test_acc.append(accuracy_score(y_test, test_pred))
            train_f1.append(f1_score(y_train, train_pred, average="weighted"))
            test_f1.append(f1_score(y_test, test_pred, average="weighted"))
            test_auc.append(final_auc)
            times.append(train_time * (n_est / max_estimators))

        # Store results
        results[name] = {
            "n_estimators": n_estimators_list,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "train_f1": train_f1,
            "test_f1": test_f1,
            "test_auc": test_auc,
            "time": times,
            "total_train_time": train_time,
        }
        trained_models[name] = ab

        # Save individual dataset results
        details_df = pd.DataFrame(
            {
                "n_estimators": n_estimators_list,
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "train_f1": train_f1,
                "test_f1": test_f1,
                "test_auc": test_auc,
                "train_time": times,
            }
        )
        details_df.to_csv(f"adaboost_scaling_{name}_details.csv", index=False)
        print(f"  ✅ Results saved to adaboost_scaling_{name}_details.csv")

        # Save metadata
        metadata = {
            "dataset": name,
            "samples": X.shape[0],
            "features": X.shape[1],
            "classes": len(np.unique(y)),
            "max_estimators": max_estimators,
            "step": step,
            "random_state": random_state,
            "test_size": test_size,
            "total_train_time": train_time,
            "n_estimators_recorded": len(n_estimators_list),
        }
        with open(f"adaboost_scaling_{name}_metadata.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["key", "value"])
            for key, value in metadata.items():
                writer.writerow([key, value])
        print(f"  ✅ Metadata saved to adaboost_scaling_{name}_metadata.csv")

    total_time = time.perf_counter() - total_start_time
    print(f"\n{'=' * 60}")
    print(f"✅ All experiments completed in {total_time:.2f} seconds")
    print("=" * 60)

    return results, trained_models, total_time


def plot_adaboost_scaling(result, dataset_name=None, save_path=None):
    """
    Generate accuracy vs n_estimators plot for AdaBoost scaling experiment.

    Parameters
    ----------
    result : dict
        Results dictionary from run_all_staged
    dataset_name : str, optional
        Name of dataset for plot title
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    axes : list
        List of axes objects
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    n_estimators = result["n_estimators"]
    train_acc = result["train_acc"]
    test_acc = result["test_acc"]

    # Plot 1: Accuracy vs n_estimators
    ax = axes[0]
    ax.plot(n_estimators, train_acc, "b-", label="Train Accuracy", linewidth=2)
    ax.plot(n_estimators, test_acc, "r-", label="Test Accuracy", linewidth=2)
    ax.set_xlabel("Number of Estimators")
    ax.set_ylabel("Accuracy")
    title = f"{dataset_name} - AdaBoost Scaling" if dataset_name else "AdaBoost Scaling"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.3, 1.05)

    # Plot 2: Overfitting gap
    ax = axes[1]
    gap = np.array(train_acc) - np.array(test_acc)
    ax.plot(n_estimators, gap, "g-", linewidth=2, marker="o", markersize=4)
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.5)
    ax.axhline(y=0.02, color="orange", linestyle="--", alpha=0.5, label="2% threshold")
    ax.axhline(y=0.05, color="red", linestyle="--", alpha=0.5, label="5% threshold")
    ax.fill_between(n_estimators, 0, gap, alpha=0.3, color="green", where=(gap > 0))
    ax.fill_between(n_estimators, 0, gap, alpha=0.3, color="red", where=(gap < 0))
    ax.set_xlabel("Number of Estimators")
    ax.set_ylabel("Train - Test Accuracy")
    ax.set_title("Overfitting Gap")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Plot saved to {save_path}")

    plt.show()

    return fig, axes


def compare_with_sklearn(model, X, y, dataset_name, random_state=42, test_size=0.2):
    """
    Compare AdaBoost implementation with sklearn's implementation.

    Parameters
    ----------
    model : AdaBoostClassifier
        Trained AdaBoost model
    X : np.ndarray
        Full dataset features
    y : np.ndarray
        Full dataset labels
    dataset_name : str
        Name of dataset for logging
    random_state : int, default=42
        Random seed
    test_size : float, default=0.2
        Test split ratio

    Returns
    -------
    our_acc : float
        Our implementation accuracy
    sk_acc : float
        sklearn implementation accuracy
    """
    from sklearn.ensemble import AdaBoostClassifier as SklearnAdaBoost
    from sklearn.tree import DecisionTreeClassifier

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Our implementation (already trained)
    our_pred = model.predict(X_test_scaled)
    our_acc = accuracy_score(y_test, our_pred)

    # sklearn implementation
    sk_ab = SklearnAdaBoost(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=len(model.estimators_),
        random_state=random_state,
        learning_rate=0.1,
    )
    sk_ab.fit(X_train_scaled, y_train)
    sk_pred = sk_ab.predict(X_test_scaled)
    sk_acc = accuracy_score(y_test, sk_pred)

    print(
        f"  {dataset_name}: Our={our_acc:.4f}, sklearn={sk_acc:.4f}, diff={abs(our_acc - sk_acc):.4f}"
    )

    return our_acc, sk_acc
