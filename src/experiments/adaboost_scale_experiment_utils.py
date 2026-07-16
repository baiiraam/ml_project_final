import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.metrics.evaluation import accuracy_calculation, f1_score
from src.trees.boosting.adaboost import AdaBoostClassifier

import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier as SklearnAdaBoost
from sklearn.tree import DecisionTreeClassifier as SklearnDT


# ============================================
# DEFAULT CONFIGURATION
# ============================================
# These are the default values used when not overridden
MAX_ESTIMATORS = 20
STEP = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42
COVERTYPE_SAMPLE_SIZE = 5_000
MNIST_SAMPLE_SIZE = 500


def get_memory_usage(obj):
    """Get memory usage of pandas or numpy object in MB."""
    if hasattr(obj, "memory_usage"):
        mem = obj.memory_usage(deep=True)
        if hasattr(mem, "sum"):
            return mem.sum() / 1024**2
        else:
            return mem / 1024**2
    elif hasattr(obj, "nbytes"):
        return obj.nbytes / 1024**2
    else:
        return 0


def save_dataset_results(name, result, sample_size=None, X=None, y=None):
    """
    Save a single dataset's results to CSV immediately after training.

    Parameters:
    -----------
    name : str
        Dataset name
    result : dict
        Results dictionary from run_adaboost_scaling_staged()
    sample_size : int or str, optional
        Sample size used for this dataset
    X, y : optional
        Dataset arrays for metadata (features, classes)
    """
    os.makedirs("../notebooks", exist_ok=True)

    # 1. Save detailed results for this dataset
    df = pd.DataFrame(
        {
            "n_estimators": result["n_estimators"],
            "train_acc": result["train_acc"],
            "test_acc": result["test_acc"],
            "train_f1": result["train_f1"],
            "test_f1": result["test_f1"],
        }
    )
    filename = f"../notebooks/adaboost_scaling_{name}_details.csv"
    df.to_csv(filename, index=False)
    print(f"   Results saved: notebooks/adaboost_scaling_{name}_details.csv")

    # 2. Save metadata for this specific dataset
    best_idx = np.argmax(result["test_acc"])
    metadata = {
        "dataset": name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_estimators_tested": len(result["n_estimators"]),
        "best_n_estimators": result["n_estimators"][best_idx],
        "best_test_accuracy": result["test_acc"][best_idx],
        "best_test_f1": result["test_f1"][best_idx],
        "sample_size": sample_size if sample_size is not None else "full",
        "n_features": X.shape[1] if X is not None else "unknown",
        "n_classes": len(np.unique(y)) if y is not None else "unknown",
    }

    metadata_df = pd.DataFrame([metadata])
    metadata_df.to_csv(
        f"../notebooks/adaboost_scaling_{name}_metadata.csv", index=False
    )
    print(f"   Metadata saved: notebooks/adaboost_scaling_{name}_metadata.csv")


def run_adaboost_scaling_staged(X, y, dataset_name, max_estimators=None, step=None):
    """
    Run AdaBoost scaling experiment using staged_predict.
    Trains ONE model with max_estimators and records accuracy at each step.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target labels
    dataset_name : str
        Name of dataset for logging
    max_estimators : int, optional
        Maximum number of estimators. Uses CONFIG if None.
    step : int, optional
        Step size for varying n_estimators. Uses CONFIG if None.

    Returns:
    --------
    tuple: (results_dict, trained_model)
    """

    if max_estimators is None:
        max_estimators = MAX_ESTIMATORS
    if step is None:
        step = STEP

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Determine F1 averaging method
    if len(np.unique(y)) == 2:
        f1_mean = "binary"
    else:
        f1_mean = "macro"

    # Create list of rounds to record: [1, 5, 10, 15, ..., 200]
    n_estimators_list = [1] + list(range(step, max_estimators + 1, step))

    # Initialize result lists
    train_acc = []
    test_acc = []
    train_f1 = []
    test_f1 = []

    # Train ONE model with max_estimators
    print(f"  Training model with {max_estimators} estimators...")
    model = AdaBoostClassifier(
        n_estimators=max_estimators,
        learning_rate=1.0,
        criterion="gini",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_scaled, y_train)

    # Use staged_predict to record accuracy at each step
    # staged_predict yields predictions after: 1, 2, 3, 4, 5, ..., 200
    round_num = 0
    for pred_train, pred_test in zip(
        model.staged_predict(X_train_scaled), model.staged_predict(X_test_scaled)
    ):
        round_num += 1

        # Only record at specified rounds (1, 5, 10, 15, ..., 200)
        if round_num in n_estimators_list:
            train_acc.append(accuracy_calculation(y_train, pred_train))
            test_acc.append(accuracy_calculation(y_test, pred_test))
            train_f1.append(f1_score(y_train, pred_train, mean=f1_mean))
            test_f1.append(f1_score(y_test, pred_test, mean=f1_mean))

    results = {
        "dataset": dataset_name,
        "n_estimators": n_estimators_list,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "train_f1": train_f1,
        "test_f1": test_f1,
    }

    return results, model  # ← Returns BOTH results AND the trained model


def save_staged_predictions(X, y, dataset_name, max_estimators=None):
    """
    Save ALL staged predictions (all 200 rounds) to CSV.

    Parameters:
    -----------
    X, y : np.ndarray
        Features and labels
    dataset_name : str
        Name of dataset for logging
    max_estimators : int, optional
        Maximum number of estimators. Uses CONFIG if None.

    Returns:
    --------
    pd.DataFrame: Staged predictions data
    """

    if max_estimators is None:
        max_estimators = MAX_ESTIMATORS

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = AdaBoostClassifier(
        n_estimators=max_estimators,
        learning_rate=1.0,
        criterion="gini",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_scaled, y_train)

    if len(np.unique(y)) == 2:
        f1_mean = "binary"
    else:
        f1_mean = "macro"

    stages = []
    train_acc = []
    test_acc = []
    train_f1 = []
    test_f1 = []

    round_num = 0
    for pred_train, pred_test in zip(
        model.staged_predict(X_train_scaled), model.staged_predict(X_test_scaled)
    ):
        round_num += 1
        stages.append(round_num)
        train_acc.append(accuracy_calculation(y_train, pred_train))
        test_acc.append(accuracy_calculation(y_test, pred_test))
        train_f1.append(f1_score(y_train, pred_train, mean=f1_mean))
        test_f1.append(f1_score(y_test, pred_test, mean=f1_mean))

    staged_df = pd.DataFrame(
        {
            "round": stages,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "train_f1": train_f1,
            "test_f1": test_f1,
        }
    )

    os.makedirs("../notebooks", exist_ok=True)
    staged_df.to_csv(f"../notebooks/staged_predictions_{dataset_name}.csv", index=False)
    print(
        f"    Staged predictions saved: notebooks/staged_predictions_{dataset_name}.csv"
    )

    return staged_df


def train_dataset_staged(name, X, y, max_estimators=None, step=None, sample_size=None):
    """
    Train a single dataset using staged approach and save results immediately.
    """

    if max_estimators is None:
        max_estimators = MAX_ESTIMATORS
    if step is None:
        step = STEP

    print(f"[{time.strftime('%H:%M:%S')}] Starting {name}...")
    print(f"  Training 1 model with {max_estimators} estimators")
    print(f"  Recording at rounds: [1, {step}, {step * 2}, ..., {max_estimators}]")
    start = time.perf_counter()

    # Run the staged experiment - gets BOTH results AND model
    result, model = run_adaboost_scaling_staged(X, y, name, max_estimators, step)
    elapsed = time.perf_counter() - start
    print(f"[{time.strftime('%H:%M:%S')}] Finished {name}! ({elapsed:.2f}s)")

    # Save results
    print(f"  Saving results for {name}...")
    save_dataset_results(name, result, sample_size, X, y)

    # Also save all staged predictions (all 200 rounds)
    save_staged_predictions(X, y, name, max_estimators)

    # ← REMOVED: compare_with_sklearn(model, X, y, name)

    return name, result, model  # ← Now returns the model too


def run_all_staged(datasets, max_estimators=None, step=None):
    """
    Run AdaBoost scaling experiments on all datasets using staged approach.
    Saves results for each dataset as soon as it completes.
    Returns both results AND trained models.
    """

    if max_estimators is None:
        max_estimators = MAX_ESTIMATORS
    if step is None:
        step = STEP

    print("\nStarting staged execution...")
    print(f"Training ONE model per dataset with {max_estimators} estimators")
    print(f"Recording at steps: [1, {step}, {step * 2}, ..., {max_estimators}]")
    print("=" * 60)
    start = time.perf_counter()

    results = {}
    trained_models = {}  # ← NEW: Store models separately

    for name, (X, y) in datasets.items():
        # Determine sample size for logging
        if name == "Covertype":
            sample_size = (
                COVERTYPE_SAMPLE_SIZE if COVERTYPE_SAMPLE_SIZE is not None else len(X)
            )
        elif name == "MNIST":
            sample_size = MNIST_SAMPLE_SIZE if MNIST_SAMPLE_SIZE is not None else len(X)
        else:
            sample_size = len(X)

        # Train and save results for this dataset
        name, result, model = train_dataset_staged(
            name, X, y, max_estimators, step, sample_size
        )
        results[name] = result
        trained_models[name] = model  # ← Store the trained model
        print("-" * 40)

    total = time.perf_counter() - start
    print("=" * 60)
    print(f"All datasets complete! Total time: {total:.2f}s ({total / 60:.2f} minutes)")
    print("All results have been saved to CSV files.")

    return results, trained_models, total  # ← Now returns models too


def plot_adaboost_scaling(results, save_path=None):
    """
    Plot accuracy and F1 vs number of estimators.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    axes[0].plot(
        results["n_estimators"],
        results["train_acc"],
        label="Training",
        marker="o",
        markersize=4,
        linewidth=2,
    )
    axes[0].plot(
        results["n_estimators"],
        results["test_acc"],
        label="Test",
        marker="s",
        markersize=4,
        linewidth=2,
    )
    axes[0].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    axes[0].set_xlabel("Number of Estimators")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title(f"{results['dataset']} - Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # F1 Score plot
    axes[1].plot(
        results["n_estimators"],
        results["train_f1"],
        label="Training",
        marker="o",
        markersize=4,
        linewidth=2,
    )
    axes[1].plot(
        results["n_estimators"],
        results["test_f1"],
        label="Test",
        marker="s",
        markersize=4,
        linewidth=2,
    )
    axes[1].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    axes[1].set_xlabel("Number of Estimators")
    axes[1].set_ylabel("F1 Score")
    axes[1].set_title(f"{results['dataset']} - F1 Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs("../figures", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Figure saved: {save_path}")

    plt.show()


def compare_with_sklearn(model, X, y, dataset_name):
    """
    Compare our trained AdaBoost with sklearn's implementation.
    Uses the already trained model for our implementation.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # USE THE ALREADY TRAINED MODEL (no retraining!)
    our_acc = accuracy_calculation(y_test, model.predict(X_test_scaled))

    # sklearn still needs training (can't reuse our model)
    sk_model = SklearnAdaBoost(
        estimator=SklearnDT(max_depth=1),
        n_estimators=model.n_estimators,
        learning_rate=1.0,
        random_state=RANDOM_STATE,
    )
    sk_model.fit(X_train_scaled, y_train)
    sk_acc = accuracy_calculation(y_test, sk_model.predict(X_test_scaled))

    diff = abs(our_acc - sk_acc)

    print(f"  Our:      {our_acc:.4f}")
    print(f"  sklearn:  {sk_acc:.4f}")
    print(f"  Diff:     {diff:.4f}")

    if diff < 0.02:
        print("  OK: Within 2% tolerance")
    else:
        print("  WARNING: Difference > 2%")

    return our_acc, sk_acc
