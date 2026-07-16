from __future__ import annotations
from typing import Any, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.metrics.evaluation import accuracy_calculation
from src.trees.bagging.random_forest import RandomForestClassifier
from src.trees.boosting.adaboost import AdaBoostClassifier


def generate_bootstrap_sample(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).ravel()

    rng = np.random.RandomState(random_state)

    while True:
        indices = rng.choice(
            len(y_train),
            size=len(y_train),
            replace=True,
        )

        X_bootstrap = X_train[indices]
        y_bootstrap = y_train[indices]

        if len(np.unique(y_bootstrap)) == len(np.unique(y_train)):
            return X_bootstrap, y_bootstrap


def get_positive_class_probabilities(
    model: Any,
    X: np.ndarray,
    positive_class: Any,
) -> np.ndarray:
    probabilities = np.asarray(
        model.predict_proba(X),
        dtype=np.float64,
    )

    classes = np.asarray(model.classes_)

    positive_index = int(np.flatnonzero(classes == positive_class)[0])

    return probabilities[:, positive_index]


def calculate_bias_variance(
    predictions: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    predictions = np.asarray(
        predictions,
        dtype=np.float64,
    )

    y_test = np.asarray(
        y_test,
        dtype=np.float64,
    ).ravel()

    mean_prediction = predictions.mean(axis=0)

    bias_squared = float(np.mean((mean_prediction - y_test) ** 2))

    variance = float(np.mean((predictions - mean_prediction[None, :]) ** 2))

    total_error = float(np.mean((predictions - y_test[None, :]) ** 2))

    bias_plus_variance = bias_squared + variance

    return {
        "bias_squared": bias_squared,
        "variance": variance,
        "total_error": total_error,
        "bias_plus_variance": (bias_plus_variance),
        "difference": abs(total_error - bias_plus_variance),
    }


def run_bias_variance_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_bootstrap: int = 100,
    n_estimators: int = 100,
    random_state: int = 42,
    rf_n_jobs: int = -1,
) -> pd.DataFrame:
    X_train = np.asarray(
        X_train,
        dtype=np.float64,
    )

    X_test = np.asarray(
        X_test,
        dtype=np.float64,
    )

    y_train = np.asarray(y_train).ravel()
    y_test = np.asarray(y_test).ravel()

    classes = np.unique(y_train)

    if len(classes) != 2:
        raise ValueError("Only binary classification is supported.")

    positive_class = classes[1]
    negative_class = classes[0]

    y_test_binary = (y_test == positive_class).astype(np.float64)

    ada_probabilities: list[np.ndarray] = []
    rf_probabilities: list[np.ndarray] = []

    ada_accuracies: list[float] = []
    rf_accuracies: list[float] = []

    for bootstrap_id in range(n_bootstrap):
        seed = random_state + bootstrap_id

        X_bootstrap, y_bootstrap = generate_bootstrap_sample(
            X_train,
            y_train,
            seed,
        )

        adaboost = AdaBoostClassifier(
            n_estimators=n_estimators,
            learning_rate=1.0,
            criterion="gini",
            random_state=seed,
        )

        random_forest = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            max_features="sqrt",
            min_samples_split=2,
            criterion="gini",
            bootstrap=True,
            oob_score=False,
            n_jobs=rf_n_jobs,
            random_state=seed,
        )

        adaboost.fit(
            X_bootstrap,
            y_bootstrap,
        )

        random_forest.fit(
            X_bootstrap,
            y_bootstrap,
        )

        ada_probability = get_positive_class_probabilities(
            adaboost,
            X_test,
            positive_class,
        )

        rf_probability = get_positive_class_probabilities(
            random_forest,
            X_test,
            positive_class,
        )

        ada_probabilities.append(ada_probability)

        rf_probabilities.append(rf_probability)

        ada_prediction = adaboost.predict(X_test)

        rf_prediction = random_forest.predict(X_test)

        ada_accuracies.append(
            float(
                accuracy_calculation(
                    y_test,
                    ada_prediction,
                )
            )
        )

        rf_accuracies.append(
            float(
                accuracy_calculation(
                    y_test,
                    rf_prediction,
                )
            )
        )

    ada_probabilities_array = np.asarray(
        ada_probabilities,
        dtype=np.float64,
    )

    rf_probabilities_array = np.asarray(
        rf_probabilities,
        dtype=np.float64,
    )

    ada_result = calculate_bias_variance(
        ada_probabilities_array,
        y_test_binary,
    )

    rf_result = calculate_bias_variance(
        rf_probabilities_array,
        y_test_binary,
    )

    ada_mean_probability = ada_probabilities_array.mean(axis=0)

    rf_mean_probability = rf_probabilities_array.mean(axis=0)

    ada_ensemble_prediction = np.where(
        ada_mean_probability >= 0.5,
        positive_class,
        negative_class,
    )

    rf_ensemble_prediction = np.where(
        rf_mean_probability >= 0.5,
        positive_class,
        negative_class,
    )

    return pd.DataFrame(
        [
            {
                "model": "AdaBoost",
                "bias_squared": (ada_result["bias_squared"]),
                "variance": (ada_result["variance"]),
                "total_error": (ada_result["total_error"]),
                "bias_plus_variance": (ada_result["bias_plus_variance"]),
                "difference": (ada_result["difference"]),
                "mean_accuracy": float(np.mean(ada_accuracies)),
                "ensemble_accuracy": float(
                    accuracy_calculation(
                        y_test,
                        ada_ensemble_prediction,
                    )
                ),
            },
            {
                "model": "Random Forest",
                "bias_squared": (rf_result["bias_squared"]),
                "variance": (rf_result["variance"]),
                "total_error": (rf_result["total_error"]),
                "bias_plus_variance": (rf_result["bias_plus_variance"]),
                "difference": (rf_result["difference"]),
                "mean_accuracy": float(np.mean(rf_accuracies)),
                "ensemble_accuracy": float(
                    accuracy_calculation(
                        y_test,
                        rf_ensemble_prediction,
                    )
                ),
            },
        ]
    )


def plot_bias_variance(
    results: pd.DataFrame,
) -> None:
    results.set_index("model")[
        [
            "bias_squared",
            "variance",
        ]
    ].plot(
        kind="bar",
        figsize=(8, 5),
    )

    plt.title("Bias² and Variance")
    plt.xlabel("Model")
    plt.ylabel("Error")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
