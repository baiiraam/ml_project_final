from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.metrics.evaluation import accuracy_calculation, f1_score
from src.utils.rf_scale import RandomForestClassifier


@dataclass
class RandomForestScalingConfig:
    n_estimators_values: list[int] = field(
        default_factory=lambda: (
            [1, 3]
            + [i * 5 for i in range(1, 6)]
            + [30, 40, 50]
            + [i * 10 for i in range(6, 21)]
        )
    )
    max_depth_values: list[int] = field(default_factory=lambda: list(range(1, 21)))
    fixed_n_estimators: int = 100
    test_size: float = 0.2
    random_state: int = 42
    max_features: str | int | None = "log2"
    n_jobs: int = -2
    figures_dir: Path = Path("figures")


class RandomForestScalingExperiment:
    def __init__(
        self,
        config: RandomForestScalingConfig | None = None,
    ) -> None:
        self.config = config or RandomForestScalingConfig()
        self.config.figures_dir.mkdir(parents=True, exist_ok=True)

    def _prepare_data(
        self,
        X: Any,
        y: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_array = np.asarray(X)
        y_array = np.asarray(y).ravel()

        X_train, X_test, y_train, y_test = train_test_split(
            X_array,
            y_array,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y_array,
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    @staticmethod
    def _f1_mean(y: np.ndarray) -> str:
        return "binary" if len(np.unique(y)) == 2 else "macro"

    def _evaluate_model(
        self,
        model: RandomForestClassifier,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        f1_mean: str,
    ) -> dict[str, float]:
        fit_start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - fit_start

        prediction_start = time.perf_counter()
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)
        prediction_time = time.perf_counter() - prediction_start

        oob_score = getattr(model, "oob_score_", None)

        if not isinstance(oob_score, (float, int, np.floating, np.integer)):
            oob_score = np.nan

        return {
            "train_acc": accuracy_calculation(y_train, train_predictions),
            "test_acc": accuracy_calculation(y_test, test_predictions),
            "oob_acc": float(oob_score),
            "train_f1": f1_score(
                y_train,
                train_predictions,
                mean=f1_mean,
            ),
            "test_f1": f1_score(
                y_test,
                test_predictions,
                mean=f1_mean,
            ),
            "fit_time": fit_time,
            "prediction_time": prediction_time,
        }

    def run_n_estimators_sweep(
        self,
        X: Any,
        y: Any,
        dataset_name: str,
    ) -> dict[str, Any]:
        X_train, X_test, y_train, y_test = self._prepare_data(X, y)
        f1_mean = self._f1_mean(y_train)

        result: dict[str, Any] = {
            "dataset": dataset_name,
            "n_estimators": [],
            "train_acc": [],
            "test_acc": [],
            "oob_acc": [],
            "train_f1": [],
            "test_f1": [],
            "fit_time": [],
            "prediction_time": [],
        }

        print("\n" + "=" * 70)
        print(f"{dataset_name}: N_ESTIMATORS SWEEP")
        print("=" * 70)

        for n_estimators in self.config.n_estimators_values:
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=None,
                max_features=self.config.max_features,
                oob_score=True,
                n_jobs=self.config.n_jobs,
                random_state=self.config.random_state,
            )

            metrics = self._evaluate_model(
                model,
                X_train,
                X_test,
                y_train,
                y_test,
                f1_mean,
            )

            result["n_estimators"].append(n_estimators)

            for metric_name, metric_value in metrics.items():
                result[metric_name].append(metric_value)

            print(
                f"n_estimators={n_estimators:3d} | "
                f"train_acc={metrics['train_acc']:.4f} | "
                f"test_acc={metrics['test_acc']:.4f} | "
                f"oob_acc={metrics['oob_acc']:.4f} | "
                f"test_f1={metrics['test_f1']:.4f} | "
                f"fit={metrics['fit_time']:.3f}s | "
                f"predict={metrics['prediction_time']:.3f}s"
            )

        return result

    def run_max_depth_sweep(
        self,
        X: Any,
        y: Any,
        dataset_name: str,
    ) -> dict[str, Any]:
        X_train, X_test, y_train, y_test = self._prepare_data(X, y)
        f1_mean = self._f1_mean(y_train)

        result: dict[str, Any] = {
            "dataset": dataset_name,
            "max_depth": [],
            "train_acc": [],
            "test_acc": [],
            "oob_acc": [],
            "train_f1": [],
            "test_f1": [],
            "fit_time": [],
            "prediction_time": [],
        }

        print("\n" + "=" * 70)
        print(f"{dataset_name}: MAX_DEPTH SWEEP")
        print("=" * 70)

        for max_depth in self.config.max_depth_values:
            model = RandomForestClassifier(
                n_estimators=self.config.fixed_n_estimators,
                max_depth=max_depth,
                max_features=self.config.max_features,
                oob_score=True,
                n_jobs=self.config.n_jobs,
                random_state=self.config.random_state,
            )

            metrics = self._evaluate_model(
                model,
                X_train,
                X_test,
                y_train,
                y_test,
                f1_mean,
            )

            result["max_depth"].append(max_depth)

            for metric_name, metric_value in metrics.items():
                result[metric_name].append(metric_value)

            print(
                f"max_depth={max_depth:2d} | "
                f"train_acc={metrics['train_acc']:.4f} | "
                f"test_acc={metrics['test_acc']:.4f} | "
                f"oob_acc={metrics['oob_acc']:.4f} | "
                f"test_f1={metrics['test_f1']:.4f} | "
                f"fit={metrics['fit_time']:.3f}s | "
                f"predict={metrics['prediction_time']:.3f}s"
            )

        return result

    def _plot_dataset_results(
        self,
        result_n: dict[str, Any],
        result_depth: dict[str, Any],
    ) -> None:
        dataset_name = result_n["dataset"]

        figure, axes = plt.subplots(2, 2, figsize=(15, 10))

        axes[0, 0].plot(
            result_n["n_estimators"],
            result_n["train_acc"],
            marker="o",
            markersize=3,
            label="Training",
        )
        axes[0, 0].plot(
            result_n["n_estimators"],
            result_n["test_acc"],
            marker="s",
            markersize=3,
            label="Test",
        )
        axes[0, 0].plot(
            result_n["n_estimators"],
            result_n["oob_acc"],
            marker="^",
            markersize=3,
            linestyle="--",
            label="OOB",
        )
        axes[0, 0].set_title("Accuracy vs n_estimators")
        axes[0, 0].set_xlabel("n_estimators")
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        axes[0, 1].plot(
            result_n["n_estimators"],
            result_n["train_f1"],
            marker="o",
            markersize=3,
            label="Training",
        )
        axes[0, 1].plot(
            result_n["n_estimators"],
            result_n["test_f1"],
            marker="s",
            markersize=3,
            label="Test",
        )
        axes[0, 1].set_title("F1 score vs n_estimators")
        axes[0, 1].set_xlabel("n_estimators")
        axes[0, 1].set_ylabel("F1 score")
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        axes[1, 0].plot(
            result_depth["max_depth"],
            result_depth["train_acc"],
            marker="o",
            markersize=3,
            label="Training",
        )
        axes[1, 0].plot(
            result_depth["max_depth"],
            result_depth["test_acc"],
            marker="s",
            markersize=3,
            label="Test",
        )
        axes[1, 0].plot(
            result_depth["max_depth"],
            result_depth["oob_acc"],
            marker="^",
            markersize=3,
            linestyle="--",
            label="OOB",
        )
        axes[1, 0].set_title("Accuracy vs max_depth")
        axes[1, 0].set_xlabel("max_depth")
        axes[1, 0].set_ylabel("Accuracy")
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        axes[1, 1].plot(
            result_depth["max_depth"],
            result_depth["train_f1"],
            marker="o",
            markersize=3,
            label="Training",
        )
        axes[1, 1].plot(
            result_depth["max_depth"],
            result_depth["test_f1"],
            marker="s",
            markersize=3,
            label="Test",
        )
        axes[1, 1].set_title("F1 score vs max_depth")
        axes[1, 1].set_xlabel("max_depth")
        axes[1, 1].set_ylabel("F1 score")
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

        figure.suptitle(
            f"Random Forest Scaling — {dataset_name}",
            fontsize=16,
        )
        figure.tight_layout()

        filename = f"random_forest_scaling_{dataset_name.lower().replace(' ', '_')}.png"
        save_path = self.config.figures_dir / filename

        figure.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(figure)

        print(f"Figure saved: {save_path}")

    def _plot_overfitting(
        self,
        results_depth: dict[str, dict[str, Any]],
    ) -> None:
        dataset_count = len(results_depth)

        if dataset_count == 0:
            return

        columns = 2
        rows = int(np.ceil(dataset_count / columns))

        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(14, 5 * rows),
            squeeze=False,
        )

        flat_axes = axes.flatten()

        for index, (dataset_name, result) in enumerate(results_depth.items()):
            train_accuracy = np.asarray(result["train_acc"])
            test_accuracy = np.asarray(result["test_acc"])
            gap = train_accuracy - test_accuracy

            axis = flat_axes[index]

            axis.plot(
                result["max_depth"],
                gap,
                marker="o",
                markersize=4,
            )
            axis.axhline(0, linestyle="-", alpha=0.5)
            axis.axhline(
                0.02,
                linestyle="--",
                alpha=0.6,
                label="2% threshold",
            )
            axis.axhline(
                0.05,
                linestyle="--",
                alpha=0.6,
                label="5% threshold",
            )
            axis.fill_between(
                result["max_depth"],
                0,
                gap,
                alpha=0.2,
            )

            axis.set_title(f"{dataset_name} — overfitting gap")
            axis.set_xlabel("max_depth")
            axis.set_ylabel("Train accuracy - test accuracy")
            axis.legend()
            axis.grid(alpha=0.3)

        for index in range(dataset_count, len(flat_axes)):
            figure.delaxes(flat_axes[index])

        figure.tight_layout()

        save_path = self.config.figures_dir / "random_forest_scaling_overfitting.png"

        figure.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(figure)

        print(f"Figure saved: {save_path}")

    @staticmethod
    def _print_dataset_summary(
        dataset_name: str,
        result_n: dict[str, Any],
        result_depth: dict[str, Any],
    ) -> dict[str, Any]:
        best_n_index = int(np.argmax(result_n["test_acc"]))
        best_depth_index = int(np.argmax(result_depth["test_acc"]))

        summary = {
            "Dataset": dataset_name,
            "Best n_estimators": result_n["n_estimators"][best_n_index],
            "Best n_estimators accuracy": result_n["test_acc"][best_n_index],
            "Best n_estimators OOB": result_n["oob_acc"][best_n_index],
            "Best max_depth": result_depth["max_depth"][best_depth_index],
            "Best max_depth accuracy": result_depth["test_acc"][best_depth_index],
            "Best max_depth OOB": result_depth["oob_acc"][best_depth_index],
        }

        n_gap = np.asarray(result_n["train_acc"]) - np.asarray(result_n["test_acc"])
        depth_gap = np.asarray(result_depth["train_acc"]) - np.asarray(
            result_depth["test_acc"]
        )

        print("\n" + "-" * 70)
        print(f"{dataset_name} SUMMARY")
        print("-" * 70)
        print(f"Best n_estimators: {summary['Best n_estimators']}")
        print(f"Best test accuracy: {summary['Best n_estimators accuracy']:.4f}")
        print(f"OOB accuracy: {summary['Best n_estimators OOB']:.4f}")
        print(f"Best max_depth: {summary['Best max_depth']}")
        print(f"Best depth test accuracy: {summary['Best max_depth accuracy']:.4f}")
        print(f"Maximum n_estimators overfitting gap: {np.max(n_gap):.4f}")
        print(f"Maximum max_depth overfitting gap: {np.max(depth_gap):.4f}")

        return summary

    def run(
        self,
        datasets: dict[str, tuple[Any, Any]],
    ) -> tuple[
        dict[str, dict[str, Any]],
        dict[str, dict[str, Any]],
        pd.DataFrame,
    ]:
        experiment_start = time.perf_counter()

        results_n: dict[str, dict[str, Any]] = {}
        results_depth: dict[str, dict[str, Any]] = {}
        summaries: list[dict[str, Any]] = []

        print("\n" + "=" * 70)
        print("RANDOM FOREST SCALING EXPERIMENT")
        print("=" * 70)
        print(f"n_estimators values: {self.config.n_estimators_values}")
        print(f"max_depth values: {self.config.max_depth_values}")
        print(f"Fixed trees for depth sweep: {self.config.fixed_n_estimators}")
        print(f"Figures directory: {self.config.figures_dir}")

        for dataset_name, (X, y) in datasets.items():
            dataset_start = time.perf_counter()

            print("\n" + "#" * 70)
            print(f"DATASET: {dataset_name}")
            print(f"Samples: {len(X):,}")
            print(f"Features: {np.asarray(X).shape[1]}")
            print(f"Classes: {len(np.unique(np.asarray(y).ravel()))}")
            print("#" * 70)

            result_n = self.run_n_estimators_sweep(
                X,
                y,
                dataset_name,
            )
            result_depth = self.run_max_depth_sweep(
                X,
                y,
                dataset_name,
            )

            results_n[dataset_name] = result_n
            results_depth[dataset_name] = result_depth

            self._plot_dataset_results(
                result_n,
                result_depth,
            )

            summaries.append(
                self._print_dataset_summary(
                    dataset_name,
                    result_n,
                    result_depth,
                )
            )

            dataset_time = time.perf_counter() - dataset_start
            print(f"\n{dataset_name} completed in {dataset_time:.2f} seconds.")

        self._plot_overfitting(results_depth)

        summary_df = pd.DataFrame(summaries)

        print("\n" + "=" * 70)
        print("FINAL RANDOM FOREST SCALING SUMMARY")
        print("=" * 70)

        if not summary_df.empty:
            print(summary_df.to_string(index=False))

            best_index = summary_df["Best n_estimators accuracy"].idxmax()
            best_result = summary_df.loc[best_index]

            print("\nBest overall result:")
            print(f"Dataset: {best_result['Dataset']}")
            print(f"Accuracy: {best_result['Best n_estimators accuracy']:.4f}")
            print(f"n_estimators: {best_result['Best n_estimators']}")

        total_time = time.perf_counter() - experiment_start

        print("\n" + "=" * 70)
        print(f"EXPERIMENT COMPLETED IN {total_time:.2f} SECONDS")
        print("=" * 70)

        return results_n, results_depth, summary_df
