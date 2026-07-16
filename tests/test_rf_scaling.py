from __future__ import annotations
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from src.experiments.random_forest_experiment import (
    RandomForestScalingConfig,
    RandomForestScalingExperiment,
)


@pytest.fixture
def binary_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)

    X = rng.normal(size=(100, 4))
    y = np.array([0, 1] * 50)

    return X, y


@pytest.fixture
def multiclass_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)

    X = rng.normal(size=(90, 4))
    y = np.array([0, 1, 2] * 30)

    return X, y


@pytest.fixture
def config(tmp_path: Path) -> RandomForestScalingConfig:
    return RandomForestScalingConfig(
        n_estimators_values=[1, 5],
        max_depth_values=[1, 2],
        fixed_n_estimators=5,
        test_size=0.2,
        random_state=42,
        max_features="log2",
        n_jobs=1,
        figures_dir=tmp_path / "figures",
    )


@pytest.fixture
def experiment(
    config: RandomForestScalingConfig,
) -> RandomForestScalingExperiment:
    return RandomForestScalingExperiment(config)


def test_config_creates_figures_directory(
    config: RandomForestScalingConfig,
) -> None:
    assert not config.figures_dir.exists()

    RandomForestScalingExperiment(config)

    assert config.figures_dir.exists()
    assert config.figures_dir.is_dir()


def test_prepare_data_shapes(
    experiment: RandomForestScalingExperiment,
    binary_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = binary_data

    X_train, X_test, y_train, y_test = experiment._prepare_data(X, y)

    assert X_train.shape == (80, 4)
    assert X_test.shape == (20, 4)
    assert y_train.shape == (80,)
    assert y_test.shape == (20,)


def test_prepare_data_standardizes_training_data(
    experiment: RandomForestScalingExperiment,
    binary_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = binary_data

    X_train, _, _, _ = experiment._prepare_data(X, y)

    assert np.allclose(X_train.mean(axis=0), 0.0, atol=1e-7)
    assert np.allclose(X_train.std(axis=0), 1.0, atol=1e-7)


def test_prepare_data_preserves_class_distribution(
    experiment: RandomForestScalingExperiment,
    binary_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = binary_data

    _, _, y_train, y_test = experiment._prepare_data(X, y)

    assert np.mean(y_train == 1) == pytest.approx(0.5)
    assert np.mean(y_test == 1) == pytest.approx(0.5)


def test_f1_mean_returns_binary_for_two_classes() -> None:
    y = np.array([0, 1, 0, 1])

    result = RandomForestScalingExperiment._f1_mean(y)

    assert result == "binary"


def test_f1_mean_returns_macro_for_multiclass() -> None:
    y = np.array([0, 1, 2, 0, 1, 2])

    result = RandomForestScalingExperiment._f1_mean(y)

    assert result == "macro"


def test_evaluate_model_returns_expected_metrics(
    experiment: RandomForestScalingExperiment,
) -> None:
    X_train = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    )
    X_test = np.array(
        [
            [0.2, 0.1],
            [0.8, 0.9],
        ]
    )
    y_train = np.array([0, 1, 0, 1])
    y_test = np.array([0, 1])

    model = MagicMock()
    model.oob_score_ = 0.75
    model.predict.side_effect = [
        np.array([0, 1, 0, 1]),
        np.array([0, 1]),
    ]

    metrics = experiment._evaluate_model(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        f1_mean="binary",
    )

    assert model.fit.call_count == 1
    assert model.predict.call_count == 2

    assert metrics["train_acc"] == pytest.approx(1.0)
    assert metrics["test_acc"] == pytest.approx(1.0)
    assert metrics["oob_acc"] == pytest.approx(0.75)
    assert metrics["train_f1"] == pytest.approx(1.0)
    assert metrics["test_f1"] == pytest.approx(1.0)
    assert metrics["fit_time"] >= 0.0
    assert metrics["prediction_time"] >= 0.0


def test_evaluate_model_uses_nan_when_oob_is_missing(
    experiment: RandomForestScalingExperiment,
) -> None:
    model = MagicMock()
    model.predict.side_effect = [
        np.array([0, 1, 0, 1]),
        np.array([0, 1]),
    ]

    metrics = experiment._evaluate_model(
        model=model,
        X_train=np.zeros((4, 2)),
        X_test=np.zeros((2, 2)),
        y_train=np.array([0, 1, 0, 1]),
        y_test=np.array([0, 1]),
        f1_mean="binary",
    )

    assert np.isnan(metrics["oob_acc"])


@patch("src.experiments.random_forest_experiment.RandomForestClassifier")
def test_n_estimators_sweep_returns_correct_structure(
    mock_rf_class: MagicMock,
    experiment: RandomForestScalingExperiment,
    binary_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = binary_data

    model = MagicMock()
    mock_rf_class.return_value = model

    expected_metrics = {
        "train_acc": 0.95,
        "test_acc": 0.90,
        "oob_acc": 0.88,
        "train_f1": 0.94,
        "test_f1": 0.89,
        "fit_time": 0.1,
        "prediction_time": 0.01,
    }

    with patch.object(
        experiment,
        "_evaluate_model",
        return_value=expected_metrics,
    ):
        result = experiment.run_n_estimators_sweep(
            X,
            y,
            "Test Dataset",
        )

    assert result["dataset"] == "Test Dataset"
    assert result["n_estimators"] == [1, 5]

    assert result["train_acc"] == [0.95, 0.95]
    assert result["test_acc"] == [0.90, 0.90]
    assert result["oob_acc"] == [0.88, 0.88]
    assert result["train_f1"] == [0.94, 0.94]
    assert result["test_f1"] == [0.89, 0.89]

    assert mock_rf_class.call_count == 2

    assert mock_rf_class.call_args_list[0].kwargs["n_estimators"] == 1
    assert mock_rf_class.call_args_list[1].kwargs["n_estimators"] == 5


@patch("src.experiments.random_forest_experiment.RandomForestClassifier")
def test_max_depth_sweep_returns_correct_structure(
    mock_rf_class: MagicMock,
    experiment: RandomForestScalingExperiment,
    binary_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = binary_data

    model = MagicMock()
    mock_rf_class.return_value = model

    expected_metrics = {
        "train_acc": 0.96,
        "test_acc": 0.91,
        "oob_acc": 0.89,
        "train_f1": 0.95,
        "test_f1": 0.90,
        "fit_time": 0.1,
        "prediction_time": 0.01,
    }

    with patch.object(
        experiment,
        "_evaluate_model",
        return_value=expected_metrics,
    ):
        result = experiment.run_max_depth_sweep(
            X,
            y,
            "Test Dataset",
        )

    assert result["dataset"] == "Test Dataset"
    assert result["max_depth"] == [1, 2]

    assert result["train_acc"] == [0.96, 0.96]
    assert result["test_acc"] == [0.91, 0.91]
    assert result["oob_acc"] == [0.89, 0.89]

    assert mock_rf_class.call_count == 2

    assert mock_rf_class.call_args_list[0].kwargs["max_depth"] == 1
    assert mock_rf_class.call_args_list[1].kwargs["max_depth"] == 2

    assert (
        mock_rf_class.call_args_list[0].kwargs["n_estimators"]
        == experiment.config.fixed_n_estimators
    )


def test_plot_dataset_results_creates_figure(
    experiment: RandomForestScalingExperiment,
) -> None:
    result_n = {
        "dataset": "Test Dataset",
        "n_estimators": [1, 5],
        "train_acc": [0.90, 0.95],
        "test_acc": [0.85, 0.90],
        "oob_acc": [0.82, 0.88],
        "train_f1": [0.89, 0.94],
        "test_f1": [0.84, 0.89],
    }

    result_depth = {
        "dataset": "Test Dataset",
        "max_depth": [1, 2],
        "train_acc": [0.88, 0.96],
        "test_acc": [0.84, 0.91],
        "oob_acc": [0.81, 0.89],
        "train_f1": [0.87, 0.95],
        "test_f1": [0.83, 0.90],
    }

    experiment._plot_dataset_results(result_n, result_depth)

    expected_path = (
        experiment.config.figures_dir / "random_forest_scaling_test_dataset.png"
    )

    assert expected_path.exists()
    assert expected_path.is_file()
    assert expected_path.stat().st_size > 0


def test_plot_overfitting_creates_figure(
    experiment: RandomForestScalingExperiment,
) -> None:
    results_depth = {
        "Dataset One": {
            "max_depth": [1, 2, 3],
            "train_acc": [0.80, 0.90, 0.98],
            "test_acc": [0.78, 0.86, 0.89],
        },
        "Dataset Two": {
            "max_depth": [1, 2, 3],
            "train_acc": [0.82, 0.91, 0.97],
            "test_acc": [0.80, 0.88, 0.90],
        },
    }

    experiment._plot_overfitting(results_depth)

    expected_path = (
        experiment.config.figures_dir / "random_forest_scaling_overfitting.png"
    )

    assert expected_path.exists()
    assert expected_path.is_file()
    assert expected_path.stat().st_size > 0


def test_plot_overfitting_does_nothing_for_empty_results(
    experiment: RandomForestScalingExperiment,
) -> None:
    experiment._plot_overfitting({})

    expected_path = (
        experiment.config.figures_dir / "random_forest_scaling_overfitting.png"
    )

    assert not expected_path.exists()


def test_same_experiment_overwrites_same_figure(
    experiment: RandomForestScalingExperiment,
) -> None:
    result_n = {
        "dataset": "Test Dataset",
        "n_estimators": [1, 5],
        "train_acc": [0.90, 0.95],
        "test_acc": [0.85, 0.90],
        "oob_acc": [0.82, 0.88],
        "train_f1": [0.89, 0.94],
        "test_f1": [0.84, 0.89],
    }

    result_depth = {
        "dataset": "Test Dataset",
        "max_depth": [1, 2],
        "train_acc": [0.88, 0.96],
        "test_acc": [0.84, 0.91],
        "oob_acc": [0.81, 0.89],
        "train_f1": [0.87, 0.95],
        "test_f1": [0.83, 0.90],
    }

    expected_path = (
        experiment.config.figures_dir / "random_forest_scaling_test_dataset.png"
    )

    experiment._plot_dataset_results(result_n, result_depth)

    assert expected_path.exists()

    first_modified_time = expected_path.stat().st_mtime_ns

    result_n["test_acc"] = [0.50, 0.60]

    experiment._plot_dataset_results(result_n, result_depth)

    second_modified_time = expected_path.stat().st_mtime_ns

    assert expected_path.exists()
    assert second_modified_time >= first_modified_time


def test_other_figures_are_not_deleted(
    experiment: RandomForestScalingExperiment,
) -> None:
    other_figure = experiment.config.figures_dir / "another_experiment.png"
    other_figure.write_bytes(b"existing figure")

    result_n = {
        "dataset": "Test Dataset",
        "n_estimators": [1],
        "train_acc": [0.90],
        "test_acc": [0.85],
        "oob_acc": [0.82],
        "train_f1": [0.89],
        "test_f1": [0.84],
    }

    result_depth = {
        "dataset": "Test Dataset",
        "max_depth": [1],
        "train_acc": [0.88],
        "test_acc": [0.84],
        "oob_acc": [0.81],
        "train_f1": [0.87],
        "test_f1": [0.83],
    }

    experiment._plot_dataset_results(result_n, result_depth)

    assert other_figure.exists()
    assert other_figure.read_bytes() == b"existing figure"


def test_print_dataset_summary_selects_best_values() -> None:
    result_n = {
        "n_estimators": [1, 5, 10],
        "train_acc": [0.85, 0.95, 0.94],
        "test_acc": [0.80, 0.92, 0.90],
        "oob_acc": [0.78, 0.89, 0.88],
    }

    result_depth = {
        "max_depth": [1, 2, 3],
        "train_acc": [0.82, 0.91, 0.98],
        "test_acc": [0.79, 0.89, 0.87],
        "oob_acc": [0.77, 0.86, 0.85],
    }

    summary = RandomForestScalingExperiment._print_dataset_summary(
        "Test Dataset",
        result_n,
        result_depth,
    )

    assert summary["Dataset"] == "Test Dataset"
    assert summary["Best n_estimators"] == 5
    assert summary["Best n_estimators accuracy"] == pytest.approx(0.92)
    assert summary["Best n_estimators OOB"] == pytest.approx(0.89)

    assert summary["Best max_depth"] == 2
    assert summary["Best max_depth accuracy"] == pytest.approx(0.89)
    assert summary["Best max_depth OOB"] == pytest.approx(0.86)


def test_run_returns_all_results_and_summary(
    experiment: RandomForestScalingExperiment,
    binary_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = binary_data

    fake_n_result = {
        "dataset": "Binary",
        "n_estimators": [1, 5],
        "train_acc": [0.90, 0.95],
        "test_acc": [0.85, 0.92],
        "oob_acc": [0.82, 0.89],
        "train_f1": [0.89, 0.94],
        "test_f1": [0.84, 0.91],
        "fit_time": [0.1, 0.2],
        "prediction_time": [0.01, 0.02],
    }

    fake_depth_result = {
        "dataset": "Binary",
        "max_depth": [1, 2],
        "train_acc": [0.88, 0.96],
        "test_acc": [0.84, 0.91],
        "oob_acc": [0.81, 0.88],
        "train_f1": [0.87, 0.95],
        "test_f1": [0.83, 0.90],
        "fit_time": [0.1, 0.2],
        "prediction_time": [0.01, 0.02],
    }

    with (
        patch.object(
            experiment,
            "run_n_estimators_sweep",
            return_value=fake_n_result,
        ),
        patch.object(
            experiment,
            "run_max_depth_sweep",
            return_value=fake_depth_result,
        ),
        patch.object(experiment, "_plot_dataset_results") as plot_dataset,
        patch.object(experiment, "_plot_overfitting") as plot_overfitting,
    ):
        results_n, results_depth, summary_df = experiment.run({"Binary": (X, y)})

    assert "Binary" in results_n
    assert "Binary" in results_depth

    assert results_n["Binary"] == fake_n_result
    assert results_depth["Binary"] == fake_depth_result

    assert isinstance(summary_df, pd.DataFrame)
    assert len(summary_df) == 1

    assert summary_df.iloc[0]["Dataset"] == "Binary"
    assert summary_df.iloc[0]["Best n_estimators"] == 5
    assert summary_df.iloc[0]["Best max_depth"] == 2

    plot_dataset.assert_called_once_with(
        fake_n_result,
        fake_depth_result,
    )
    plot_overfitting.assert_called_once()
