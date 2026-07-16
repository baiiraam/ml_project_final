"""
Tests for adaboost_scale_experiment_utils.py
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from sklearn.datasets import make_classification

# Import the functions to test
import sys

sys.path.append("..")
from src.experiments.adaboost_scale_experiment_utils import (
    get_memory_usage,
    run_adaboost_scaling_staged,
    compare_with_sklearn,
    save_dataset_results,
    save_staged_predictions,
)
from src.trees.boosting.adaboost import AdaBoostClassifier


# ============================================
# Fixtures
# ============================================


@pytest.fixture
def sample_data():
    """Create a small synthetic dataset for testing."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )
    return X, y


@pytest.fixture
def mock_results():
    """Create mock results dictionary."""
    return {
        "dataset": "test_dataset",
        "n_estimators": [1, 5, 10, 15, 20],
        "train_acc": [0.85, 0.88, 0.90, 0.91, 0.92],
        "test_acc": [0.80, 0.82, 0.83, 0.84, 0.84],
        "train_f1": [0.84, 0.87, 0.89, 0.90, 0.91],
        "test_f1": [0.79, 0.81, 0.82, 0.83, 0.83],
    }


# ============================================
# Tests for get_memory_usage
# ============================================


def test_get_memory_usage_numpy():
    """Test memory usage calculation for numpy arrays."""
    arr = np.random.randn(1000, 100)
    mem = get_memory_usage(arr)
    assert mem > 0
    assert isinstance(mem, float)


def test_get_memory_usage_pandas():
    """Test memory usage calculation for pandas objects."""
    df = pd.DataFrame(np.random.randn(1000, 10))
    mem = get_memory_usage(df)
    assert mem > 0
    assert isinstance(mem, float)

    series = pd.Series(np.random.randn(1000))
    mem = get_memory_usage(series)
    assert mem > 0
    assert isinstance(mem, float)


def test_get_memory_usage_other():
    """Test memory usage for unsupported objects."""
    mem = get_memory_usage([1, 2, 3])
    assert mem == 0


def test_get_memory_usage_empty():
    """Test memory usage with empty objects."""
    arr = np.array([])
    mem = get_memory_usage(arr)
    assert mem == 0.0

    df = pd.DataFrame()
    mem = get_memory_usage(df)
    # Empty DataFrame still has some memory overhead
    assert mem >= 0.0
    assert isinstance(mem, float)


# ============================================
# Tests for run_adaboost_scaling_staged
# ============================================


def test_run_adaboost_scaling_staged(sample_data):
    """Test that the staged experiment runs and returns expected structure."""
    X, y = sample_data
    results, model = run_adaboost_scaling_staged(
        X, y, dataset_name="test", max_estimators=10, step=2
    )

    # Check results structure
    assert "dataset" in results
    assert results["dataset"] == "test"
    assert "n_estimators" in results
    assert len(results["n_estimators"]) > 0
    assert len(results["train_acc"]) == len(results["n_estimators"])
    assert len(results["test_acc"]) == len(results["n_estimators"])
    assert len(results["train_f1"]) == len(results["n_estimators"])
    assert len(results["test_f1"]) == len(results["n_estimators"])

    # Check model
    assert isinstance(model, AdaBoostClassifier)
    assert model.n_estimators == 10


def test_run_adaboost_scaling_staged_with_defaults(sample_data):
    """Test that the function works with default parameters."""
    X, y = sample_data
    results, model = run_adaboost_scaling_staged(X, y, "test")

    # Should use default values (MAX_ESTIMATORS=20, STEP=5)
    assert len(results["n_estimators"]) > 0
    assert isinstance(model, AdaBoostClassifier)


def test_run_adaboost_scaling_staged_custom_params(sample_data):
    """Test that custom parameters override defaults."""
    X, y = sample_data
    max_estimators = 8
    step = 2

    results, model = run_adaboost_scaling_staged(
        X, y, dataset_name="test", max_estimators=max_estimators, step=step
    )

    # Check that custom parameters were used
    expected_list = [1] + list(range(step, max_estimators + 1, step))
    assert results["n_estimators"] == expected_list
    assert model.n_estimators == max_estimators


def test_run_adaboost_scaling_staged_single_estimator(sample_data):
    """Test with a single estimator."""
    X, y = sample_data
    results, model = run_adaboost_scaling_staged(
        X, y, dataset_name="test", max_estimators=1, step=1
    )

    # With step=1, max_estimators=1: [1] + range(1,2,1) = [1, 1]
    # So we get duplicate 1s
    assert len(results["n_estimators"]) == 2
    assert results["n_estimators"][0] == 1
    assert results["n_estimators"][1] == 1


# ============================================
# Tests for save_dataset_results (with mocking)
# ============================================


@patch("src.experiments.adaboost_scale_experiment_utils.pd.DataFrame")
@patch("src.experiments.adaboost_scale_experiment_utils.os.makedirs")
def test_save_dataset_results(mock_makedirs, mock_dataframe, mock_results):
    """Test that save_dataset_results creates DataFrame and saves files."""
    # Setup mocks
    mock_df = Mock()
    mock_dataframe.return_value = mock_df

    # Call the function
    save_dataset_results(
        "test_dataset",
        mock_results,
        sample_size=100,
        X=np.array([[1, 2]]),
        y=np.array([0, 1]),
    )

    # Verify makedirs was called
    mock_makedirs.assert_called_once_with("../notebooks", exist_ok=True)

    # Verify DataFrame was created
    mock_dataframe.assert_called()

    # Verify to_csv was called twice (details and metadata)
    assert mock_df.to_csv.call_count == 2


@patch("src.experiments.adaboost_scale_experiment_utils.pd.DataFrame")
@patch("src.experiments.adaboost_scale_experiment_utils.os.makedirs")
def test_save_dataset_results_without_metadata(
    mock_makedirs, mock_dataframe, mock_results
):
    """Test save_dataset_results works without X and y."""
    mock_df = Mock()
    mock_dataframe.return_value = mock_df

    save_dataset_results("test_dataset", mock_results)

    mock_makedirs.assert_called_once_with("../notebooks", exist_ok=True)
    mock_dataframe.assert_called()
    assert mock_df.to_csv.call_count == 2


# ============================================
# Tests for save_staged_predictions (with mocking)
# ============================================


@patch("src.experiments.adaboost_scale_experiment_utils.StandardScaler")
@patch("src.experiments.adaboost_scale_experiment_utils.train_test_split")
@patch("src.experiments.adaboost_scale_experiment_utils.AdaBoostClassifier")
@patch("src.experiments.adaboost_scale_experiment_utils.pd.DataFrame")
@patch("src.experiments.adaboost_scale_experiment_utils.os.makedirs")
def test_save_staged_predictions(
    mock_makedirs,
    mock_dataframe,
    mock_adaboost,
    mock_train_test_split,
    mock_scaler,
    sample_data,
):
    """Test save_staged_predictions with mocked model."""
    X, y = sample_data

    # Mock train_test_split to return splits
    split_idx = len(y) // 2
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    mock_train_test_split.return_value = (X_train, X_test, y_train, y_test)

    # Mock StandardScaler
    mock_scaler_instance = Mock()
    mock_scaler_instance.fit_transform.return_value = X_train
    mock_scaler_instance.transform.return_value = X_test
    mock_scaler.return_value = mock_scaler_instance

    # Create a generator mock for staged_predict
    def mock_staged_predict(data):
        # Yield predictions for 5 rounds
        for i in range(5):
            yield np.array([0, 1] * (len(data) // 2) + [0] * (len(data) % 2))

    # Mock the model
    mock_model = Mock()
    mock_model.staged_predict.side_effect = mock_staged_predict
    mock_adaboost.return_value = mock_model

    # Mock DataFrame
    mock_df = Mock()
    mock_dataframe.return_value = mock_df

    # Call the function
    result_df = save_staged_predictions(X, y, "test_dataset", max_estimators=5)

    # Verify makedirs was called
    mock_makedirs.assert_called_once_with("../notebooks", exist_ok=True)

    # Verify DataFrame was created
    mock_dataframe.assert_called()

    # Verify to_csv was called
    mock_df.to_csv.assert_called_once()

    # Verify result is a DataFrame
    assert result_df.equals(mock_df)


# ============================================
# Tests for compare_with_sklearn (with mocking)
# ============================================


@patch("src.experiments.adaboost_scale_experiment_utils.SklearnAdaBoost")
@patch("src.experiments.adaboost_scale_experiment_utils.train_test_split")
def test_compare_with_sklearn(
    mock_train_test_split, mock_sklearn_adaboost, sample_data
):
    """Test comparison with sklearn's implementation using mocks."""
    X, y = sample_data

    # Split data properly for mock
    split_idx = len(y) // 2
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Mock train_test_split to return proper splits
    mock_train_test_split.return_value = (X_train, X_test, y_train, y_test)

    # Mock our model
    mock_our_model = Mock()
    mock_our_model.predict.return_value = y_test  # Predict same as test labels
    mock_our_model.n_estimators = 10

    # Mock sklearn model
    mock_sk_model = Mock()
    mock_sk_model.predict.return_value = y_test  # Predict same as test labels
    mock_sklearn_adaboost.return_value = mock_sk_model

    # Call the function
    our_acc, sk_acc = compare_with_sklearn(mock_our_model, X, y, "test_dataset")

    # With perfect predictions, accuracy should be 1.0
    assert our_acc == 1.0
    assert sk_acc == 1.0

    # Verify sklearn model was created with correct parameters
    mock_sklearn_adaboost.assert_called_once()
    call_kwargs = mock_sklearn_adaboost.call_args[1]
    assert call_kwargs["n_estimators"] == 10


# ============================================
# Integration tests (with mocking for file operations)
# ============================================


@patch("src.experiments.adaboost_scale_experiment_utils.save_dataset_results")
@patch("src.experiments.adaboost_scale_experiment_utils.save_staged_predictions")
def test_train_dataset_staged(mock_save_staged, mock_save_dataset, sample_data):
    """Test train_dataset_staged function."""
    from src.experiments.adaboost_scale_experiment_utils import train_dataset_staged

    X, y = sample_data

    # Call the function
    name, result, model = train_dataset_staged(
        "test", X, y, max_estimators=5, step=1, sample_size=100
    )

    # Verify results
    assert name == "test"
    assert "n_estimators" in result
    assert len(result["n_estimators"]) > 0
    assert isinstance(model, AdaBoostClassifier)

    # Verify save functions were called
    mock_save_dataset.assert_called_once()
    mock_save_staged.assert_called_once()


@patch("src.experiments.adaboost_scale_experiment_utils.train_dataset_staged")
def test_run_all_staged(mock_train_dataset, sample_data):
    """Test run_all_staged function."""
    from src.experiments.adaboost_scale_experiment_utils import run_all_staged

    X, y = sample_data
    datasets = {"test1": (X, y), "test2": (X, y)}

    # Mock the train_dataset_staged return for each call
    def mock_train_side_effect(name, X, y, max_estimators, step, sample_size):
        mock_result = {
            "dataset": name,
            "n_estimators": [1, 2, 3],
            "train_acc": [0.8, 0.9, 0.95],
            "test_acc": [0.7, 0.8, 0.85],
            "train_f1": [0.79, 0.89, 0.94],
            "test_f1": [0.69, 0.79, 0.84],
        }
        mock_model = Mock()
        return name, mock_result, mock_model

    mock_train_dataset.side_effect = mock_train_side_effect

    # Call the function
    results, models, total_time = run_all_staged(datasets, max_estimators=5, step=1)

    # Verify results
    assert len(results) == 2
    assert len(models) == 2
    assert total_time >= 0

    # Verify train_dataset_staged was called for each dataset
    assert mock_train_dataset.call_count == 2

    # Verify result keys match dataset names
    assert "test1" in results
    assert "test2" in results


# ============================================
# Run tests if script is executed directly
# ============================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
