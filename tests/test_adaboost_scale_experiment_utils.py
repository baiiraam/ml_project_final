"""
Tests for adaboost_scale_experiment_utils.py
"""

import numpy as np
from sklearn.datasets import make_classification
from src.utils.adaboost_scale_experiment_utils import get_memory_usage


def test_get_memory_usage_numpy():
    """Test get_memory_usage with numpy array."""
    arr = np.zeros((1000, 10), dtype=np.float64)
    mem = get_memory_usage(arr)
    # 1000 * 10 * 8 bytes = 80,000 bytes ≈ 0.076 MB
    assert mem > 0.07
    assert mem < 0.08


def test_get_memory_usage_pandas():
    """Test get_memory_usage with pandas DataFrame."""
    import pandas as pd

    df = pd.DataFrame(np.zeros((1000, 10)))
    mem = get_memory_usage(df)
    assert mem > 0.07
    assert mem < 0.08


def test_run_all_staged_basic():
    """Test run_all_staged runs without errors on a small dataset."""
    from src.utils.adaboost_scale_experiment_utils import run_all_staged

    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    datasets = {"test": (X, y)}

    results, models, total_time = run_all_staged(
        datasets, max_estimators=10, step=5, random_state=42, test_size=0.2
    )

    assert "test" in results
    assert "test" in models
    assert total_time > 0
    assert len(results["test"]["n_estimators"]) == 3  # [1, 5, 10]
