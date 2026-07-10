import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from src.utils.unsupervised_helper import (
    _prepare_unsupervised_data,
    pca_task,
    kmeans_task,
    dbscan_task,
    find_best_dbscan_eps,
    plot_dbscan_best_eps,
)

plt.show = lambda *args, **kwargs: None


def make_dataset() -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    rng = np.random.default_rng(42)
    x1 = rng.normal(loc=0.0, scale=0.3, size=(20, 4))
    x2 = rng.normal(loc=5.0, scale=0.3, size=(20, 4))
    x = np.vstack([x1, x2]).astype(float)
    y = np.array([0] * 20 + [1] * 20, dtype=int)
    return x, y


def test_prepare_data() -> None:
    x = pd.DataFrame({
        "a": [1, 2, 3, np.nan],
        "b": ["4", "5", "bad", "7"],
        "all_nan": [np.nan, np.nan, np.nan, np.nan],
    })
    y = pd.DataFrame({"label": [0, 1, 0, 1]})

    x_clean, y_clean, x_scaled = _prepare_unsupervised_data(
        x,
        y,
        max_points=10,
        drop_all_nan_columns=True,
    )

    assert "all_nan" not in x_clean.columns
    assert x_clean.shape == (4, 2)
    assert y_clean.shape == (4,)
    assert x_scaled.shape == (4, 2)
    assert not np.isnan(x_scaled).any()
    assert np.allclose(x_scaled.mean(axis=0), 0.0, atol=1e-7)


def test_sampling_reproducible() -> None:
    x = np.arange(1000).reshape(100, 10)
    y = np.arange(100)

    x1, y1, s1 = _prepare_unsupervised_data(x, y, max_points=20)
    x2, y2, s2 = _prepare_unsupervised_data(x, y, max_points=20)

    assert x1.shape == (20, 10)
    assert y1.shape == (20,)
    assert s1.shape == (20, 10)
    assert x1.equals(x2)
    assert y1.equals(y2)
    assert np.allclose(s1, s2)


def test_kmeans_task() -> None:
    x, y = make_dataset()

    results = kmeans_task(
        dataset_name="Test",
        X=x,
        y=y,
        max_points=100,
        n_init=2,
    )

    assert isinstance(results, pd.DataFrame)
    assert list(results.columns) == ["k", "inertia", "ARI"]
    assert len(results) == 10
    assert results["k"].tolist() == list(range(1, 11))
    assert results["inertia"].notna().all()
    assert results["ARI"].between(-1, 1).all()


def test_dbscan_task() -> None:
    x, y = make_dataset()

    result = dbscan_task(
        dataset_name="Test",
        X=x,
        y=y,
        eps=1.5,
        min_samples=3,
        max_points=100,
    )

    assert isinstance(result, dict)
    assert result["dataset"] == "Test"
    assert result["eps"] == 1.5
    assert result["min_samples"] == 3
    assert -1 <= result["ARI"] <= 1
    assert 0 <= result["noise_fraction"] <= 1
    assert result["n_clusters"] >= 0


def test_find_best_dbscan_eps() -> None:
    x, y = make_dataset()

    results, best_row = find_best_dbscan_eps(
        dataset_name="Test",
        X=x,
        y=y,
        min_samples=3,
        max_points=100,
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0
    assert list(results.columns) == ["eps", "clusters", "ARI", "noise_fraction"]
    assert results["ARI"].between(-1, 1).all()
    assert results["noise_fraction"].between(0, 1).all()

    expected_best = results.loc[results["ARI"].idxmax()]
    assert best_row["eps"] == expected_best["eps"]
    assert best_row["ARI"] == expected_best["ARI"]


def test_pca_task_runs() -> None:
    x, y = make_dataset()

    pca_task(
        dataset_name="Test",
        X=x,
        y=y,
        n_clusters=2,
        dbscan_eps=1.5,
        dbscan_min_samples=3,
        max_points=100,
    )

    assert True


def test_plot_dbscan_best_eps_runs() -> None:
    x, y = make_dataset()

    plot_dbscan_best_eps(
        dataset_name="Test",
        X=x,
        y=y,
        best_eps=1.5,
        min_samples=3,
        max_points=100,
    )

    assert True


def run_all_tests() -> None:
    test_prepare_data()
    test_sampling_reproducible()
    test_kmeans_task()
    test_dbscan_task()
    test_find_best_dbscan_eps()
    test_pca_task_runs()
    test_plot_dbscan_best_eps_runs()
    print("All unsupervised helper tests passed.")


if __name__ == "__main__":
    run_all_tests()
