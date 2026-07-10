from pathlib import Path
import shutil
import numpy as np
import matplotlib.pyplot as plt
import src.experiments.run_all

def make_dataset():
    rng = np.random.default_rng(42)
    X1 = rng.normal(loc=0, scale=0.3, size=(15, 4))
    X2 = rng.normal(loc=5, scale=0.3, size=(15, 4))
    X = np.vstack([X1, X2])
    y = np.array([0] * 15 + [1] * 15)
    return X, y

def fake_load_datasets():
    X, y = make_dataset()
    return {
        "Test_Dataset": (X, y, 3, 100)
    }

def clean_figures_folder():
    figures = Path("figures")
    if figures.exists():
        shutil.rmtree(figures)
    figures.mkdir(exist_ok=True)

def test_clean_name():
    assert src.experiments.run_all.clean_name("Adult Income") == "Adult_Income"
    assert src.experiments.run_all.clean_name("MNIST / 784") == "MNIST_784"
    assert src.experiments.run_all.clean_name("A-B.C_1") == "A-B.C_1"


def test_save_plots_creates_png():
    clean_figures_folder()

    src.experiments.run_all.current_plot_name = "test_plot"
    src.experiments.run_all.plot_number = 1

    plt.figure()
    plt.plot([1, 2, 3], [1, 4, 9])

    src.experiments.run_all.save_plots()

    path = Path("figures/test_plot_1.png")

    assert path.exists()
    assert path.is_file()
    assert path.stat().st_size > 0
    assert len(plt.get_fignums()) == 0


def test_run_step_returns_result_and_saves_plot():
    clean_figures_folder()

    def dummy_task():
        plt.figure()
        plt.plot([1, 2, 3], [3, 2, 1])
        return {"status": "ok"}

    result = src.experiments.run_all.run_step("Dummy Plot", dummy_task)

    path = Path("figures/Dummy_Plot_1.png")

    assert result == {"status": "ok"}
    assert path.exists()
    assert path.stat().st_size > 0


def test_load_datasets_mocked():
    original_load_datasets = src.experiments.run_all.load_datasets

    try:
        src.experiments.run_all.load_datasets = fake_load_datasets

        datasets = src.experiments.run_all.load_datasets()

        assert isinstance(datasets, dict)
        assert "Test_Dataset" in datasets

        X, y, min_samples, max_points = datasets["Test_Dataset"]

        assert X.shape == (30, 4)
        assert y.shape == (30,)
        assert min_samples == 3
        assert max_points == 100

    finally:
        src.experiments.run_all.load_datasets = original_load_datasets


def test_run_all_creates_required_visualisations():
    clean_figures_folder()

    original_load_datasets = src.experiments.run_all.load_datasets

    try:
        src.experiments.run_all.load_datasets = fake_load_datasets
        src.experiments.run_all.run_all()

        figures = list(Path("figures").glob("*.png"))
        figure_names = [p.name for p in figures]

        assert len(figures) >= 5

        assert any("Test_Dataset_kmeans" in name for name in figure_names)
        assert any("Test_Dataset_dbscan_eps_search" in name for name in figure_names)
        assert any("Test_Dataset_pca" in name for name in figure_names)
        assert any("Test_Dataset_dbscan" in name for name in figure_names)
        assert any("Test_Dataset_dbscan_best_eps" in name for name in figure_names)

        for path in figures:
            assert path.stat().st_size > 0

    finally:
        src.experiments.run_all.load_datasets = original_load_datasets


def run_all_tests():
    test_clean_name()
    test_save_plots_creates_png()
    test_run_step_returns_result_and_saves_plot()
    test_load_datasets_mocked()
    test_run_all_creates_required_visualisations()

    print("All run_all tests passed.")


if __name__ == "__main__":
    run_all_tests()
