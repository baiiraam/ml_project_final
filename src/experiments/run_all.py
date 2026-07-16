# ruff: noqa: E402
import re
import sys
from pathlib import Path
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, fetch_openml
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.unsupervised_helper import (
    pca_task,
    kmeans_task,
    dbscan_task,
    find_best_dbscan_eps,
    plot_dbscan_best_eps,
)

from src.experiments.random_forest_experiment import (
    RandomForestScalingConfig,
    RandomForestScalingExperiment,
)
from src.utils.preprocessing import (
    load_breast_cancer_data,
    load_adult_income_data,
    load_covertype_data,
    load_mnist_data,
)

matplotlib.use("Agg")


def load_datasets_experiment():
    X_bc, y_bc = load_breast_cancer_data()
    X_adult, y_adult = load_adult_income_data()
    X_cover, y_cover = load_covertype_data()
    X_mnist, y_mnist = load_mnist_data()

    return {
        "Breast Cancer": (
            np.asarray(X_bc),
            np.asarray(y_bc).ravel(),
        ),
        "Adult Income": (
            np.asarray(X_adult),
            np.asarray(y_adult).ravel(),
        ),
        "Covertype": (
            np.asarray(X_cover),
            np.asarray(y_cover).ravel(),
        ),
        "MNIST": (
            np.asarray(X_mnist),
            np.asarray(y_mnist).ravel(),
        ),
    }


BASE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = BASE_DIR / "figures"


def main():
    datasets = load_datasets_experiment()

    config = RandomForestScalingConfig(
        figures_dir=FIGURES_DIR,
        random_state=42,
        n_jobs=-1,
    )

    experiment = RandomForestScalingExperiment(config)
    experiment.run(datasets)


FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)
current_plot_name = "plot"
plot_number = 1


def clean_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


def save_plots(*args, **kwargs):
    global plot_number
    for fig_num in plt.get_fignums():
        path = FIGURES_DIR / f"{current_plot_name}_{plot_number}.png"
        plt.figure(fig_num).savefig(path, dpi=150, bbox_inches="tight")
        print("saved:", path)
        plot_number += 1
    plt.close("all")


plt.show = save_plots


def run_step(plot_name, func, **kwargs):
    global current_plot_name, plot_number
    current_plot_name = clean_name(plot_name)
    plot_number = 1
    result = func(**kwargs)
    save_plots()
    return result


def load_datasets():
    datasets = {}
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    datasets["Breast_Cancer"] = (X, y, 5, 3000)
    adult = fetch_ucirepo(id=2)
    X = adult.data.features.copy()
    y = adult.data.targets.copy()
    X = X.drop(
        columns=[
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]
    )
    y = y["income"].astype(str).str.replace(".", "", regex=False)
    datasets["Adult_Income"] = (X, y, 10, 3000)
    covertype = fetch_ucirepo(id=31)
    X = covertype.data.features.copy()
    y = covertype.data.targets["Cover_Type"]
    X = X.drop(columns=list(X.filter(regex="Wilderness_Area")))
    X = X.drop(columns=list(X.filter(regex="Soil_Type")))
    datasets["Covertype"] = (X, y, 10, 3000)

    X, y = fetch_openml(
        "mnist_784",
        version=1,
        return_X_y=True,
        as_frame=False,
        parser="auto",
    )
    datasets["MNIST"] = (X, y.astype(int), 10, 1000)

    return datasets


def run_all():
    for dataset_name, (X, y, min_samples, max_points) in load_datasets().items():
        print("\nRunning", dataset_name)
        kmeans_results = run_step(
            f"{dataset_name}_kmeans",
            kmeans_task,
            dataset_name=dataset_name,
            X=X,
            y=y,
            max_points=max_points,
        )
        best_k = int(kmeans_results.loc[kmeans_results["ARI"].idxmax(), "k"])
        dbscan_results, best_dbscan = run_step(
            f"{dataset_name}_dbscan_eps_search",
            find_best_dbscan_eps,
            dataset_name=dataset_name,
            X=X,
            y=y,
            min_samples=min_samples,
            max_points=max_points,
        )

        best_eps = float(best_dbscan["eps"])
        run_step(
            f"{dataset_name}_pca",
            pca_task,
            dataset_name=dataset_name,
            X=X,
            y=y,
            n_clusters=best_k,
            dbscan_eps=best_eps,
            dbscan_min_samples=min_samples,
            max_points=max_points,
        )

        run_step(
            f"{dataset_name}_dbscan",
            dbscan_task,
            dataset_name=dataset_name,
            X=X,
            y=y,
            eps=best_eps,
            min_samples=min_samples,
            max_points=max_points,
        )

        run_step(
            f"{dataset_name}_dbscan_best_eps",
            plot_dbscan_best_eps,
            dataset_name=dataset_name,
            X=X,
            y=y,
            best_eps=best_eps,
            min_samples=min_samples,
            max_points=max_points,
        )

    print("\nDone. All plots are in the figures folder.")


if __name__ == "__main__":
    run_all()
