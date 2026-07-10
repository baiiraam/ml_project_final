
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

try:
    from src.unsupervised.pca import PCA
    from src.unsupervised.kmeans import KMeans
    from src.unsupervised.dbscan import DBSCAN
except ImportError:
    try:
        from src.unsupervised.pca import PCA
        from src.unsupervised.kmeans import KMeans
        from src.unsupervised.dbscan import DBSCAN
    except ImportError:
        from src.unsupervised.pca import PCA
        from src.unsupervised.kmeans import KMeans
        from src.unsupervised.dbscan import DBSCAN

__all__ = [
    "pca_task",
    "kmeans_task",
    "dbscan_task",
    "find_best_dbscan_eps",
    "plot_dbscan_best_eps",
]

def _prepare_unsupervised_data(X, y, max_points: int, *, drop_all_nan_columns: bool = True):
    """Convert X/y to numeric pandas objects, fill missing values, and sample rows."""
    X = pd.DataFrame(X).copy()
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    y = pd.Series(np.asarray(y).ravel()).copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    if drop_all_nan_columns:
        X = X.dropna(axis=1, how="all")
    X = X.fillna(X.median(numeric_only=True))
    if len(X) > max_points:
        idx = np.random.default_rng(42).choice(len(X), max_points, replace=False)
        X = X.iloc[idx].reset_index(drop=True)
        y = y.iloc[idx].reset_index(drop=True)
    X_scaled = StandardScaler().fit_transform(X)
    return X, y, X_scaled


def pca_task(dataset_name, X, y, n_clusters, dbscan_eps, dbscan_min_samples=5, max_points=5000):
    X, y, X_scaled = _prepare_unsupervised_data(
        X,
        y,
        max_points=max_points,
        drop_all_nan_columns=False,
    )
    pca = PCA(n_components=X_scaled.shape[1])
    X_pca = pca.fit_transform(X_scaled)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title(f"{dataset_name} - Scree Plot")
    plt.grid(True)
    plt.show()

    X_2d = X_pca[:, :2]
    true_labels, _ = pd.factorize(y)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_2d)
    dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    dbscan_labels = dbscan.fit_predict(X_2d)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ["True Labels", "K-Means Labels", "DBSCAN Labels"]
    labels_list = [true_labels, kmeans_labels, dbscan_labels]

    for ax, title, labels in zip(axes, titles, labels_list):
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=15)
        ax.set_title(f"{dataset_name} - {title}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        plt.colorbar(scatter, ax=ax)

    plt.tight_layout()
    plt.show()

    print(dataset_name)
    print("PC1 variance:", pca.explained_variance_ratio_[0])
    print("PC2 variance:", pca.explained_variance_ratio_[1])
    print("First 2 PCs total:", pca.explained_variance_ratio_[:2].sum())
    n_90 = np.argmax(cumulative_variance >= 0.90) + 1
    print("Number of PCs for >=90% variance:", n_90)


def kmeans_task(dataset_name, X, y, max_points=5000, n_init=10):
    _, y, X_scaled = _prepare_unsupervised_data(X, y, max_points=max_points)
    inertias = []
    aris = []

    for k in range(1, 11):
        best_inertia = np.inf
        best_labels = None
        for restart in range(n_init):
            model = KMeans(
                n_clusters=k,
                max_iter=300,
                tol=1e-4,
                random_state=42 + restart,
            )

            model.fit(X_scaled)
            if model.inertia_ < best_inertia:
                best_inertia = model.inertia_
                best_labels = model.labels_

        ari = adjusted_rand_score(y, best_labels)
        inertias.append(best_inertia)
        aris.append(ari)

    results = pd.DataFrame({
        "k": range(1, 11),
        "inertia": inertias,
        "ARI": aris,
    })
    plt.figure(figsize=(8, 5))
    plt.plot(results["k"], results["inertia"], marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title(f"{dataset_name} - Elbow Method")
    plt.xticks(range(1, 11))
    plt.grid(True)
    plt.show()
    print(dataset_name)
    display(results)
    best_row = results.loc[results["ARI"].idxmax()]
    print("Best ARI:", best_row["ARI"])
    print("Best k by ARI:", int(best_row["k"]))

    return results

def dbscan_task(dataset_name, X, y, eps, min_samples=5, max_points=3000):
    _, y, X_scaled = _prepare_unsupervised_data(X, y, max_points=max_points)
    distances = np.sqrt(
        np.sum((X_scaled[:, None, :] - X_scaled[None, :, :]) ** 2, axis=2)
    )
    kth_distances = np.partition(distances, min_samples, axis=1)[:, min_samples]
    kth_distances_sorted = np.sort(kth_distances)
    plt.figure(figsize=(8, 5))
    plt.plot(kth_distances_sorted)
    plt.axhline(eps, linestyle="--", label=f"chosen eps = {eps}")
    plt.xlabel("Points sorted by distance")
    plt.ylabel(f"Distance to {min_samples}-th nearest neighbor")
    plt.title(f"{dataset_name} - k-distance Plot")
    plt.grid(True)
    plt.legend()
    plt.show()

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)
    ari = adjusted_rand_score(y, labels)
    noise_fraction = np.mean(labels == -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    print(dataset_name)
    print("eps:", eps)
    print("min_samples:", min_samples)
    print("DBSCAN clusters:", n_clusters)
    print("ARI:", ari)
    print("Noise fraction:", noise_fraction)

    return {
        "dataset": dataset_name,
        "eps": eps,
        "min_samples": min_samples,
        "ARI": ari,
        "noise_fraction": noise_fraction,
        "n_clusters": n_clusters,
    }

def find_best_dbscan_eps(dataset_name, X, y, min_samples=5, max_points=3000):
    _, y, X_scaled = _prepare_unsupervised_data(X, y, max_points=max_points)
    distances = np.sqrt(
        np.sum((X_scaled[:, None, :] - X_scaled[None, :, :]) ** 2, axis=2)
    )
    kth_distances = np.partition(distances, min_samples, axis=1)[:, min_samples]
    kth_distances_sorted = np.sort(kth_distances)
    plt.figure(figsize=(8, 5))
    plt.plot(kth_distances_sorted)
    plt.xlabel("Points sorted by distance")
    plt.ylabel(f"Distance to {min_samples}-th nearest neighbor")
    plt.title(f"{dataset_name} - k-distance Plot")
    plt.grid(True)
    plt.show()
    eps_values = np.percentile(
        kth_distances_sorted,
        [50, 60, 70, 80, 85, 90, 95, 97],
    )
    eps_values = np.unique(np.round(eps_values, 3))
    rows = []

    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        ari = adjusted_rand_score(y, labels)
        noise_fraction = np.mean(labels == -1)
        rows.append({
            "eps": eps,
            "clusters": n_clusters,
            "ARI": ari,
            "noise_fraction": noise_fraction,
        })

    results = pd.DataFrame(rows)
    display(results)
    best_row = results.loc[results["ARI"].idxmax()]

    print(dataset_name)
    print("Best eps:", best_row["eps"])
    print("Best ARI:", best_row["ARI"])
    print("Clusters:", int(best_row["clusters"]))
    print("Noise fraction:", best_row["noise_fraction"])

    return results, best_row

def plot_dbscan_best_eps(dataset_name, X, y, best_eps, min_samples=5, max_points=3000):
    _, y, X_scaled = _prepare_unsupervised_data(X, y, max_points=max_points)
    dbscan = DBSCAN(eps=best_eps, min_samples=min_samples)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)

    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=dbscan_labels,
        cmap="tab10",
        s=18,
        alpha=0.75,
    )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"{dataset_name} - DBSCAN Clusters (eps={best_eps})")
    plt.colorbar(scatter, label="DBSCAN label")
    plt.grid(alpha=0.3)
    plt.show()

    print(dataset_name)
    print("eps:", best_eps)
    print("min_samples:", min_samples)
    print("clusters:", len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0))
    print("noise fraction:", np.mean(dbscan_labels == -1))
