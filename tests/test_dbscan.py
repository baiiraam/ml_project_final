import numpy as np
from sklearn.metrics import adjusted_rand_score
from src.unsupervised.dbscan import DBSCAN


SEED = 42


def make_dbscan_data(seed=SEED):
    rng = np.random.default_rng(seed)
    cluster_1 = rng.normal(loc=[0, 0], scale=0.10, size=(50, 2))
    cluster_2 = rng.normal(loc=[4, 4], scale=0.10, size=(50, 2))
    noise_centers = np.array([
        [8, 8],
        [-8, 8],
        [8, -8],
        [-8, -8],
        [0, 9],
        [9, 0],
    ])

    noise = noise_centers + rng.normal(0, 0.15, size=noise_centers.shape)
    X = np.vstack([cluster_1, cluster_2, noise])
    y_true = np.array(
        [0] * len(cluster_1) +
        [1] * len(cluster_2) +
        [-1] * len(noise)
    )

    return X, y_true


def test_dbscan_finds_random_density_clusters_and_noise():
    X, y_true = make_dbscan_data()
    dbscan = DBSCAN(eps=0.35, min_samples=5)
    labels = dbscan.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_count = np.sum(labels == -1)
    ari = adjusted_rand_score(y_true, labels)

    assert labels.shape == (106,)
    assert n_clusters == 2
    assert noise_count >= 5
    assert ari > 0.90


def test_dbscan_core_sample_indices_exist():
    X, _ = make_dbscan_data()
    dbscan = DBSCAN(eps=0.35, min_samples=5)
    dbscan.fit(X)

    assert dbscan.core_sample_indices_ is not None
    assert len(dbscan.core_sample_indices_) > 0
    assert dbscan.n_clusters_ == 2


def test_dbscan_fit_predict_matches_labels_attribute():
    X, _ = make_dbscan_data()
    dbscan = DBSCAN(eps=0.35, min_samples=5)
    labels = dbscan.fit_predict(X)

    assert np.array_equal(labels, dbscan.labels_)


def test_dbscan_all_noise_with_small_eps():
    X, _ = make_dbscan_data()
    dbscan = DBSCAN(eps=0.001, min_samples=5)
    dbscan.fit(X)

    assert np.all(dbscan.labels_ == -1)


def test_dbscan_invalid_eps():
    X, _ = make_dbscan_data()
    error_raised = False
    try:
        DBSCAN(eps=0, min_samples=5).fit(X)
    except ValueError:
        error_raised = True

    assert error_raised, "DBSCAN should raise ValueError when eps <= 0"

def test_dbscan_invalid_min_samples():
    X, _ = make_dbscan_data()
    error_raised = False
    try:
        DBSCAN(eps=0.35, min_samples=0).fit(X)
    except ValueError:
        error_raised = True

    assert error_raised, "DBSCAN should raise ValueError when min_samples <= 0"