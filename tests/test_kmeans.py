import numpy as np
from sklearn.metrics import adjusted_rand_score
from src.unsupervised.kmeans import KMeans

SEED = 42

def make_kmeans_data(seed=SEED):
    rng = np.random.default_rng(seed)
    n_per_cluster = 60
    centers = np.array([
        [0, 0, 0, 0],
        [5, 5, 5, 5],
        [-5, 5, -5, 5],
    ])

    X_parts = []
    y_parts = []
    for label, center in enumerate(centers):
        points = rng.normal(loc=center, scale=0.45, size=(n_per_cluster, 4))
        X_parts.append(points)
        y_parts.append(np.full(n_per_cluster, label))
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    return X, y

def test_kmeans_finds_three_random_clusters():
    X, y_true = make_kmeans_data()
    kmeans = KMeans(n_clusters=3, random_state=SEED)
    labels = kmeans.fit_predict(X)
    ari = adjusted_rand_score(y_true, labels)

    assert labels.shape == (180,)
    assert kmeans.centroids_.shape == (3, 4)
    assert kmeans.inertia_ >= 0
    assert ari > 0.95

def test_kmeans_predict_random_new_points():
    X, _ = make_kmeans_data()
    rng = np.random.default_rng(SEED + 1)
    X_new = np.array([
        [0, 0, 0, 0],
        [5, 5, 5, 5],
        [-5, 5, -5, 5],
    ])
    X_new = X_new + rng.normal(0, 0.1, size=X_new.shape)
    kmeans = KMeans(n_clusters=3, random_state=SEED)
    kmeans.fit(X)
    labels = kmeans.predict(X_new)

    assert labels.shape == (3,)
    assert len(set(labels)) == 3

def test_kmeans_fit_predict_matches_labels_attribute():
    X, _ = make_kmeans_data()
    kmeans = KMeans(n_clusters=3, random_state=SEED)
    labels = kmeans.fit_predict(X)

    assert np.array_equal(labels, kmeans.labels_)

def test_kmeans_reproducible_with_same_seed():
    X, _ = make_kmeans_data()
    model_1 = KMeans(n_clusters=3, random_state=123)
    model_2 = KMeans(n_clusters=3, random_state=123)
    labels_1 = model_1.fit_predict(X)
    labels_2 = model_2.fit_predict(X)

    assert np.array_equal(labels_1, labels_2)
    assert np.allclose(model_1.centroids_, model_2.centroids_)

def test_kmeans_invalid_n_clusters():
    X, _ = make_kmeans_data()
    error_raised = False
    try:
        KMeans(n_clusters=0).fit(X)
    except ValueError:
        error_raised = True

    assert error_raised, "KMeans should raise ValueError when n_clusters = 0"

    error_raised = False
    try:
        KMeans(n_clusters=len(X) + 1).fit(X)
    except ValueError:
        error_raised = True

    assert error_raised, "KMeans should raise ValueError when n_clusters > number of samples"

def test_kmeans_predict_before_fit_error():
    X, _ = make_kmeans_data()
    kmeans = KMeans(n_clusters=3)
    error_raised = False

    try:
        kmeans.predict(X)
    except ValueError:
        error_raised = True

    assert error_raised, "KMeans.predict() should raise ValueError before fit"