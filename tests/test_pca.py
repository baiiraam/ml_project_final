import numpy as np
from src.unsupervised.pca import PCA

def generate_correlated_data(seed=42):
    rng = np.random.default_rng(seed)
    n_samples = 120
    z1 = rng.normal(0, 3, size=n_samples)
    z2 = rng.normal(0, 1, size=n_samples)
    x1 = z1 + rng.normal(0, 0.2, size=n_samples)
    x2 = 0.8 * z1 + rng.normal(0, 0.2, size=n_samples)
    x3 = -0.5 * z1 + rng.normal(0, 0.2, size=n_samples)
    x4 = z2 + rng.normal(0, 0.2, size=n_samples)
    x5 = 0.5 * z2 + rng.normal(0, 0.2, size=n_samples)
    X = np.column_stack([x1, x2, x3, x4, x5])
    return X

def test_pca_fit_transform_shape():
    X = generate_correlated_data()
    pca = PCA(n_components=2)
    X_transformed = pca.fit_transform(X)

    assert X_transformed.shape == (120, 2)
    assert pca.components_.shape == (2, 5)
    assert pca.mean_.shape == (5,)
    assert pca.explained_variance_ratio_.shape == (2,)

def test_pca_explained_variance_ratio_valid():
    X = generate_correlated_data()
    pca = PCA(n_components=3)
    pca.fit(X)

    assert np.all(pca.explained_variance_ratio_ >= 0)
    assert np.sum(pca.explained_variance_ratio_) <= 1.0 + 1e-8
    assert pca.explained_variance_ratio_[0] >= pca.explained_variance_ratio_[1]

def test_pca_components_are_orthonormal():
    X = generate_correlated_data()
    pca = PCA(n_components=3)
    pca.fit(X)
    product = pca.components_ @ pca.components_.T

    assert np.allclose(product, np.eye(3), atol=1e-6)

def test_pca_inverse_transform_shape():
    X = generate_correlated_data()
    pca = PCA(n_components=2)
    X_transformed = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(X_transformed)

    assert X_reconstructed.shape == X.shape

def test_pca_transform_before_fit_error():
    X = generate_correlated_data()
    pca = PCA(n_components=2)
    error_raised = False
    try:
        pca.transform(X)
    except ValueError:
        error_raised = True

    assert error_raised

def test_pca_wrong_n_components_error():
    X = generate_correlated_data()
    error_raised = False
    try:
        PCA(n_components=10).fit(X)
    except ValueError:
        error_raised = True

    assert error_raised