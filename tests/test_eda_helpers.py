"""
Tests for eda_helpers.py
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from sklearn.datasets import make_classification, make_blobs

# Import the module to test
import sys

sys.path.append("..")

from src.utils.eda_helpers import (
    plot_all_histograms,
    plot_correlation_heatmap,
    plot_pairplot,
    plot_kde_grid,
    plot_boxplots,
    DimensionalityReducer,
    reduce_and_plot,
    get_box_dist_plot,
    find_outliers,
)


# ============================================
# Fixtures
# ============================================


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "feature_1": np.random.normal(0, 1, 100),
            "feature_2": np.random.normal(5, 2, 100),
            "feature_3": np.random.exponential(1, 100),
            "feature_4": np.random.uniform(-10, 10, 100),
            "target": np.random.choice([0, 1], 100),
        }
    )
    return df


@pytest.fixture
def sample_data():
    """Create synthetic data for dimensionality reduction tests."""
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=3,
        random_state=42,
    )
    return X, y


@pytest.fixture
def sample_blobs():
    """Create blob data for visualization tests."""
    X, y = make_blobs(
        n_samples=100, n_features=2, centers=3, cluster_std=1.0, random_state=42
    )
    return X, y


# ============================================
# Helper: Create proper mock axes
# ============================================


def create_mock_axes(n_rows, n_cols):
    """Create a properly structured mock axes array."""

    # Create mock axes with all needed methods
    def create_mock_ax():
        mock_ax = Mock()
        mock_ax.get_xticks.return_value = np.array([0, 1])
        mock_ax.get_xticklabels.return_value = ["0", "1"]
        mock_ax.set_title = Mock()
        mock_ax.set_xlabel = Mock()
        mock_ax.set_ylabel = Mock()
        mock_ax.get_legend = Mock(return_value=None)
        return mock_ax

    if n_rows == 1 and n_cols == 1:
        return create_mock_ax()

    # Create a list of mock axes
    axes_list = [[create_mock_ax() for _ in range(n_cols)] for _ in range(n_rows)]

    # Convert to numpy array for proper shape
    axes_array = np.array(axes_list)

    # For 1D cases, return a 1D array
    if n_rows == 1:
        return axes_array[0]
    elif n_cols == 1:
        return axes_array[:, 0]

    return axes_array


# ============================================
# Tests for plot_all_histograms
# ============================================


@patch("src.utils.eda_helpers.plt")
@patch("src.utils.eda_helpers.sns")
def test_plot_all_histograms_no_hue(mock_sns, mock_plt, sample_df):
    """Test histogram plotting without hue."""
    columns = ["feature_1", "feature_2", "feature_3"]
    n_cols = 2
    n_rows = 2

    # Mock subplots
    mock_fig = Mock()
    mock_axes = create_mock_axes(n_rows, n_cols)
    mock_plt.subplots.return_value = (mock_fig, mock_axes)

    # Call the function
    plot_all_histograms(
        sample_df, columns=columns, title="Test Histograms", n_cols=n_cols
    )

    # Verify subplots was called
    mock_plt.subplots.assert_called_once()

    # Verify sns.histplot was called for each column
    assert mock_sns.histplot.call_count == len(columns)

    # Verify plt.show was called (and mocked, so no plot appears)
    mock_plt.show.assert_called_once()


@patch("src.utils.eda_helpers.plt")
@patch("src.utils.eda_helpers.sns")
def test_plot_all_histograms_with_hue(mock_sns, mock_plt, sample_df):
    """Test histogram plotting with hue coloring."""
    columns = ["feature_1", "feature_2"]
    hue_col = "target"
    n_cols = 2
    n_rows = 1

    # Mock subplots
    mock_fig = Mock()
    mock_axes = create_mock_axes(n_rows, n_cols)
    mock_plt.subplots.return_value = (mock_fig, mock_axes)

    # Mock sns functions
    mock_sns.color_palette.return_value = ["blue", "green"]

    # Call the function
    plot_all_histograms(
        sample_df, columns=columns, hue_col=hue_col, title="Test Histograms with Hue"
    )

    # Verify subplots was called
    mock_plt.subplots.assert_called_once()

    # Verify sns.histplot was called for each class
    assert mock_sns.histplot.call_count >= len(columns)


@patch("src.utils.eda_helpers.plt")
@patch("src.utils.eda_helpers.sns")
def test_plot_all_histograms_single_column(mock_sns, mock_plt, sample_df):
    """Test histogram plotting with a single column."""
    columns = ["feature_1"]
    n_cols = 1
    n_rows = 1

    # Mock subplots for single column
    mock_fig = Mock()
    mock_axes = create_mock_axes(n_rows, n_cols)
    mock_plt.subplots.return_value = (mock_fig, mock_axes)

    # Call the function
    plot_all_histograms(sample_df, columns=columns, title="Single Histogram")

    # Verify subplots was called
    mock_plt.subplots.assert_called_once()


def test_plot_all_histograms_invalid_column(sample_df):
    """Test histogram plotting with invalid column name."""
    columns = ["invalid_column"]

    # This should raise a ValueError because the column doesn't exist
    with pytest.raises(ValueError) as exc_info:
        plot_all_histograms(sample_df, columns=columns)

    assert "invalid_column" in str(exc_info.value)


# ============================================
# Tests for plot_correlation_heatmap
# ============================================


@patch("src.utils.eda_helpers.plt")
@patch("src.utils.eda_helpers.sns")
def test_plot_correlation_heatmap(mock_sns, mock_plt, sample_df):
    """Test correlation heatmap plotting."""
    # Mock figure and axes
    mock_fig = Mock()
    mock_ax = Mock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    # Mock corr to return a proper DataFrame
    mock_corr = pd.DataFrame(
        np.random.randn(5, 5), index=sample_df.columns, columns=sample_df.columns
    )
    sample_df.corr = Mock(return_value=mock_corr)

    # Call the function
    plot_correlation_heatmap(
        sample_df, title="Test Correlation", figsize=(10, 8), annot=True
    )

    # Verify corr was called
    sample_df.corr.assert_called_once()

    # Verify sns.heatmap was called
    mock_sns.heatmap.assert_called_once()

    # Verify plt.show was called
    mock_plt.show.assert_called_once()


@patch("src.utils.eda_helpers.plt")
@patch("src.utils.eda_helpers.sns")
def test_plot_correlation_heatmap_no_annot(mock_sns, mock_plt, sample_df):
    """Test correlation heatmap without annotations."""
    mock_fig = Mock()
    mock_ax = Mock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    mock_corr = pd.DataFrame(
        np.random.randn(5, 5), index=sample_df.columns, columns=sample_df.columns
    )
    sample_df.corr = Mock(return_value=mock_corr)

    plot_correlation_heatmap(sample_df, annot=False)

    # Verify heatmap called with annot=False
    call_kwargs = mock_sns.heatmap.call_args[1]
    assert call_kwargs.get("annot") is False


@patch("src.utils.eda_helpers.plt")
@patch("src.utils.eda_helpers.sns")
def test_plot_correlation_heatmap_no_mask(mock_sns, mock_plt, sample_df):
    """Test correlation heatmap without upper triangle mask."""
    mock_fig = Mock()
    mock_ax = Mock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    mock_corr = pd.DataFrame(
        np.random.randn(5, 5), index=sample_df.columns, columns=sample_df.columns
    )
    sample_df.corr = Mock(return_value=mock_corr)

    plot_correlation_heatmap(sample_df, mask_upper=False)

    # Verify heatmap called with mask=None
    call_kwargs = mock_sns.heatmap.call_args[1]
    assert call_kwargs.get("mask") is None


# ============================================
# Tests for plot_pairplot
# ============================================


@patch("src.utils.eda_helpers.plt")
@patch("src.utils.eda_helpers.sns")
def test_plot_pairplot_with_hue(mock_sns, mock_plt, sample_df):
    """Test pairplot with hue coloring."""
    hue_col = "target"

    # Mock pairplot - properly mock get_texts()
    mock_pairplot = Mock()
    mock_pairplot.fig = Mock()
    mock_legend = Mock()
    # Mock get_texts to return an empty list (no text objects)
    mock_legend.get_texts.return_value = []
    mock_pairplot._legend = mock_legend
    mock_sns.pairplot.return_value = mock_pairplot

    # Call the function
    plot_pairplot(sample_df, hue_col=hue_col, title="Test Pairplot with Hue")

    # Verify pairplot was called with hue
    call_kwargs = mock_sns.pairplot.call_args[1]
    assert call_kwargs["hue"] == hue_col

    # Verify legend was updated
    assert mock_legend.set_bbox_to_anchor.called
    assert mock_legend.set_loc.called


# ============================================
# Tests for plot_kde_grid
# ============================================


@patch("src.utils.eda_helpers.plt")
@patch("src.utils.eda_helpers.sns")
def test_plot_kde_grid(mock_sns, mock_plt, sample_df):
    """Test KDE grid plotting."""
    columns = ["feature_1", "feature_2", "feature_3"]
    hue_col = "target"
    n_cols = 2
    n_rows = 2

    # Mock subplots with proper axes
    mock_fig = Mock()
    mock_axes = create_mock_axes(n_rows, n_cols)
    mock_plt.subplots.return_value = (mock_fig, mock_axes)

    # Mock color palette
    mock_sns.color_palette.return_value = ["blue", "green"]

    # Call the function
    plot_kde_grid(
        sample_df,
        columns=columns,
        hue_col=hue_col,
        n_cols=n_cols,
        legend_loc="upper center",
    )

    # Verify subplots was called
    mock_plt.subplots.assert_called_once()

    # Verify sns.kdeplot was called
    assert mock_sns.kdeplot.call_count > 0


# ============================================
# Tests for plot_boxplots
# ============================================


@patch("src.utils.eda_helpers.plt")
@patch("src.utils.eda_helpers.sns")
def test_plot_boxplots_no_hue(mock_sns, mock_plt, sample_df):
    """Test boxplot grid without hue."""
    columns = ["feature_1", "feature_2"]
    n_cols = 2
    n_rows = 1

    mock_fig = Mock()
    mock_axes = create_mock_axes(n_rows, n_cols)
    mock_plt.subplots.return_value = (mock_fig, mock_axes)

    plot_boxplots(sample_df, columns=columns, n_cols=n_cols)

    assert mock_sns.boxplot.call_count == len(columns)


@patch("src.utils.eda_helpers.plt")
@patch("src.utils.eda_helpers.sns")
def test_plot_boxplots_with_hue(mock_sns, mock_plt, sample_df):
    """Test boxplot grid with hue."""
    columns = ["feature_1", "feature_2"]
    hue_col = "target"
    n_cols = 2
    n_rows = 1

    mock_fig = Mock()
    # create_mock_axes already configures get_xticks and get_xticklabels
    mock_axes = create_mock_axes(n_rows, n_cols)
    mock_plt.subplots.return_value = (mock_fig, mock_axes)

    plot_boxplots(sample_df, columns=columns, hue_col=hue_col, n_cols=n_cols)

    # Verify boxplot called with hue
    call_kwargs = mock_sns.boxplot.call_args_list[0][1]
    assert call_kwargs["x"] == hue_col


# ============================================
# Tests for DimensionalityReducer
# ============================================


def test_dimensionality_reducer_init():
    """Test DimensionalityReducer initialization."""
    reducer = DimensionalityReducer(n_components=3, random_state=123)
    assert reducer.n_components == 3
    assert reducer.random_state == 123
    assert reducer.results == {}
    assert reducer.models == {}


def test_dimensionality_reducer_fit_transform(sample_data):
    """Test DimensionalityReducer fit_transform method."""
    X, y = sample_data

    reducer = DimensionalityReducer(n_components=2, random_state=42)
    results = reducer.fit_transform(X, y, methods=["PCA", "t-SNE"])

    # Verify results
    assert "PCA" in results
    assert "t-SNE" in results
    assert results["PCA"].shape == (X.shape[0], 2)
    assert results["t-SNE"].shape == (X.shape[0], 2)

    # Verify models stored
    assert "PCA" in reducer.models
    assert "t-SNE" in reducer.models


def test_dimensionality_reducer_fit_transform_no_methods(sample_data):
    """Test DimensionalityReducer with no methods specified (uses all)."""
    X, y = sample_data

    reducer = DimensionalityReducer(n_components=2, random_state=42)
    results = reducer.fit_transform(X, y, methods=None)  # Use all methods

    # Should have PCA and t-SNE (the only two available)
    assert "PCA" in results
    assert "t-SNE" in results


def test_dimensionality_reducer_fit_transform_unknown_method(sample_data):
    """Test DimensionalityReducer with unknown method."""
    X, y = sample_data

    reducer = DimensionalityReducer(n_components=2, random_state=42)
    results = reducer.fit_transform(X, y, methods=["UnknownMethod"])

    # Should not have any results
    assert len(results) == 0


def test_dimensionality_reducer_fit_transform_lda_removed(sample_data):
    """Test DimensionalityReducer when LDA is requested but not available."""
    X, y = sample_data

    reducer = DimensionalityReducer(n_components=2, random_state=42)
    results = reducer.fit_transform(X, y, methods=["LDA"])

    # LDA should not be in results (it was removed)
    assert "LDA" not in results
    assert len(results) == 0


@patch("src.utils.eda_helpers.plt")
def test_dimensionality_reducer_plot_results(mock_plt, sample_data):
    """Test DimensionalityReducer plot_results method."""
    X, y = sample_data

    reducer = DimensionalityReducer(n_components=2, random_state=42)
    reducer.fit_transform(X, y, methods=["PCA", "t-SNE"])

    # Mock subplots
    mock_fig = Mock()
    mock_ax = Mock()
    mock_plt.subplots.return_value = (mock_fig, [mock_ax, mock_ax])

    # Call plot_results
    reducer.plot_results(y, title_prefix="Test")

    # Verify plt was called
    mock_plt.subplots.assert_called_once()
    mock_plt.show.assert_called_once()


def test_dimensionality_reducer_plot_results_no_results():
    """Test plot_results when no results are available."""
    reducer = DimensionalityReducer()

    # Should print message and return without error
    reducer.plot_results(np.array([0, 1]))


# ============================================
# Tests for reduce_and_plot
# ============================================


@patch("src.utils.eda_helpers.DimensionalityReducer")
def test_reduce_and_plot(mock_reducer_class, sample_data):
    """Test reduce_and_plot convenience function."""
    X, y = sample_data

    # Mock reducer
    mock_reducer = Mock()
    mock_reducer.fit_transform.return_value = {"PCA": np.zeros((X.shape[0], 2))}
    mock_reducer.plot_results = Mock()
    mock_reducer_class.return_value = mock_reducer

    # Call function
    reducer = reduce_and_plot(X, y, methods=["PCA"], n_components=2, title="Test")

    # Verify reducer was used
    mock_reducer.fit_transform.assert_called_once()
    mock_reducer.plot_results.assert_called_once()

    # Verify returned reducer
    assert reducer == mock_reducer


# ============================================
# Tests for get_box_dist_plot
# ============================================


@patch("src.utils.eda_helpers.plt")
@patch("src.utils.eda_helpers.sns")
def test_get_box_dist_plot(mock_sns, mock_plt):
    """Test get_box_dist_plot function."""
    x = np.random.normal(0, 1, 100)

    # Mock subplots
    mock_fig = Mock()
    mock_ax_box = Mock()
    mock_ax_hist = Mock()
    mock_plt.subplots.return_value = (mock_fig, (mock_ax_box, mock_ax_hist))

    # Call the function
    get_box_dist_plot(x)

    # Verify subplots was called
    mock_plt.subplots.assert_called_once()

    # Verify sns.boxplot and sns.histplot were called
    mock_sns.boxplot.assert_called_once()
    mock_sns.histplot.assert_called_once()

    # Verify plt.show was called (mocked, so no plot appears)
    mock_plt.show.assert_called_once()


# ============================================
# Tests for find_outliers
# ============================================


def test_find_outliers():
    """Test find_outliers function."""
    # Create data with clear outliers
    np.random.seed(42)
    data = np.random.randn(100, 5)
    # Add some outliers
    data[0] = [100, 100, 100, 100, 100]
    data[1] = [-100, -100, -100, -100, -100]

    indices, distances = find_outliers(data, n_outliers=2)

    # Should find the two outliers
    assert len(indices) == 2
    assert 0 in indices
    assert 1 in indices
    assert len(distances) == 2
    assert distances[0] > 0
    assert distances[1] > 0


def test_find_outliers_empty():
    """Test find_outliers with empty input."""
    data = np.array([]).reshape(0, 5)
    indices, distances = find_outliers(data)

    assert len(indices) == 0
    assert len(distances) == 0


def test_find_outliers_single_class():
    """Test find_outliers with single class."""
    # Should work with single class
    data = np.random.randn(50, 3)
    indices, distances = find_outliers(data, n_outliers=3)

    assert len(indices) == 3
    assert len(distances) == 3


def test_find_outliers_few_samples():
    """Test find_outliers with fewer samples than outliers requested."""
    data = np.random.randn(3, 5)
    indices, distances = find_outliers(data, n_outliers=5)

    # Should return at most the number of samples
    assert len(indices) == 3
    assert len(distances) == 3


# ============================================
# Integration Tests
# ============================================


@patch("src.utils.eda_helpers.plt")
@patch("src.utils.eda_helpers.sns")
def test_full_eda_workflow(mock_sns, mock_plt, sample_df, sample_data):
    """Test a complete EDA workflow with the visualization module."""
    X, y = sample_data

    # Mock subplots for all plotting functions
    mock_fig = Mock()
    mock_axes = create_mock_axes(1, 2)
    mock_plt.subplots.return_value = (mock_fig, mock_axes)
    mock_sns.color_palette.return_value = ["blue", "green"]

    # Mock pairplot
    mock_pairplot = Mock()
    mock_pairplot.fig = Mock()
    mock_legend = Mock()
    mock_legend.get_texts.return_value = []
    mock_pairplot._legend = mock_legend
    mock_sns.pairplot.return_value = mock_pairplot

    # 1. Test histogram plotting
    plot_all_histograms(sample_df, columns=["feature_1", "feature_2"], hue_col="target")

    # 2. Test correlation heatmap
    mock_corr = pd.DataFrame(
        np.random.randn(5, 5), index=sample_df.columns, columns=sample_df.columns
    )
    sample_df.corr = Mock(return_value=mock_corr)
    plot_correlation_heatmap(sample_df)

    # 3. Test dimensionality reduction
    reducer = DimensionalityReducer(n_components=2)
    results = reducer.fit_transform(X, y, methods=["PCA", "t-SNE"])
    assert len(results) == 2

    # 4. Test pairplot
    plot_pairplot(sample_df, hue_col="target")

    # 5. Test boxplots
    plot_boxplots(sample_df, columns=["feature_1", "feature_2"])

    # 6. Test KDE grid
    plot_kde_grid(sample_df, columns=["feature_1", "feature_2"], hue_col="target")


# ============================================
# Run tests if executed directly
# ============================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
