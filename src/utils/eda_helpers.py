import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import seaborn as sns
import pandas as pd


def plot_all_histograms(
    df,
    columns,
    hue_col=None,
    title="Histograms",
    n_cols=2,
    figsize=(15, 15),
    kde=True,
    alpha=0.7,
    bins="auto",
    edgecolor="black",
    palette=None,
    legend_fontsize=12,
    legend_title_fontsize=14,
):
    """
    Plot histograms with KDE for multiple columns in a grid with optional hue coloring.
    """
    n_cols = min(n_cols, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes_flat = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    use_hue = hue_col is not None and hue_col in df.columns

    # Store legend info
    legend_handles = None
    legend_labels = None

    for i, col in enumerate(columns):
        if use_hue:
            # Get unique classes
            classes = sorted(df[hue_col].unique())

            # Determine colors
            if palette is None:
                if len(classes) <= 2:
                    colors = ["blue", "green"]
                else:
                    colors = sns.color_palette("tab10", len(classes))
            else:
                if isinstance(palette, list):
                    colors = palette[: len(classes)]
                else:
                    colors = sns.color_palette(palette, len(classes))

            color_map = {cls: colors[i] for i, cls in enumerate(classes)}

            # Plot each class separately
            for cls in classes:
                subset = df[df[hue_col] == cls]
                if len(subset) > 0:
                    sns.histplot(
                        data=subset,
                        x=col,
                        kde=kde,
                        alpha=alpha,
                        bins=bins,
                        edgecolor=edgecolor,
                        color=color_map[cls],
                        label=str(cls),
                        ax=axes_flat[i],
                    )

            # Store legend info from first plot
            if i == 0:
                from matplotlib.patches import Patch

                legend_handles = [Patch(color=color_map[cls]) for cls in classes]
                legend_labels = [str(cls) for cls in classes]
        else:
            # No hue - single color plot
            sns.histplot(
                data=df,
                x=col,
                kde=kde,
                alpha=alpha,
                bins=bins,
                edgecolor=edgecolor,
                ax=axes_flat[i],
            )

        axes_flat[i].set_title(f"{col}")
        axes_flat[i].set_xlabel("")

        # Remove individual legend
        legend = axes_flat[i].get_legend()
        if legend:
            legend.remove()

    # Remove empty subplots
    for j in range(len(columns), len(axes_flat)):
        fig.delaxes(axes_flat[j])

    # Add single legend if hue is used
    if use_hue and legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            title=hue_col,
            title_fontsize=legend_title_fontsize,
            fontsize=legend_fontsize,
            handlelength=2,
            handleheight=1.5,
            handletextpad=1,
            ncol=len(legend_labels),
        )
        plt.tight_layout(rect=[0, 0, 1, 0.92])
    else:
        plt.tight_layout()

    fig.suptitle(title, fontsize=16, y=1.02)
    plt.show()


def plot_correlation_heatmap(
    df,
    title="Correlation Matrix",
    figsize=(14, 12),
    annot=True,
    annot_size=7,
    fmt=".2f",
    vmin=-1,
    vmax=1,
    cmap=None,
    mask_upper=True,
):
    """
    Plot a correlation heatmap with customizable options.

    Args:
        df (DataFrame): Input DataFrame
        title (str): Title of the plot
        figsize (tuple): Figure size (width, height)
        annot (bool): Whether to show correlation values
        annot_size (int): Font size for annotations
        fmt (str): Format string for annotations
        vmin (float): Minimum value for colormap
        vmax (float): Maximum value for colormap
        cmap (str or colormap): Colormap to use (default: diverging palette)
        mask_upper (bool): Whether to mask the upper triangle
    """
    # Calculate correlation matrix
    corr = df.corr()

    # Generate mask for upper triangle if requested
    mask = np.triu(np.ones_like(corr, dtype=bool)) if mask_upper else None

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=figsize)

    # Generate colormap if not provided
    if cmap is None:
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Create heatmap
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.6},
        annot=annot,
        fmt=fmt if annot else None,
        annot_kws={"size": annot_size} if annot else None,
        ax=ax,
    )

    plt.title(title, fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()


def plot_pairplot(
    df,
    hue_col=None,
    columns=None,
    diag_kind="kde",
    palette=None,
    figsize=None,
    title=None,
    legend_loc="upper right",
    legend_fontsize=12,
    legend_title_fontsize=14,
    height=2.5,
):
    """
    Create a pairplot for selected columns with hue coloring.

    Args:
        df (DataFrame): Input DataFrame
        hue_col (str): Column name for coloring (target variable)
        columns (list): List of column names to include (if None, uses all numeric columns)
        diag_kind (str): 'kde' or 'hist' for diagonal plots
        palette (list or dict): Color palette for hue categories
        figsize (tuple): Figure size (not directly used by pairplot, but for reference)
        title (str): Title for the plot
        legend_loc (str): Legend location ('upper right', 'lower right', 'best', etc.)
        legend_fontsize (int): Font size for legend labels
        legend_title_fontsize (int): Font size for legend title
        height (float): Height of each subplot in inches

    Returns:
        PairGrid: The pairplot object
    """
    # Select columns if specified
    if columns is not None:
        # Ensure hue column is included
        if hue_col is not None and hue_col not in columns:
            columns = list(columns) + [hue_col]
        df_subset = df[columns]
    else:
        df_subset = df

    # Set default palette if not provided
    if palette is None:
        if hue_col is not None and df[hue_col].nunique() <= 2:
            palette = ["blue", "green"]
        else:
            palette = "tab10"

    # Create pairplot
    pairplot = sns.pairplot(
        df_subset,
        hue=hue_col,
        diag_kind=diag_kind,
        palette=palette,
        height=height,
        plot_kws={"alpha": 0.6, "s": 30},
    )

    # Customize the legend
    if hue_col is not None:
        # Get the legend
        legend = pairplot._legend

        # Set legend position
        if legend_loc == "upper right":
            legend.set_bbox_to_anchor((0.98, 0.98))
            legend.set_loc("upper right")
        elif legend_loc == "upper left":
            legend.set_bbox_to_anchor((0.02, 0.98))
            legend.set_loc("upper left")
        elif legend_loc == "lower right":
            legend.set_bbox_to_anchor((0.98, 0.02))
            legend.set_loc("lower right")
        elif legend_loc == "lower left":
            legend.set_bbox_to_anchor((0.02, 0.02))
            legend.set_loc("lower left")
        elif legend_loc == "best":
            legend.set_loc("best")
        else:
            # Default: put legend outside on the right
            legend.set_bbox_to_anchor((1.02, 0.5))
            legend.set_loc("center left")

        # Set font sizes
        legend.set_title(hue_col, prop={"size": legend_title_fontsize})
        for text in legend.get_texts():
            text.set_fontsize(legend_fontsize)

    # Add title if provided
    if title:
        pairplot.fig.suptitle(title, y=1.02, fontsize=16)

    plt.tight_layout()
    plt.show()

    return pairplot


def plot_kde_grid(
    df,
    columns,
    hue_col,
    n_cols=2,
    figsize=(15, 20),
    fill=True,
    alpha=0.5,
    palette=None,
    legend_loc="upper center",
    legend_fontsize=12,
    legend_title_fontsize=14,
    handlelength=2,
    handleheight=1.5,
):
    """
    Plot KDE plots for multiple columns in a grid with hue coloring.

    Args:
        legend_loc (str): Legend location ('upper center', 'upper right', etc.)
        legend_fontsize (int): Font size for legend labels
        legend_title_fontsize (int): Font size for legend title
        handlelength (float): Length of color swatches in legend
        handleheight (float): Height of color swatches in legend
    """
    n_rows = (len(columns) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes_flat = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    # Get unique classes for legend
    classes = sorted(df[hue_col].unique())

    # Determine colors
    if palette is None:
        if len(classes) <= 2:
            colors = ["blue", "green"]
        else:
            colors = sns.color_palette("tab10", len(classes))
    else:
        if isinstance(palette, list):
            colors = palette[: len(classes)]
        else:
            colors = sns.color_palette(palette, len(classes))

    # Create color mapping
    color_map = {cls: colors[i] for i, cls in enumerate(classes)}

    # Plot each class separately
    for i, col in enumerate(columns):
        for cls in classes:
            subset = df[df[hue_col] == cls]
            if len(subset) > 0:
                sns.kdeplot(
                    data=subset,
                    x=col,
                    fill=fill,
                    alpha=alpha,
                    color=color_map[cls],
                    label=str(cls),
                    ax=axes_flat[i],
                )

        axes_flat[i].set_title(col)
        axes_flat[i].set_xlabel("")

        # Remove individual legend
        legend = axes_flat[i].get_legend()
        if legend:
            legend.remove()

    # Remove empty subplots
    for j in range(len(columns), len(axes_flat)):
        fig.delaxes(axes_flat[j])

    # Create legend patches
    from matplotlib.patches import Patch

    legend_handles = [Patch(color=color_map[cls]) for cls in classes]
    legend_labels = [str(cls) for cls in classes]

    # Add single legend with custom sizes
    if legend_loc == "upper center":
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            title=hue_col,
            title_fontsize=legend_title_fontsize,
            fontsize=legend_fontsize,
            handlelength=handlelength,
            handleheight=handleheight,
            handletextpad=1,
            ncol=len(classes),
        )
        plt.tight_layout(rect=[0, 0, 1, 0.92])
    elif legend_loc == "right":
        fig.legend(
            legend_handles,
            legend_labels,
            loc="center right",
            bbox_to_anchor=(1.02, 0.5),
            title=hue_col,
            title_fontsize=legend_title_fontsize,
            fontsize=legend_fontsize,
            handlelength=handlelength,
            handleheight=handleheight,
            handletextpad=1,
        )
        plt.tight_layout(rect=[0, 0, 0.88, 1])
    else:
        fig.legend(
            legend_handles,
            legend_labels,
            loc=legend_loc,
            title=hue_col,
            title_fontsize=legend_title_fontsize,
            fontsize=legend_fontsize,
            handlelength=handlelength,
            handleheight=handleheight,
            handletextpad=1,
        )
        plt.tight_layout()

    plt.show()


def plot_boxplots(df, columns, hue_col=None, figsize=(15, 10), n_cols=3, palette=None):
    """
    Plot boxplots for multiple columns in a grid.

    Args:
        df (DataFrame): Input DataFrame
        columns (list): List of column names to plot
        hue_col (str): Column name for coloring (optional)
        figsize (tuple): Figure size (width, height)
        n_cols (int): Number of columns in the grid
        palette (list or dict): Color palette for hue categories
    """
    n_cols = min(n_cols, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes_flat = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    use_hue = hue_col is not None and hue_col in df.columns

    for i, col in enumerate(columns):
        if use_hue:
            # Use hue parameter to properly apply palette
            sns.boxplot(
                data=df,
                x=hue_col,
                y=col,
                hue=hue_col,
                palette=palette,
                ax=axes_flat[i],
                legend=False,
            )  # Remove individual legends
        else:
            sns.boxplot(data=df, y=col, ax=axes_flat[i])

        axes_flat[i].set_title(col)
        axes_flat[i].set_xlabel("")

        # Rotate x-axis tick labels for better readability - FIXED
        if use_hue:
            # Get the current tick positions and set them with rotation
            ticks = axes_flat[i].get_xticks()
            tick_labels = axes_flat[i].get_xticklabels()
            if len(ticks) == len(tick_labels):
                axes_flat[i].set_xticks(ticks)
                axes_flat[i].set_xticklabels(tick_labels, rotation=45)
            else:
                # Fallback: just rotate without setting ticks
                for label in tick_labels:
                    label.set_rotation(45)

    # Remove empty subplots
    for j in range(len(columns), len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.tight_layout()
    plt.show()


class DimensionalityReducer:
    """
    A unified class for applying dimensionality reduction techniques
    and visualizing the results.
    """

    def __init__(self, n_components=2, random_state=42):
        """
        Initialize the dimensionality reducer.

        Args:
            n_components (int): Number of components for reduction
            random_state (int): Random seed for reproducibility
        """
        self.n_components = n_components
        self.random_state = random_state
        self.results = {}
        self.models = {}

    def get_reducers(self):
        """Return dictionary of available reducers."""
        reducers = {
            "PCA": PCA(n_components=self.n_components, random_state=self.random_state),
            "t-SNE": TSNE(
                n_components=self.n_components,
                random_state=self.random_state,
                perplexity=30,
                max_iter=500,
                verbose=0,
            ),
        }
        return reducers

    def fit_transform(self, X, y=None, methods=None, scale=True):
        """
        Apply dimensionality reduction using specified methods.

        Args:
            X (ndarray): Feature matrix
            y (ndarray): Target labels (kept for API compatibility, not used)
            methods (list): List of methods to apply (if None, use all)
            scale (bool): Whether to standardize features before reduction

        Returns:
            dict: Results for each method
        """
        # Scale the data
        if scale:
            X_scaled = StandardScaler().fit_transform(X)
        else:
            X_scaled = X

        # Get reducers
        all_reducers = self.get_reducers()

        # Filter methods if specified
        if methods is not None:
            all_reducers = {k: v for k, v in all_reducers.items() if k in methods}

        # Apply each method
        self.results = {}
        self.models = {}

        for name, reducer in all_reducers.items():
            try:
                X_transformed = reducer.fit_transform(X_scaled)

                self.results[name] = X_transformed
                self.models[name] = reducer

                # Print explained variance for PCA
                if hasattr(reducer, "explained_variance_ratio_"):
                    evr = reducer.explained_variance_ratio_
                    if len(evr) > 0:
                        total_ev = evr.sum()
                        top_ev = ", ".join(
                            [f"{v:.2%}" for v in evr[: min(2, len(evr))]]
                        )
                        print(f"{name}: Explained variance = {total_ev:.2%} ({top_ev})")
                    else:
                        print(f"{name}: Applied successfully")
                else:
                    print(f"{name}: Applied successfully")

            except Exception as e:
                print(f"{name} failed: {str(e)}")
                continue

        return self.results

    def plot_results(
        self,
        y,
        title_prefix="Dimensionality Reduction",
        figsize=(15, 12),
        n_cols=2,
        cmap="tab10",
        alpha=0.7,
    ):
        """
        Plot the results of all applied dimensionality reduction methods.

        Args:
            y (ndarray): Target labels for coloring
            title_prefix (str): Prefix for plot titles
            figsize (tuple): Figure size
            n_cols (int): Number of columns in the grid
            cmap (str): Colormap for scatter plot
            alpha (float): Transparency of points
        """
        n_methods = len(self.results)
        if n_methods == 0:
            print("No results to plot. Run fit_transform() first.")
            return

        n_cols = min(n_cols, n_methods)
        n_rows = (n_methods + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

        # Handle axes flattening properly for different subplot configurations
        if n_rows == 1 and n_cols == 1:
            axes_flat = [axes]
        elif n_rows == 1:
            axes_flat = axes
        else:
            axes_flat = axes.flatten()

        for i, (name, X_transformed) in enumerate(self.results.items()):
            ax = axes_flat[i]

            # Handle different component dimensions
            if X_transformed.shape[1] >= 2:
                scatter = ax.scatter(
                    X_transformed[:, 0],
                    X_transformed[:, 1],
                    c=y,
                    cmap=cmap,
                    alpha=alpha,
                    s=10,
                )
                ax.set_ylabel("Component 2")
            else:
                # If only 1 component, plot as 1D
                scatter = ax.scatter(
                    X_transformed[:, 0],
                    np.zeros_like(X_transformed[:, 0]),
                    c=y,
                    cmap=cmap,
                    alpha=alpha,
                    s=10,
                )
                ax.set_ylabel("")

            ax.set_title(f"{name}")
            ax.set_xlabel("Component 1")

            # Add colorbar only for the LAST subplot to avoid duplicates
            if i == len(self.results) - 1:
                cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("Class")

        # Remove empty subplots
        for j in range(len(self.results), len(axes_flat)):
            fig.delaxes(axes_flat[j])

        fig.suptitle(title_prefix, fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()

    def plot_2d_grid(
        self,
        y,
        title_prefix="Dimensionality Reduction",
        figsize=(15, 12),
        n_cols=2,
        cmap="tab10",
        alpha=0.7,
    ):
        """Alias for plot_results() for backward compatibility."""
        self.plot_results(y, title_prefix, figsize, n_cols, cmap, alpha)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def reduce_and_plot(
    X,
    y,
    methods=None,
    n_components=2,
    title="Dimensionality Reduction",
    figsize=(15, 12),
    n_cols=2,
    scale=True,
    random_state=42,
):
    """
    One-liner to apply dimensionality reduction and plot results.

    Args:
        X (ndarray): Feature matrix
        y (ndarray): Target labels
        methods (list): List of methods to apply (default: ['PCA', 't-SNE'])
        n_components (int): Number of components
        title (str): Title for the plot
        figsize (tuple): Figure size
        n_cols (int): Number of columns in grid
        scale (bool): Whether to standardize features
        random_state (int): Random seed

    Returns:
        DimensionalityReducer: The reducer object with results
    """
    if methods is None:
        methods = ["PCA", "t-SNE"]

    reducer = DimensionalityReducer(
        n_components=n_components, random_state=random_state
    )
    reducer.fit_transform(X, y, methods=methods, scale=scale)
    reducer.plot_results(y, title_prefix=title, figsize=figsize, n_cols=n_cols)
    return reducer


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def get_box_dist_plot(x):
    """This will give a box plot and a distplot one below the other"""
    f, (ax_box, ax_hist) = plt.subplots(
        2, sharex=True, gridspec_kw={"height_ratios": (0.15, 0.85)}
    )
    sns.boxplot(x, ax=ax_box)
    sns.histplot(x, kde=True)
    ax_box.set(yticks=[])
    sns.despine(ax=ax_hist)
    sns.despine(ax=ax_box, left=True)
    plt.show()


def find_outliers(class_images, n_outliers=5):
    if len(class_images) == 0:
        return np.array([]), np.array([])
    class_mean = class_images.mean(axis=0)
    distances = np.sqrt(((class_images - class_mean) ** 2).sum(axis=1))
    n_outliers = min(n_outliers, len(distances))
    outlier_indices = np.argsort(distances)[-n_outliers:]
    return outlier_indices, distances[outlier_indices]


def print_dataset_summary(
    X_bc,
    X_adult,
    X_cover,
    X_mnist,
    y_bc,
    y_adult,
    y_cover,
    y_mnist,
    df_bc,
    df_adult,
    df_cover,
    df_mnist,
):
    """
    Print a summary comparison of all loaded datasets.
    """
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    # Helper to get unique count from y
    def get_n_classes(y):
        if isinstance(y, pd.DataFrame):
            # If DataFrame, get first column
            return len(np.unique(y.iloc[:, 0]))
        elif isinstance(y, pd.Series):
            return len(np.unique(y))
        else:
            return len(np.unique(y))

    # Helper to get memory usage
    def get_memory_mb(df):
        return df.memory_usage(deep=True).sum() / 1024**2

    summary = {
        "Dataset": ["Breast Cancer", "Adult Income", "Covertype", "MNIST"],
        "Samples": [
            X_bc.shape[0],
            X_adult.shape[0],
            X_cover.shape[0],
            X_mnist.shape[0],
        ],
        "Features": [
            X_bc.shape[1],
            X_adult.shape[1],
            X_cover.shape[1],
            X_mnist.shape[1],
        ],
        "Classes": [
            get_n_classes(y_bc),
            get_n_classes(y_adult),
            get_n_classes(y_cover),
            get_n_classes(y_mnist),
        ],
        "Memory (MB)": [
            get_memory_mb(df_bc),
            get_memory_mb(df_adult),
            get_memory_mb(df_cover),
            get_memory_mb(df_mnist),
        ],
    }

    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))
    print("=" * 60)
