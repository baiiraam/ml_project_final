import os
import pandas as pd
import numpy as np
from typing import Optional
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo


def get_project_root():
    """
    Get the project root directory using __file__.
    This works reliably regardless of where the script is run from.

    Returns:
        str: Absolute path to the project root directory.
    """
    current = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    if os.path.exists(os.path.join(current, "src")):
        return current

    return os.getcwd()


PROJECT_ROOT = get_project_root()


def standardize(X_train: np.ndarray, X_test: np.ndarray) -> tuple:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
) -> tuple:
    rng = np.random.RandomState(random_state)
    n = X.shape[0]
    indices = np.arange(n)
    rng.shuffle(indices)
    split = int(n * (1 - test_size))
    train_idx = indices[:split]
    test_idx = indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def optimize_dataframe_memory(df, exclude_cols=None, verbose=True):
    """
    Optimize memory usage of a DataFrame by downcasting numeric columns.
    """
    if exclude_cols is None:
        exclude_cols = []

    mem_before = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:
        if col in exclude_cols:
            continue

        # Try to convert object columns to numeric if possible
        if df[col].dtype == "object":
            try:
                df.loc[:, col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                continue

        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        current_dtype = df[col].dtype

        if pd.api.types.is_integer_dtype(df[col]):
            min_val = df[col].min()
            max_val = df[col].max()

            if min_val >= 0:
                if max_val <= 255 and current_dtype != np.uint8:
                    df.loc[:, col] = pd.to_numeric(df[col], downcast="integer")
                elif max_val <= 65535 and current_dtype != np.uint16:
                    df.loc[:, col] = pd.to_numeric(df[col], downcast="integer")
                elif max_val <= 4294967295 and current_dtype != np.uint32:
                    df.loc[:, col] = pd.to_numeric(df[col], downcast="integer")
            else:
                if min_val >= -128 and max_val <= 127 and current_dtype != np.int8:
                    df.loc[:, col] = pd.to_numeric(df[col], downcast="integer")
                elif (
                    min_val >= -32768 and max_val <= 32767 and current_dtype != np.int16
                ):
                    df.loc[:, col] = pd.to_numeric(df[col], downcast="integer")
                elif (
                    min_val >= -2147483648
                    and max_val <= 2147483647
                    and current_dtype != np.int32
                ):
                    df.loc[:, col] = pd.to_numeric(df[col], downcast="integer")
        elif pd.api.types.is_float_dtype(df[col]):
            if current_dtype == np.float64:
                df.loc[:, col] = pd.to_numeric(df[col], downcast="float")

    mem_after = df.memory_usage(deep=True).sum() / 1024**2

    if verbose:
        reduction = (1 - mem_after / mem_before) * 100 if mem_before > 0 else 0
        if reduction > 0:
            print(
                f"  Memory: {mem_before:.4f} MB → {mem_after:.4f} MB "
                f"({reduction:.1f}% reduction)"
            )
        else:
            print(f"  Memory: {mem_before:.4f} MB (already optimized)")

    return df


def optimize_numpy_array(arr, dtype=None, verbose=True):
    """
    Optimize memory usage of a numpy array by downcasting.
    """
    mem_before = arr.nbytes / 1024**2

    if dtype is None:
        if np.issubdtype(arr.dtype, np.integer):
            min_val = arr.min()
            max_val = arr.max()
            if min_val >= 0:
                if max_val <= 255:
                    dtype = np.uint8
                elif max_val <= 65535:
                    dtype = np.uint16
                elif max_val <= 4294967295:
                    dtype = np.uint32
                else:
                    dtype = np.uint64
            else:
                if min_val >= -128 and max_val <= 127:
                    dtype = np.int8
                elif min_val >= -32768 and max_val <= 32767:
                    dtype = np.int16
                elif min_val >= -2147483648 and max_val <= 2147483647:
                    dtype = np.int32
                else:
                    dtype = np.int64
        elif np.issubdtype(arr.dtype, np.floating):
            dtype = np.float32

    arr_optimized = arr.astype(dtype)
    mem_after = arr_optimized.nbytes / 1024**2

    if verbose:
        print(
            f"  Memory: {mem_before:.4f} MB → {mem_after:.4f} MB "
            f"({(1 - mem_after / mem_before) * 100:.4f}% reduction)"
        )

    return arr_optimized


def optimize_series_memory(series, verbose=True):
    """
    Optimize memory usage of a pandas Series.
    """
    mem_before = series.memory_usage(deep=True) / 1024**2

    if pd.api.types.is_numeric_dtype(series):
        if pd.api.types.is_integer_dtype(series):
            series = pd.to_numeric(series, downcast="integer")
        elif pd.api.types.is_float_dtype(series):
            series = pd.to_numeric(series, downcast="float")

    mem_after = series.memory_usage(deep=True) / 1024**2

    if verbose:
        print(
            f"  Memory: {mem_before:.4f} MB → {mem_after:.4f} MB "
            f"({(1 - mem_after / mem_before) * 100:.4f}% reduction)"
        )

    return series


def _load_parquet_or_fetch(
    parquet_filename, fetch_func, optimize_memory=True, verbose=True
):
    """Helper function: try to load from CSV, otherwise fetch and save as CSV."""
    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Use CSV instead of Parquet
    csv_filename = parquet_filename.replace(".parquet", ".csv")
    csv_path = os.path.join(data_dir, csv_filename)

    if os.path.exists(csv_path):
        if verbose:
            print(f"Loaded from CSV: {csv_path}")
        df = pd.read_csv(csv_path, index_col=0)
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]
        return X, y

    if verbose:
        print(f"{csv_path} not found locally. Downloading...")

    X, y = fetch_func()

    if optimize_memory:
        if verbose:
            print("Optimizing memory before saving...")
        X = optimize_dataframe_memory(X, verbose=verbose)
        y = optimize_series_memory(y, verbose=verbose)

    df = pd.concat([X, y], axis=1)

    if verbose:
        print(f"Saving to {csv_path}...")
    df.to_csv(csv_path, index=True)

    if verbose:
        print(f"Saved to {csv_path}")

    return X, y


def load_breast_cancer_data(optimize_memory=True, verbose=True):
    """
    Load the Breast Cancer Wisconsin dataset.
    First tries to load from data/breast_cancer.parquet, otherwise downloads.
    """

    def _fetch_breast_cancer():
        X, y = load_breast_cancer(return_X_y=True, as_frame=True)
        return X, y

    X_bc, y_bc = _load_parquet_or_fetch(
        "breast_cancer.parquet",
        _fetch_breast_cancer,
        optimize_memory=optimize_memory,
        verbose=verbose,
    )

    df_bc = pd.concat([X_bc, y_bc], axis=1)
    return X_bc, y_bc, df_bc


def load_adult_income_data(drop_categorical=True, optimize_memory=True, verbose=True):
    """
    Load the Adult Income dataset from UCI.
    First tries to load from data/adult_income.parquet, otherwise downloads.
    """

    def _fetch_adult_income():
        adult = fetch_ucirepo(id=2)
        X = adult.data.features
        y = adult.data.targets

        # Ensure y is a Series (not a DataFrame)
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]  # Extract the first (and only) column as Series

        # Remove dots from income values (e.g., '<=50K.' -> '<=50K')
        y = y.str.replace(".", "", regex=False)

        if drop_categorical:
            cols_to_drop = [
                "workclass",
                "education",
                "marital-status",
                "occupation",
                "relationship",
                "race",
                "sex",
                "native-country",
            ]
            X = X.drop(columns=cols_to_drop)

        return X, y

    X_adult, y_adult = _load_parquet_or_fetch(
        "adult_income.parquet",
        _fetch_adult_income,
        optimize_memory=optimize_memory,
        verbose=verbose,
    )

    df_adult = pd.concat([X_adult, y_adult], axis=1)
    return X_adult, y_adult, df_adult


def load_covertype_data(drop_categorical=True, optimize_memory=True, verbose=True):
    """
    Load the Covertype dataset from UCI.
    First tries to load from data/covertype.parquet, otherwise downloads.
    """

    def _fetch_covertype():
        covertype = fetch_ucirepo(id=31)
        X = covertype.data.features
        y = covertype.data.targets

        # Ensure y is a Series (not a DataFrame)
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]  # Extract the first (and only) column as Series

        if drop_categorical:
            X = X[X.columns.drop(list(X.filter(regex="Wilderness_Area")))]
            X = X[X.columns.drop(list(X.filter(regex="Soil_Type")))]

        return X, y

    X_cover, y_cover = _load_parquet_or_fetch(
        "covertype.parquet",
        _fetch_covertype,
        optimize_memory=optimize_memory,
        verbose=verbose,
    )

    df_cover = pd.concat([X_cover, y_cover], axis=1)
    return X_cover, y_cover, df_cover


def load_mnist_data(optimize_memory=True, verbose=True, return_numpy=False):
    """
    Load the MNIST dataset.
    First tries to load from data/mnist.parquet, otherwise downloads from OpenML.
    """

    def _fetch_mnist():
        from sklearn.datasets import fetch_openml

        X, y = fetch_openml(
            "mnist_784",
            version=1,
            return_X_y=True,
            as_frame=False,
            parser="pandas",
            data_home="./mnist_cache",
            n_retries=5,
        )
        # Ensure y is a Series with proper dtype
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        y = y.astype(int)

        df_temp = pd.DataFrame(X, columns=[f"pixel_{i}" for i in range(X.shape[1])])
        df_temp["label"] = y

        X_df = df_temp.drop("label", axis=1)
        y_series = df_temp["label"]

        return X_df, y_series

    X_mnist, y_mnist = _load_parquet_or_fetch(
        "mnist.parquet", _fetch_mnist, optimize_memory=optimize_memory, verbose=verbose
    )

    df_mnist = pd.concat([X_mnist, y_mnist], axis=1)

    if return_numpy:
        X_mnist = X_mnist.values
        y_mnist = y_mnist.values

    return X_mnist, y_mnist, df_mnist
