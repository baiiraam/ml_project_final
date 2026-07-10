import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from ucimlrepo import fetch_ucirepo


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
        if df[col].dtype == 'object':
            try:
                df.loc[:, col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                continue  # Skip if can't convert to numeric

        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        # Check current dtype
        current_dtype = df[col].dtype

        # Only attempt downcasting if there's a potential benefit
        if pd.api.types.is_integer_dtype(df[col]):
            # Check if we can downcast
            min_val = df[col].min()
            max_val = df[col].max()

            if min_val >= 0:
                if max_val <= 255 and current_dtype != np.uint8:
                    df.loc[:, col] = pd.to_numeric(df[col], downcast='integer')
                elif max_val <= 65535 and current_dtype != np.uint16:
                    df.loc[:, col] = pd.to_numeric(df[col], downcast='integer')
                elif max_val <= 4294967295 and current_dtype != np.uint32:
                    df.loc[:, col] = pd.to_numeric(df[col], downcast='integer')
            else:
                if min_val >= -128 and max_val <= 127 and current_dtype != np.int8:
                    df.loc[:, col] = pd.to_numeric(df[col], downcast='integer')
                elif min_val >= -32768 and max_val <= 32767 and current_dtype != np.int16:
                    df.loc[:, col] = pd.to_numeric(df[col], downcast='integer')
                elif min_val >= -2147483648 and max_val <= 2147483647 and current_dtype != np.int32:
                    df.loc[:, col] = pd.to_numeric(df[col], downcast='integer')
        elif pd.api.types.is_float_dtype(df[col]):
            # Only downcast if it's float64
            if current_dtype == np.float64:
                df.loc[:, col] = pd.to_numeric(df[col], downcast='float')

    mem_after = df.memory_usage(deep=True).sum() / 1024**2

    if verbose:
        reduction = (1 - mem_after/mem_before) * 100 if mem_before > 0 else 0
        if reduction > 0:
            print(f"  Memory: {mem_before:.4f} MB → {mem_after:.4f} MB "
                  f"({reduction:.1f}% reduction)")
        else:
            print(f"  Memory: {mem_before:.4f} MB (already optimized)")

    return df


def optimize_numpy_array(arr, dtype=None, verbose=True):
    """
    Optimize memory usage of a numpy array by downcasting.

    Args:
        arr (ndarray): Input numpy array
        dtype (type): Target dtype (if None, auto-detect)
        verbose (bool): If True, print memory usage before/after

    Returns:
        ndarray: Optimized array
    """
    mem_before = arr.nbytes / 1024**2

    if dtype is None:
        # Auto-detect optimal dtype
        if np.issubdtype(arr.dtype, np.integer):
            # Find minimum integer type that can hold all values
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
            dtype = np.float32  # float64 → float32 usually sufficient

    arr_optimized = arr.astype(dtype)
    mem_after = arr_optimized.nbytes / 1024**2

    if verbose:
        print(f"  Memory: {mem_before:.4f} MB → {mem_after:.4f} MB "
              f"({(1 - mem_after/mem_before) * 100:.4f}% reduction)")

    return arr_optimized


def optimize_series_memory(series, verbose=True):
    """
    Optimize memory usage of a pandas Series.

    Args:
        series (Series): Input Series
        verbose (bool): If True, print memory usage before/after

    Returns:
        Series: Optimized Series
    """
    mem_before = series.memory_usage(deep=True) / 1024**2

    if pd.api.types.is_numeric_dtype(series):
        if pd.api.types.is_integer_dtype(series):
            series = pd.to_numeric(series, downcast='integer')
        elif pd.api.types.is_float_dtype(series):
            series = pd.to_numeric(series, downcast='float')

    mem_after = series.memory_usage(deep=True) / 1024**2

    if verbose:
        print(f"  Memory: {mem_before:.4f} MB → {mem_after:.4f} MB "
              f"({(1 - mem_after/mem_before) * 100:.4f}% reduction)")

    return series


def load_breast_cancer_data(optimize_memory=True, verbose=True):
    """
    Load the Breast Cancer Wisconsin dataset.

    Args:
        optimize_memory (bool): If True, optimize memory usage
        verbose (bool): If True, print dataset info and memory stats

    Returns:
        X_bc (DataFrame): Features (30 columns)
        y_bc (Series): Target (0 = malignant, 1 = benign)
        df_bc (DataFrame): Combined features and target
    """
    X_bc, y_bc = load_breast_cancer(return_X_y=True, as_frame=True)

    if optimize_memory:
        X_bc = optimize_dataframe_memory(X_bc, verbose=verbose)
        y_bc = optimize_series_memory(y_bc, verbose=verbose)

    df_bc = pd.concat([X_bc, y_bc], axis=1)

    return X_bc, y_bc, df_bc


def load_adult_income_data(drop_categorical=True, optimize_memory=True, verbose=True):
    """
    Load the Adult Income dataset from UCI.

    Args:
        drop_categorical (bool): If True, drops categorical columns,
                                 keeping only numeric features.
        optimize_memory (bool): If True, optimize memory usage
        verbose (bool): If True, print dataset info and memory stats

    Returns:
        X_adult (DataFrame): Features
        y_adult (DataFrame): Target (income)
        df_adult (DataFrame): Combined features and target
    """
    adult = fetch_ucirepo(id=2)
    X_adult = adult.data.features
    y_adult = adult.data.targets

    # Clean target (remove dots from labels)
    y_adult.loc[:, 'income'] = y_adult['income'].str.replace('.', '', regex=False)

    if drop_categorical:
        cols_to_drop = [
            'workclass', 'education', 'marital-status',
            'occupation', 'relationship', 'race', 'sex',
            'native-country'
        ]
        X_adult = X_adult.drop(columns=cols_to_drop)

    if optimize_memory:
        X_adult = optimize_dataframe_memory(X_adult, verbose=verbose)
        # y_adult is categorical, skip optimization

    df_adult = pd.concat([X_adult, y_adult], axis=1)

    return X_adult, y_adult, df_adult


def load_covertype_data(drop_categorical=True, optimize_memory=True, verbose=True):
    """
    Load the Covertype dataset from UCI.

    Args:
        drop_categorical (bool): If True, drops Wilderness_Area and Soil_Type
                                 categorical indicator columns.
        optimize_memory (bool): If True, optimize memory usage
        verbose (bool): If True, print dataset info and memory stats

    Returns:
        X_cover (DataFrame): Features
        y_cover (DataFrame): Target (Cover_Type: 1-7)
        df_cover (DataFrame): Combined features and target
    """
    covertype = fetch_ucirepo(id=31)
    X_cover = covertype.data.features
    y_cover = covertype.data.targets

    if drop_categorical:
        X_cover = X_cover[X_cover.columns.drop(
            list(X_cover.filter(regex='Wilderness_Area'))
        )]
        X_cover = X_cover[X_cover.columns.drop(
            list(X_cover.filter(regex='Soil_Type'))
        )]

    if optimize_memory:
        X_cover = optimize_dataframe_memory(X_cover, verbose=verbose)
        y_cover = optimize_dataframe_memory(y_cover, verbose=verbose)

    df_cover = pd.concat([X_cover, y_cover], axis=1)

    return X_cover, y_cover, df_cover


def load_mnist_data(optimize_memory=True, verbose=True, return_numpy=False):
    """
    Load the MNIST dataset with multiple fallback options.
    """
    X, y = None, None

    # Try 4: Local file
    if X is None:
        try:
            import pickle
            with open('mnist_data.pkl', 'rb') as f:
                X, y = pickle.load(f)
            if verbose:
                print("Loaded MNIST from local file")
        except FileNotFoundError:
            pass
        except ModuleNotFoundError:
            pass

    # Try 1: OpenML (most reliable)
    if X is None:
        try:
            from sklearn.datasets import fetch_openml
            X, y = fetch_openml(
                'mnist_784',
                version=1,
                return_X_y=True,
                as_frame=False,
                parser='pandas',
                data_home='./mnist_cache',
                n_retries=5
            )
            y = y.astype(int)
            if verbose:
                print("Loaded MNIST from OpenML")
        except Exception as e:
            if verbose:
                print(f"OpenML failed: {e}")

    # If all fail, raise error
    if X is None:
        raise ImportError(
            "Could not load MNIST dataset. Please install one of:\n"
            "  pip install tensorflow\n"
            "  pip install torch torchvision\n"
            "  pip install scikit-learn\n"
            "Or download manually from: https://www.openml.org/d/554"
        )

    # Optimize memory if requested
    if optimize_memory:
        X = optimize_numpy_array(X, verbose=verbose)
        y = optimize_numpy_array(y, verbose=verbose)

    # Create DataFrame
    df_mnist = pd.DataFrame(
        X,
        columns=[f'pixel_{i}' for i in range(X.shape[1])]
    )
    df_mnist['label'] = y

    if not return_numpy:
        X_mnist = df_mnist.drop('label', axis=1)
        y_mnist = df_mnist['label']
    else:
        X_mnist = X
        y_mnist = y

    return X_mnist, y_mnist, df_mnist
