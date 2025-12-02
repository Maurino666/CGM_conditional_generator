from typing import Tuple, List

import pandas as pd
import numpy as np

from scipy.spatial.distance import pdist, squareform

try:
    from sklearn.feature_selection import mutual_info_regression
    _USE_KNN = True
except ImportError:
    _USE_KNN = False

def get_valid_segments(s: pd.Series, min_length: int = 5) -> list[pd.Series]:
    """
    Identify continuous segments in a pandas Series that do not contain NaN values.
    :param s: Input pandas Series.
    :param min_length: Minimum length of the series to be considered valid.
    :return: List of pandas Series, each representing a valid continuous segment.
    """
    if len(s) < min_length:
        return []

    s_copy = s.copy()
    mask = s_copy.notna()
    # Identifica i punti di cambio di True/False
    groups = mask.ne(mask.shift(fill_value=False)).cumsum()

    segments = []
    for _, seg in s_copy.groupby(groups):
        # Aggiungi solo i segmenti che non hanno NaN e soddisfano la lunghezza minima
        if seg.notna().all() and len(seg) >= min_length:
            segments.append(seg)
    return segments

def align_data(target: pd.Series, df: pd.DataFrame, df_cols: list, freq: pd.Timedelta) -> Tuple[
    pd.Series, pd.DataFrame, pd.DatetimeIndex]:
    """
    Aligns the target series and event dataframe to a common time grid with specified frequency.
    Handles missing data appropriately for correlation and mutual information calculations.
    If the input series are already aligned and on the desired grid, they are returned as is.
    :param target: target time series (pandas Series with DateTimeIndex)
    :param df: dataframe containing event series (pandas DataFrame with DateTimeIndex)
    :param df_cols: list of columns in df to consider as event series
    :param freq: desired frequency for alignment
    :return: tuple containing:
        - G: aligned target series (pandas Series)
        - E_all: aligned event dataframe (pandas DataFrame)
        - grid: the common time grid (pandas DateTimeIndex)
    """
    target = target.copy()
    df = df.copy()

    for df_series in [target, df]:
        if not isinstance(df_series.index, pd.DatetimeIndex):
            df_series.index = pd.to_datetime(df_series.index)

    is_aligned_and_on_grid = (
            target.index.equals(df.index) and
            target.index.freq == freq
    )

    if is_aligned_and_on_grid:
        grid = target.index
        G = pd.to_numeric(target, errors="coerce")
        E_all = df[df_cols].apply(pd.to_numeric, errors="coerce")

    else:

        tmin = max(target.index.min(), df.index.min())
        tmax = min(target.index.max(), df.index.max())

        if pd.isna(tmin) or pd.isna(tmax) or (tmin >= tmax):
            raise ValueError("No temporal overlap between target and events.")

        grid = pd.date_range(tmin, tmax, freq=freq)

        G = pd.to_numeric(target, errors="coerce").reindex(grid)

        E_all = df[df_cols].apply(pd.to_numeric, errors="coerce") \
            .reindex(grid)

    E_all = E_all.fillna(0.0)

    return G, E_all, grid

def standardize_series(series: pd.Series) -> pd.Series:
    """
    Applies z-score standardization to a pandas Series, handling NaNs appropriately.
    :param series: input pandas Series
    :return: standardized pandas Series
    """
    mean = series.mean(skipna=True)
    std = series.std(ddof=1, skipna=True)

    if pd.isna(std) or std == 0:
        # Degenerate case: constant or insufficient data. Just center (no scaling).
        return series - mean
    else:
        # Standardize (z-score)
        return (series - mean) / std

def compute_lags(max_lag: pd.Timedelta, freq: pd.Timedelta) -> Tuple[np.ndarray, pd.TimedeltaIndex]:
    """
    Computes integer lag steps and corresponding TimedeltaIndex for given maximum lag and frequency.
    :param max_lag: maximum lag (positive) as Timedelta
    :param freq: frequency of the time series as Timedelta
    :return: tuple containing:
        - lag_steps: numpy array of integer lag steps in [-L, +L]
        - lags: pandas TimedeltaIndex corresponding to lag_steps
    """
    # Number of integer lag steps in [-L, +L]
    L = int(max_lag / freq)
    lag_steps = np.arange(-L, L + 1, dtype=int)
    lags = lag_steps * freq  # TimedeltaIndex for plotting/indexing
    return lag_steps, lags

def build_lagged_view(df, target_col, target_lags, exog_lags, dropna = True):
    """
    Builds a lagged view of the DataFrame.
    :param df: dataframe to be lagged
    :param target_col: target column name
    :param target_lags: list of lags for the target
    :param exog_lags: dictionary of exogenous variable lags {feature_name: [lags]}
    :param dropna: whether to drop rows with NaN after lagging
    :return:
    """
    df_lagged = df.copy()

    for k in target_lags:
        df_lagged[f"{target_col}_lag{k}"] = df_lagged[target_col].shift(k)

    for f, lags in exog_lags.items():
        for k in lags:
            df_lagged[f"{f}_lag{k}"] = df_lagged[f].shift(k)

    if dropna:
        df_lagged = df_lagged.dropna()

    return df_lagged

def build_future_targets(df: pd.DataFrame, target_col: str, horizons: List[int]) -> pd.DataFrame:
    Y = pd.DataFrame(index=df.index)
    for h in horizons:
        Y[f"{target_col}_lead{h}"] = df[target_col].shift(-h)
    return Y

def distance_correlation(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute the distance correlation between two (possibly multivariate) arrays.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features_X)
    Y : array-like, shape (n_samples, n_features_Y)

    Returns
    -------
    dcor : float in [0, 1]
        0 means statistical independence; larger values indicate stronger dependence.
    """
    # Ensure 2D arrays
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)

    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of rows (samples).")

    # If any of the inputs has no variance, dCor is 0 by definition here
    if np.allclose(X, X.mean(axis=0)) or np.allclose(Y, Y.mean(axis=0)):
        return 0.0

    # Pairwise Euclidean distances
    a = squareform(pdist(X, metric="euclidean"))
    b = squareform(pdist(Y, metric="euclidean"))

    # Double-centering
    A = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
    B = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()

    # Distance covariance and variances
    dcov2_xy = (A * B).mean()
    dcov2_xx = (A * A).mean()
    dcov2_yy = (B * B).mean()

    # Guard against zero division
    denom = np.sqrt(dcov2_xx * dcov2_yy)
    if denom <= 0:
        return 0.0

    return np.sqrt(dcov2_xy / denom)