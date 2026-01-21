from typing import Sequence

import numpy as np
import pandas as pd

WindowMetadata = tuple[int, int]

def build_sliding_windows(
    all_data: list[pd.DataFrame],
    feature_cols: list[str],
    seq_len: int,
    step: int,
    ids: Sequence[int] | None = None,
    max_missing_ratio: float = 0.0,
) -> tuple[np.ndarray, list[WindowMetadata]]:
    """
    Build contiguous sliding windows from a list of patient DataFrames.

    Each DataFrame is assumed to contain a time series for a single subject,
    already cleaned and aligned on a regular time grid (e.g. every 5 minutes).

    For each subject, the function extracts windows of length `seq_len` by
    moving a sliding window with stride `step` over the rows. Windows that
    contain too many missing values (above `max_missing_ratio`) are discarded.

    Parameters
    ----------
    all_data : list of pd.DataFrame
        List of patient DataFrames.
    feature_cols : sequence of str
        Column names to use as features in the windows.
    seq_len : int
        Length of each window (number of time steps).
    step : int
        Sliding window step (number of rows between the start of consecutive windows).
    ids : sequence of int, optional
        Sequence of subject IDs for metadata.
    max_missing_ratio : float, optional
        Maximum allowed fraction of NaN values inside a window (between 0 and 1).
        Windows with a higher missing ratio are discarded. Default is 0.0 (no NaNs allowed).

    Returns
    -------
    X : np.ndarray
        3D array of shape (num_windows, seq_len, num_features), dtype float32.
    metadata : list[WindowMetadata]
        List of WindowMetadata objects (tuple(id, start)), one for each window.
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if step <= 0:
        raise ValueError("step must be positive")
    if not 0.0 <= max_missing_ratio <= 1.0:
        raise ValueError("max_missing_ratio must be between 0.0 and 1.0")

    windows: list[np.ndarray] = []
    metadata: list[WindowMetadata] = []

    for df_idx, df in enumerate(all_data):
        if df.empty:
            continue

        if df.index is not None:
            df = df.sort_index()

        # Check that all required feature columns are present
        missing_features = [c for c in feature_cols if c not in df.columns]
        if missing_features:
            raise KeyError(
                f"DataFrame {df_idx} is missing required feature columns: {missing_features}"
            )

        # Restrict to feature columns
        sub = df[list(feature_cols)].copy()
        num_rows = len(sub)

        if num_rows < seq_len:
            # Not enough data points for a single window
            continue

        # Slide over the DataFrame rows
        start = 0
        while start + seq_len <= num_rows:
            window = sub.iloc[start : start + seq_len]

            # Compute missing ratio across all features and time steps
            missing_ratio = window.isna().mean().mean()

            if missing_ratio <= max_missing_ratio:
                # Convert to numpy array, cast to float32 for efficiency
                values = window.to_numpy(dtype=np.float32)
                # values shape: (seq_len, num_features)
                windows.append(values)
                if ids is not None:
                    global_id = int(ids[df_idx])
                else:
                    global_id = int(df_idx)

                metadata.append((global_id, start))

            start += step

    if not windows:
        # No valid windows were found
        return np.empty((0, seq_len, len(feature_cols)), dtype=np.float32), []

    # Stack all windows into a single 3D array
    X = np.stack(windows, axis=0)  # shape: (num_windows, seq_len, num_features)
    return X, metadata

def build_sliding_windows_conditional(
    all_data: list[pd.DataFrame],
    seq_len: int,
    step: int,
    target_col: str,
    cond_cols: list[str],
    ids: Sequence[int] | None = None,
    max_missing_ratio: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, list[WindowMetadata]]:
    """
    Build sliding windows and split them into target and conditioning parts.

    Parameters
    ----------
    all_data : list[pd.DataFrame]
        Sequence of subject DataFrames.
    seq_len : int
        Window length (number of time steps).
    step : int
        Stride between consecutive windows.
    target_col : str
        Name of the target column (first feature in the windows).
    cond_cols : list[str]
        Names of conditioning columns to include in the windows.
    ids : sequence of int
        Sequence of subject IDs for metadata.
    max_missing_ratio: float, optional
        Max ratio of missing values in window.

    Returns
    -------
    X_target : np.ndarray
        Target windows of shape (num_windows, seq_len, 1).
    X_cond : np.ndarray
        Conditioning windows of shape (num_windows, seq_len, num_cond_features).
    metadata : list[WindowMetadata]
        List of WindowMetadata objects (tuple(id, start)), one for each window.

    """
    # Ensure target is not duplicated in cond_cols
    cond_cols_clean = [c for c in cond_cols if c != target_col]

    # Complete feature list: target first, then conditioning
    all_cols: list[str] = [target_col] + cond_cols_clean

    # Sanity check: required columns must exist in all DataFrames
    for i, df in enumerate(all_data):
        missing = [c for c in all_cols if c not in df.columns]
        if missing:
            raise KeyError(
                f"DataFrame {i} is missing required columns for sliding windows: {missing}"
            )

    # Use the existing generic sliding-window builder
    X_full, metadata = build_sliding_windows(
        all_data = all_data,
        seq_len=seq_len,
        step=step,
        feature_cols=all_cols,
        ids=ids,
        max_missing_ratio=max_missing_ratio,
    )

    # X_full shape: (num_windows, seq_len, 1 + len(cond_cols_clean))

    # Split into target (first feature) and conditioning (remaining features)
    X_target = X_full[:, :, :1]        # (N, seq_len, 1)
    X_cond = X_full[:, :, 1:]          # (N, seq_len, num_cond_features)

    return X_target, X_cond, metadata

