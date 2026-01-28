from __future__ import annotations

import numpy as np
import pandas as pd


class WindowMetadata:
    """
    Simple container to track the origin of a window.
    """

    def __init__(self, subject_id: str, start_index: int) -> None:
        self.subject_id = subject_id
        self.start_index = start_index

    def __repr__(self) -> str:
        return f"Meta(id={self.subject_id}, start={self.start_index})"

def build_conditional_windows(
        dfs: list[pd.DataFrame],
        seq_len: int,
        step: int,
        target_col: str,
        cond_cols: list[str],
        max_missing_ratio: float = 0.0,
        allow_target_nan: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[WindowMetadata]]:
    """
    Function specifically to extract windows from list of dfs.

    Args:
        dfs: List of DataFrames.
        seq_len: Window length.
        step: Stride.
        target_col: The column to predict (y).
        cond_cols: The columns to condition on (c).
        max_missing_ratio: Max NaN allowance.
        allow_target_nan: whether to allow target NaN.

    Returns:
        y_windows: (N, seq_len, 1)
        c_windows: (N, seq_len, n_cond_cols)
        metadata: List of metadata.
    """

    all_y = []
    all_c = []
    metadata = []

    required_cols = cond_cols + [target_col]

    for df in dfs:
        subj_id = str(df.attrs.get("unique_id", "unknown"))

        # Check columns
        if not set(required_cols).issubset(df.columns):
            continue

        # Numpy Extraction
        c_data = df[cond_cols].to_numpy(dtype=np.float32)
        y_data = df[[target_col]].to_numpy(dtype=np.float32)

        n_rows = len(df)

        for start_idx in range(0, n_rows - seq_len + 1, step):
            end_idx = start_idx + seq_len

            y_window = y_data[start_idx:end_idx]
            c_window = c_data[start_idx:end_idx]

            # Check: target can't be NaN
            if np.isnan(y_window).any() and not allow_target_nan:
                continue

            # Discard window if NaN values above threshold
            if np.isnan(c_window).mean() > max_missing_ratio:
                continue

            all_y.append(y_window)
            all_c.append(c_window)
            metadata.append(WindowMetadata(subject_id=subj_id, start_index=start_idx))

    if len(all_y) == 0:
        # Empty case
        return (
            np.empty((0, seq_len, 1), dtype=np.float32),
            np.empty((0, seq_len, len(cond_cols)), dtype=np.float32),
            []
        )

    y_windows = np.stack(all_y, axis=0)
    c_windows = np.stack(all_c, axis=0)

    return y_windows, c_windows, metadata


def extract_full_sequences(
        dfs: list[pd.DataFrame],
        target_col: str,
        cond_cols: list[str],
        allow_target_nan: bool = True
) -> tuple[list[np.ndarray], list[np.ndarray], list[WindowMetadata]]:
    """
    Extracts the entire time series from a list of DataFrames without slicing.

    Unlike 'build_conditional_windows', this function does not apply sliding windows
    or padding. It preserves the original length of each DataFrame.
    Consequently, it returns lists of numpy arrays (ragged arrays) instead of
    a single stacked numpy array, as the lengths may vary between subjects.

    Args:
        dfs: List of source DataFrames (usually normalized).
        target_col: The name of the target column (y).
        cond_cols: List of names of the conditional columns (c).
        allow_target_nan: If True, fills the target array with NaNs if the
                          target column is missing from the DataFrame (useful for
                          pure inference/generation scenarios).

    Returns:
        A tuple containing three lists:
        1. List of Target Arrays [(L1, 1), (L2, 1), ...]
        2. List of Conditional Arrays [(L1, C), (L2, C), ...]
        3. List of Metadata objects (one per subject).
    """

    all_y = []
    all_c = []
    metadata = []

    # 1. Iterate over each subject DataFrame
    for df in dfs:
        # Retrieve unique identifier (default to 'unknown' if missing)
        subj_id = str(df.attrs.get("unique_id", "unknown"))
        n_rows = len(df)

        # 2. Extract Target (y)
        if target_col in df.columns:
            # Extract existing target as float32
            y_curr = df[[target_col]].to_numpy(dtype=np.float32)
        else:
            # Handle missing target
            if not allow_target_nan:
                print(f"[Utils] Skipping {subj_id}: Target column '{target_col}' missing.")
                continue

            # Create a placeholder array full of NaNs
            y_curr = np.full((n_rows, 1), np.nan, dtype=np.float32)

        # 3. Extract Conditions (c)
        # Verify that all required conditional columns are present
        if not set(cond_cols).issubset(df.columns):
            missing = list(set(cond_cols) - set(df.columns))
            print(f"[Utils] Skipping {subj_id}: Missing conditional columns {missing}")
            continue

        c_curr = df[cond_cols].to_numpy(dtype=np.float32)

        # 4. Consistency Check
        # Ensure target and condition arrays have the same length (row count)
        if len(y_curr) != len(c_curr):
            print(f"[Utils] Skipping {subj_id}: Length mismatch (y={len(y_curr)}, c={len(c_curr)})")
            continue

        # 5. Append to lists
        all_y.append(y_curr)
        all_c.append(c_curr)

        # Start index is always 0 for full sequences
        metadata.append(WindowMetadata(subject_id=subj_id, start_index=0))

    return all_y, all_c, metadata