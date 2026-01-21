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


def build_raw_windows(
        dfs: list[pd.DataFrame],
        cols: list[str],
        seq_len: int,
        step: int,
        max_missing_ratio: float = 0.05
) -> tuple[np.ndarray, list[WindowMetadata]]:
    """
    Base function: extracts generic sliding windows for a specific set of columns.

    This function does NOT split into target/conditions. It returns a single
    3D tensor containing all requested columns.

    Args:
        dfs: List of DataFrames (source data).
        cols: List of column names to include in the window.
        seq_len: Length of the window (timesteps).
        step: Stride between windows.
        max_missing_ratio: Max allowed fraction of NaNs in a window (0.0 to 1.0).

    Returns:
        windows: Numpy array of shape (N_windows, seq_len, N_columns).
        metadata: List of metadata objects tracking subject and time.
    """
    all_windows = []
    metadata = []

    for df in dfs:
        # 1. Retrieve Subject ID safely from attributes
        # (Injected by the DataSplitter step)
        subj_id = str(df.attrs.get("subject_id", "unknown"))

        # 2. Check if columns exist
        if not set(cols).issubset(df.columns):
            missing = set(cols) - set(df.columns)
            print(f"   [!] Error: Subject {subj_id} is missing columns: {missing}. Skipping.")
            continue

        # 3. Extract raw numpy array for the requested columns only
        # shape: (n_rows, n_columns)
        data_matrix = df[cols].to_numpy(dtype=np.float32)
        n_rows = len(df)

        # 4. Sliding Window Loop
        # Range: ensure the last window fits completely
        for start_idx in range(0, n_rows - seq_len + 1, step):
            end_idx = start_idx + seq_len

            # Slice the window
            window_slice = data_matrix[start_idx:end_idx]

            # Check for Missing Values (NaNs)
            # Fast check using numpy
            if np.isnan(window_slice).mean() > max_missing_ratio:
                continue

            all_windows.append(window_slice)
            metadata.append(WindowMetadata(subject_id=subj_id, start_index=start_idx))

    # 5. Stack results
    if len(all_windows) == 0:
        print("   [!] Warning: No windows generated. Check constraints.")
        return np.empty((0, seq_len, len(cols)), dtype=np.float32), []

    # Final Shape: (Total_Windows, Seq_Len, Num_Columns)
    final_tensor = np.stack(all_windows, axis=0)

    return final_tensor, metadata


def build_conditional_windows(
        dfs: list[pd.DataFrame],
        seq_len: int,
        step: int,
        target_col: str,
        cond_cols: list[str],
        max_missing_ratio: float = 0.05
) -> tuple[np.ndarray, np.ndarray, list[WindowMetadata]]:
    """
    Wrapper function specifically for Conditional TimeGAN.

    It uses 'build_raw_windows' to extract data, then splits the tensor
    into Target (y) and Conditions (c).

    Args:
        dfs: List of DataFrames.
        seq_len: Window length.
        step: Stride.
        target_col: The column to predict (y).
        cond_cols: The columns to condition on (c).
        max_missing_ratio: Max NaN allowance.

    Returns:
        y_windows: (N, seq_len, 1)
        c_windows: (N, seq_len, n_cond_cols)
        metadata: List of metadata.
    """
    # 1. Define the full list of columns to extract (Order matters!)
    # We put target first, then conditions.
    all_cols = [target_col] + cond_cols

    # 2. Call the base function
    raw_tensor, metadata = build_raw_windows(
        dfs=dfs,
        cols=all_cols,
        seq_len=seq_len,
        step=step,
        max_missing_ratio=max_missing_ratio
    )

    if len(raw_tensor) == 0:
        # Return empty arrays with correct feature dimensions
        return (
            np.empty((0, seq_len, 1), dtype=np.float32),
            np.empty((0, seq_len, len(cond_cols)), dtype=np.float32),
            []
        )

    # 3. Split the tensor back into components
    # The first column (index 0) is the target because of how we built 'all_columns'
    y_windows = raw_tensor[:, :, :1]  # Shape: (N, seq_len, 1)
    c_windows = raw_tensor[:, :, 1:]  # Shape: (N, seq_len, n_cond_cols)

    return y_windows, c_windows, metadata