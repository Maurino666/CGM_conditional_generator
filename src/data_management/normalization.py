from __future__ import annotations

import pandas as pd
import numpy as np


class MinMaxNormalizer:
    """
    Stateful Normalizer for lists of Pandas DataFrames.

    It follows the standard Fit/Transform pattern to ensure no data leakage:
    1. .fit(train_dfs): Learns min/max statistics from TRAINING data only.
    2. .transform(dfs): Applies scaling to any list of DataFrames (Train, Val, or Test).

    Features:
    - Supports Fixed Ranges (Physiological limits, e.g., Glucose 0-600) via config.
    - Supports Data-Driven Ranges (Calculated dynamically from the Train set).
    - Handles Chimera Datasets (Gracefully skips columns if missing in specific subjects).
    - Safely clips values to [0, 1] interval.
    """

    def __init__(
        self,
        cols_to_normalize: list[str],
        fixed_ranges: dict[str, tuple[float, float]] | None = None
    ) -> None:
        """
        Initialize the normalizer.

        Args:
            cols_to_normalize: List of column names that require normalization.
            fixed_ranges: Dictionary mapping column names to (min, max) tuples.
                          If a column is provided here, these values are used
                          regardless of the data distribution.
        """
        self.cols_to_normalize = cols_to_normalize
        self.fixed_ranges = fixed_ranges or {}

        # Dictionary to store the final (min, max) used for scaling each column.
        # Structure: { 'column_name': (min_val, max_val) }
        self.scaling_params: dict[str, tuple[float, float]] = {}
        self._is_fitted = False

    def fit(self, train_dfs: list[pd.DataFrame]) -> None:
        """
        Calculates min/max statistics based ONLY on the provided TRAINING DataFrames.

        Args:
            train_dfs: List of DataFrames belonging to the training split.
        """
        print(f"   [Normalizer] Fitting on {len(train_dfs)} training subjects...")

        # Reset parameters before fitting
        self.scaling_params = {}

        for col in self.cols_to_normalize:
            # CASE A: Fixed Range (Defined in Configuration)
            if col in self.fixed_ranges and self.fixed_ranges[col] is not None:
                self.scaling_params[col] = self.fixed_ranges[col]
                continue

            # CASE B: Data-Driven Range (Calculate from Train Data)
            # We iterate over all DataFrames because the input is a list, not a single monolithic DF.
            # We must handle the case where a column might be missing in some DataFrames (Chimera).

            global_min = float('inf')
            global_max = float('-inf')
            found_data = False

            for df in train_dfs:
                if col in df.columns:
                    # Skip NaNs automatically
                    c_min = df[col].min()
                    c_max = df[col].max()

                    if not pd.isna(c_min) and not pd.isna(c_max):
                        global_min = min(global_min, float(c_min))
                        global_max = max(global_max, float(c_max))
                        found_data = True

            # If the column was not found in any training dataframe, skip it.
            if not found_data:
                print(f"     [!] Warning: Column '{col}' not found in any training data. Skipping.")
                continue

            # Handle edge case: Constant feature (max == min)
            if abs(global_max - global_min) < 1e-9:
                print(f"     [!] Warning: Column '{col}' is constant ({global_min}). Adding epsilon margin.")
                global_max += 1.0

            self.scaling_params[col] = (global_min, global_max)

        self._is_fitted = True
        print(f"   [Normalizer] Fitted successfully on {len(self.scaling_params)} features.")

    def transform(self, dfs: list[pd.DataFrame]) -> list[pd.DataFrame]:
        """
        Applies the learned normalization to a list of DataFrames.

        Formula: x_norm = (x - min) / (max - min)
        Values are clipped to the [0, 1] interval.

        Args:
            dfs: List of DataFrames to normalize (can be Train, Val, or Test).

        Returns:
            A new list of normalized DataFrames (does not mutate inputs in-place).

        Raises:
            RuntimeError: If called before .fit().
        """
        if not self._is_fitted:
            raise RuntimeError("Normalizer must be fitted before calling transform.")

        out_dfs = []

        # Optimization: Pre-calculate spans to avoid re-computing inside the loop
        # param_map structure: { col_name: (min_val, span_val) }
        param_map = {}
        for col, (min_v, max_v) in self.scaling_params.items():
            span = max_v - min_v
            # Safety check against division by zero (should be handled in fit, but double check)
            if span < 1e-9:
                span = 1.0
            param_map[col] = (min_v, span)

        for original_df in dfs:
            # Create a copy to ensure we do not modify the original data
            df = original_df.copy()

            for col, (min_v, span) in param_map.items():
                if col in df.columns:
                    # Apply MinMax Scaling Formula
                    df[col] = (df[col] - min_v) / span

                    # Clip to ensure valid range [0, 1]
                    # This is critical for fixed ranges, as test data might exceed physiological limits.
                    df[col] = df[col].clip(0.0, 1.0)

            out_dfs.append(df)

        return out_dfs

    def get_params(self) -> dict[str, tuple[float, float]]:
        """
        Returns the dictionary of learned ranges.
        Useful for passing these parameters to the WindowReconstructor later.

        Returns:
            Dictionary {feature_name: (min, max)}
        """
        return self.scaling_params


def denormalize_numpy_array(
        array: np.ndarray,
        feature_name: str,
        scaling_params: dict[str, tuple[float, float]],
) -> np.ndarray:
    """
    Helper function used by the Reconstructor to reverse normalization.
    It operates on Numpy arrays (typically the output of the GAN).

    Formula: x_real = x_norm * (max - min) + min

    Args:
        array: Normalized numpy array (usually in [0, 1]).
        feature_name: Name of the feature (key to look up in params).
        scaling_params: Dictionary {feature_name: (min, max)}.

    Returns:
        Denormalized array in physical units.
        If feature_name is not found in scaling_params, returns the array unchanged.
    """
    if feature_name not in scaling_params:
        return array

    min_val, max_val = scaling_params[feature_name]
    span = max_val - min_val

    # Apply inverse transformation
    # We assume 'array' is float32/64.
    return array * span + min_val


def denormalize_dataframes(
        dfs: list[pd.DataFrame],
        scaling_params: dict[str, tuple[float, float]]
) -> list[pd.DataFrame]:
    """
    Denormalizes a list of DataFrames in-place (or returning copies)
    using the provided scaling parameters.

    It ignores columns in the DataFrame that are not present in scaling_params.
    """
    if not dfs:
        return []

    # 1. Pre-calculate spans to make the loop faster
    # map: col_name -> (min, span)
    params_map = {}
    for col, (min_v, max_v) in scaling_params.items():
        params_map[col] = (min_v, max_v - min_v)

    denormalized_dfs = []

    for df in dfs:
        # Work on a copy to avoid side-effects on the original templates
        df_out = df.copy()

        for col, (min_v, span) in params_map.items():
            if col in df_out.columns:
                # Formula: real = norm * span + min
                df_out[col] = df_out[col] * span + min_v

        df_out.attrs = df.attrs.copy()
        denormalized_dfs.append(df_out)

    return denormalized_dfs

