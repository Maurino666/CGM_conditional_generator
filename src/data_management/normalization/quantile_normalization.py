from __future__ import annotations

import joblib
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer

from .normalization_interface import Normalizer


class QuantileNormalizer(Normalizer):
    """
    Quantile Normalizer (Gaussian Rank Transformation) with Mask-Aware Fitting.

    This normalizer transforms the features to follow a Gaussian distribution (Standard Normal: mean=0, std=1).
    This output distribution is ideal for Diffusion Models (DDPM) which assume Gaussian priors.

    Key Features:
    1.  **Robust to Outliers:** Handles heavy-tailed distributions (like Type 1 Diabetes glucose levels)
        by performing a non-linear mapping based on cumulative distribution functions (CDF).
        Extreme values are mapped to the tails of the Gaussian (e.g., +/- 3 sigma) rather than
        breaking the numerical scale.
    2.  **Chimera Dataset Support:** It fits a separate transformer for each column. If a column is missing
        in a specific subject (DataFrame), that subject is simply ignored for that feature's statistics.
    3.  **Mask-Aware Fitting:** It can optionally ignore padded values (zeros) during the .fit() phase
        if a corresponding mask column is found. This prevents the "Zero Spike" problem where
        padding corrupts the learned distribution.

    It implements the Normalizer interface and manages a separate Sklearn QuantileTransformer
    for each column.
    """

    def __init__(
            self,
            cols_to_normalize: list[str],
            mask_suffix: str = "_mask",
            n_quantiles: int = 1000,
            output_distribution: str = "normal",
            subsample: int = 1000000
    ) -> None:
        """
        Initialize the QuantileNormalizer.

        Args:
            cols_to_normalize: List of column names that require normalization (e.g., ['glucose', 'insulin']).
            mask_suffix: Suffix used to identify mask columns. If col='glucose' and suffix='_mask',
                         the normalizer looks for 'glucose_mask' to filter out padding during fitting.
            n_quantiles: Number of quantiles to compute. 1000 is standard for high resolution.
                         Defines the granularity of the mapping.
            output_distribution: Target distribution.
                                 'normal': transforms data to Standard Normal (Gaussian). Best for Diffusion/VAE.
                                 'uniform': transforms data to Uniform [0, 1].
            subsample: Maximum number of samples used to estimate the quantiles.
                       Lower values speed up fitting on huge datasets but might miss rare outliers.
        """
        self.cols_to_normalize = cols_to_normalize
        self.mask_suffix = mask_suffix
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.subsample = subsample

        # Dictionary to store the fitted Sklearn transformers for each column.
        # Structure: { 'column_name': QuantileTransformerObject }
        self.transformers: dict[str, QuantileTransformer] = {}
        self._is_fitted = False

    def fit(self, dfs: list[pd.DataFrame]) -> None:
        """
        Learns the quantile distribution from the TRAINING data (Mask-Aware).

        It aggregates data from all subjects (DataFrames) for a specific column
        to learn a global distribution mapping.

        CRITICAL: If a mask column is present (e.g. 'glucose_mask'), it uses it to filter
        out padding (zeros) so they don't skew the distribution.

        Args:
            dfs: List of DataFrames belonging to the training split.
        """
        print(f"   [QuantileNormalizer] Fitting on {len(dfs)} training subjects (Mask-Aware)...")

        # Reset state
        self.transformers = {}

        for col in self.cols_to_normalize:
            # 1. Collect all valid data for this column across all subjects
            collected_values = []
            mask_col_name = f"{col}{self.mask_suffix}"  # e.g., glucose_mask

            for df in dfs:
                if col in df.columns:
                    # Extract raw data
                    data = df[col].values

                    # Mask Handling Logic
                    if mask_col_name in df.columns:
                        # If mask exists, keep only values where mask > 0 (Present)
                        # Assumes mask is 1 for data, 0 for padding
                        mask = df[mask_col_name].values > 0
                        valid_data = data[mask]
                    else:
                        # If no mask, assume all non-NaNs are valid
                        valid_data = df[col].dropna().values

                    # Accumulate only if there is valid data in this subject
                    if len(valid_data) > 0:
                        collected_values.append(valid_data)

            # 2. If no data found for this column, warn and skip
            if not collected_values:
                print(f"     [!] Warning: Column '{col}' not found (or fully masked) in training data. Skipping.")
                continue

            # 3. Concatenate into a single long array (N_samples, 1)
            # Sklearn expects shape (N_samples, N_features)
            full_data = np.concatenate(collected_values).reshape(-1, 1)

            # Safety check: QuantileTransformer needs variance
            if len(np.unique(full_data)) < 5:
                print(f"     [!] Warning: Column '{col}' is mostly constant. Normalization might be unstable.")

            # 4. Initialize and fit the Sklearn Transformer
            # We handle n_quantiles dynamically: cannot be > n_samples
            n_samples = full_data.shape[0]
            actual_quantiles = min(self.n_quantiles, n_samples)

            qt = QuantileTransformer(
                n_quantiles=actual_quantiles,
                output_distribution=self.output_distribution,
                subsample=self.subsample,
                random_state=42,  # Ensure reproducibility
                copy=True
            )

            qt.fit(full_data)
            self.transformers[col] = qt

        self._is_fitted = True
        print(f"   [QuantileNormalizer] Fitted successfully on {len(self.transformers)} features.")

    def transform(self, dfs: list[pd.DataFrame]) -> list[pd.DataFrame]:
        """
        Applies the Gaussian Rank Transformation to the data.

        NOTE: This transforms EVERYTHING, including padding zeros.
        It maps the padding zeros to whatever z-score corresponds to the value 0 in the distribution
        (e.g., -5.2 if 0 is the minimum).

        It is expected that the downstream Neural Network uses the corresponding mask
        to ignore these transformed padding values.

        Args:
            dfs: List of DataFrames to normalize.

        Returns:
            List of new DataFrames with normalized values (Gaussians).
        """
        if not self._is_fitted:
            raise RuntimeError("Normalizer must be fitted before calling transform.")

        out_dfs = []

        for original_df in dfs:
            # Create a copy to prevent mutation of the original DF
            df = original_df.copy()

            for col, transformer in self.transformers.items():
                if col in df.columns:
                    # Sklearn needs (N, 1) shape
                    # We handle NaNs by creating a mask, as Sklearn can choke on NaNs
                    data = df[col].values.reshape(-1, 1)

                    # Transform returns a numpy array
                    # Note: QuantileTransformer is generally robust to NaNs in recent versions
                    # (it propagates them or ignores them depending on config), but we assume
                    # standard numerical data here.
                    transformed_data = transformer.transform(data)

                    # Flatten back to 1D and assign
                    df[col] = transformed_data.flatten()

            out_dfs.append(df)

        return out_dfs

    def inverse_transform(self, dfs: list[pd.DataFrame]) -> list[pd.DataFrame]:
        """
        Reverts the transformation (Gaussian -> Original Scale).
        Crucial for calculating metrics (RMSE, MAE) on physical units.
        """
        if not dfs:
            return []

        denormalized_dfs = []

        for original_df in dfs:
            df = original_df.copy()

            for col, transformer in self.transformers.items():
                if col in df.columns:
                    data = df[col].values.reshape(-1, 1)

                    # Inverse map: Gaussian -> Raw distribution
                    original_scale_data = transformer.inverse_transform(data)

                    df[col] = original_scale_data.flatten()

            denormalized_dfs.append(df)

        return denormalized_dfs

    def inverse_transform_array(self, array: np.ndarray, feature_name: str) -> np.ndarray:
        """
        Reverts normalization for a single numpy array (feature).
        Used often by the generation pipeline output (e.g., GAN/Diffusion output).

        Args:
            array: 1D or 2D Numpy array of normalized values (z-scores).
            feature_name: Name of the feature (e.g., 'glucose').
        """
        if feature_name not in self.transformers:
            # warnings.warn(f"Feature '{feature_name}' not found in normalizer. Returning unchanged.")
            return array

        transformer = self.transformers[feature_name]

        # Ensure correct shape (N, 1)
        original_shape = array.shape
        data_reshaped = array.reshape(-1, 1)

        # Inverse transform
        inverse_data = transformer.inverse_transform(data_reshaped)

        # Restore original shape
        return inverse_data.reshape(original_shape)

    def save_params(self, save_path: str | Path) -> None:
        """
        Saves the fitted transformer objects to disk.
        Since QuantileTransformer contains complex state (quantile arrays), we use joblib.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save parameters: Normalizer is not fitted yet.")

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        file_path = save_path / "quantile_transformers.joblib"

        # We save the whole dictionary of transformers
        joblib.dump(self.transformers, file_path)

        print(f"   [QuantileNormalizer] Transformers saved to {file_path}")

    def load_params(self, load_path: str | Path) -> None:
        """
        Loads the transformers from disk and sets the normalizer as fitted.
        """
        load_path = Path(load_path)
        file_path = load_path / "quantile_transformers.joblib"

        if not file_path.exists():
            # Fallback check if user provided the direct file path instead of directory
            if load_path.suffix == '.joblib' and load_path.exists():
                file_path = load_path
            else:
                raise FileNotFoundError(f"Transformer file not found at: {file_path}")

        self.transformers = joblib.load(file_path)
        self._is_fitted = True
        print(f"   [QuantileNormalizer] Loaded transformers for {len(self.transformers)} features.")

    def get_params(self) -> dict:
        """
        Returns metadata about the normalizer configuration.
        Note: Does not return the full quantile arrays as they are too large for simple logging.
        """
        return {
            "type": "QuantileNormalizer",
            "cols": list(self.transformers.keys()),
            "output_distribution": self.output_distribution,
            "n_quantiles": self.n_quantiles,
            "mask_suffix": self.mask_suffix
        }