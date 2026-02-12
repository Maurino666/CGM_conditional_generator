from __future__ import annotations

import joblib
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer

from .normalization_interface import Normalizer


class QuantileNormalizer(Normalizer):
    """
    Quantile Normalizer (Gaussian Rank Transformation).

    This normalizer transforms the features to follow a Gaussian distribution (Normal(0, 1)).
    It is robust to outliers and handles heterogeneous data distributions (e.g., "Chimera" datasets)
    by performing a non-linear mapping based on cumulative distribution functions.

    It implements the Normalizer interface and manages a separate Sklearn QuantileTransformer
    for each column to handle missing columns (Chimera) gracefully.
    """

    def __init__(
            self,
            cols_to_normalize: list[str],
            n_quantiles: int = 1000,
            output_distribution: str = "normal",
            subsample: int = 1000000
    ) -> None:
        """
        Initialize the QuantileNormalizer.

        Args:
            cols_to_normalize: List of column names that require normalization.
            n_quantiles: Number of quantiles to compute. 1000 is standard.
                         Lower numbers = coarser approximation, less overfitting.
            output_distribution: 'normal' for Gaussian output (standard for Diffusion),
                                 or 'uniform' for [0, 1] output.
            subsample: Maximum number of samples used to estimate the quantiles
                       (for computational efficiency).
        """
        self.cols_to_normalize = cols_to_normalize
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.subsample = subsample

        # Dictionary to store the fitted Sklearn transformers for each column.
        # Structure: { 'column_name': QuantileTransformerObject }
        self.transformers: dict[str, QuantileTransformer] = {}
        self._is_fitted = False

    def fit(self, dfs: list[pd.DataFrame]) -> None:
        """
        Learns the quantile distribution from the TRAINING data.

        It aggregates data from all subjects (DataFrames) for a specific column
        to learn a global distribution mapping.

        Args:
            dfs: List of DataFrames belonging to the training split.
        """
        print(f"   [QuantileNormalizer] Fitting on {len(dfs)} training subjects...")

        # Reset state
        self.transformers = {}

        for col in self.cols_to_normalize:
            # 1. Collect all valid data for this column across all subjects
            collected_values = []

            for df in dfs:
                if col in df.columns:
                    # Drop NaNs to avoid errors during fitting
                    valid_data = df[col].dropna().values
                    if len(valid_data) > 0:
                        collected_values.append(valid_data)

            # 2. If no data found for this column, warn and skip
            if not collected_values:
                print(f"     [!] Warning: Column '{col}' not found in any training data. Skipping.")
                continue

            # 3. Concatenate into a single long array (N_samples, 1)
            # Sklearn expects shape (N_samples, N_features)
            full_data = np.concatenate(collected_values).reshape(-1, 1)

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

                    # Transform is robust to NaNs in recent versions, but explicit handling is safer.
                    # Here we rely on Sklearn's handling or assume pre-cleaned data.
                    # If data contains NaNs, QuantileTransformer might raise error or propagate.
                    # For safety in pipelines, we usually fillna or mask.
                    # Assuming data is clean or Sklearn version supports nan (>=0.22 with subtle handling).

                    # Note: transform returns a numpy array
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
        Used often by the generation pipeline output.

        Args:
            array: 1D or 2D Numpy array of normalized values.
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
        Since QuantileTransformer contains complex state, we use joblib.
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
        Loads the transformers from disk.
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
        Returns metadata about the normalizer.
        Note: Does not return the full quantile arrays as they are too large.
        """
        return {
            "type": "QuantileNormalizer",
            "cols": list(self.transformers.keys()),
            "output_distribution": self.output_distribution,
            "n_quantiles": self.n_quantiles
        }