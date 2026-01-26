from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


class Normalizer(ABC):
    """
    Abstract Base Class (Interface) for Data Normalization.
    Enforces a standard API for fitting, transforming, and persisting parameters.
    """

    @abstractmethod
    def fit(self, dfs: list[pd.DataFrame]) -> None:
        """Learns the scaling parameters from the training data."""
        pass

    @abstractmethod
    def transform(self, dfs: list[pd.DataFrame]) -> list[pd.DataFrame]:
        """Applies the normalization to the data."""
        pass

    @abstractmethod
    def inverse_transform(self, dfs: list[pd.DataFrame]) -> list[pd.DataFrame]:
        """Reverts the normalization (denormalization)."""
        pass

    @abstractmethod
    def inverse_transform_array(self, array: np.ndarray, feature_name: str) -> np.ndarray:
        """Reverts the normalization (denormalization)."""
        pass

    @abstractmethod
    def save_params(self, save_path: str | Path) -> None:
        """Saves the learned parameters to disk."""
        pass

    @abstractmethod
    def load_params(self, load_path: str | Path) -> None:
        """Loads parameters from disk and sets the normalizer as fitted."""
        pass

    @abstractmethod
    def get_params(self) -> dict:
        """Returns the internal scaling parameters (useful for external logging/usage)."""
        pass