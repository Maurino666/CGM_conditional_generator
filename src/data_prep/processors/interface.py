from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class DataProcessor(ABC):
    """
    Abstract base class for all data processing steps.
    """

    @abstractmethod
    def process(self, data_list: list[pd.DataFrame], context: dict[str, Any]) -> list[pd.DataFrame]:
        """
        Applies a transformation to the list of DataFrames.

        Args:
            data_list: The list of subject DataFrames.
            context: A dictionary containing config, global_config, schema, etc.

        Returns:
            The modified list of DataFrames.
        """
        pass