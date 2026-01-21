import pandas as pd
from pathlib import Path
from typing import Any

from .interface import DataProcessor
from ..utils import clean_duplicates, print_duplicate_counts


class DuplicateRemover(DataProcessor):
    """
    Processor responsible for handling duplicate timestamps in the dataset.

    It wraps the original `clean_duplicates` utility function to aggregate or remove
    rows sharing the same time index.
    """

    def process(self, data_list: list[pd.DataFrame], context: dict[str, Any]) -> list[pd.DataFrame]:
        """
        Detects and resolves duplicate timestamps using the utility logic.

        Args:
            data_list: list of subject DataFrames.
            context: Pipeline context containing 'logging_dir'.

        Returns:
            list[pd.DataFrame]: DataFrames with unique time indices.
        """
        logging_dir = context.get('logging_dir')

        print(f"   [DuplicateRemover] Checking and resolving duplicate timestamps...")

        # 1. Logging (Optional)
        # Saves the count of duplicates before cleaning, for audit purposes.
        if logging_dir is not None:
            log_path = Path(logging_dir) / "duplicate_counts.txt"
            print_duplicate_counts(data_list, log_path)

        # 2. Execution
        # Delegates the actual logic to the centralized utility function.
        cleaned_data = clean_duplicates(data_list)

        return cleaned_data