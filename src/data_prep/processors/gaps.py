import pandas as pd
from pathlib import Path
from typing import Any

from .interface import DataProcessor
from ..utils import fill_data


class GapFiller(DataProcessor):
    """
    Processor responsible for filling small gaps in the target column (e.g. glucose).

    It wraps the original `fill_data` utility function which typically performs
    linear interpolation for gaps smaller than a configured threshold.
    """

    def process(self, data_list: list[pd.DataFrame], context: dict[str, Any]) -> list[pd.DataFrame]:
        """
        Fills missing values in the target column based on time proximity.

        Args:
            data_list: list of subject DataFrames.
            context: Pipeline context containing:
                     - 'time_steps': Frequency of the dataset.
                     - 'max_small_gap': Max duration to interpolate.
                     - 'target_col': Name of the column to fill.
                     - 'defaults': dictionary for fallback values.
                     - 'logging_dir': Path for saving gap reports.

        Returns:
            list[pd.DataFrame]: DataFrames with small gaps filled.
        """
        print(f"   [GapFiller] Filling small gaps in target column...")

        # Extract parameters from context
        time_steps = context.get('time_steps')
        max_small_gap = context.get('max_small_gap')
        target_col = context.get('target_col')
        defaults = context.get('defaults', {})
        logging_dir = context.get('logging_dir')

        # Prepare logging path
        gaps_log_path = None
        if logging_dir is not None:
            gaps_log_path = Path(logging_dir) / "gaps"
            gaps_log_path.mkdir(parents=True, exist_ok=True)

        # Execution
        # Calls the utility function with parameters extracted from the pipeline context
        filled_data = fill_data(
            data=data_list,
            expected=time_steps,
            max_gap=max_small_gap,
            target_col=target_col,
            defaults=defaults,
            logging_path=gaps_log_path
        )

        return filled_data