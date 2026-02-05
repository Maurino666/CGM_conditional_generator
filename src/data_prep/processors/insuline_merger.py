import pandas as pd
from typing import Any

from .interface import DataProcessor


class InsulinComponentsMerger(DataProcessor):
    """
    Merges separate insulin components into a single 'total insulin' column.

    This processor is essential for datasets (like AZT1D) where Basal Rate (continuous)
    and Bolus (discrete) are stored separately. It calculates the total insulin delivered
    within each time bucket.
    """

    def __init__(self, basal_col="basal_rate", bolus_col="bolus", target_col="insulin"):
        """
        Args:
            basal_col: Name of the column containing Basal Rate (in U/h).
            bolus_col: Name of the column containing Bolus doses (in U).
            target_col: Name of the new column to create (in U).
        """
        self.basal_col = basal_col
        self.bolus_col = bolus_col
        self.target_col = target_col

    def process(self, data_list: list[pd.DataFrame], context: dict[str, Any]) -> list[pd.DataFrame]:
        print(
            f"   [InsulinComponentsMerger] Merging '{self.basal_col}' and '{self.bolus_col}' into '{self.target_col}'...")

        # Calculate the conversion factor from Rate (U/h) to Quantity (U) per time step.
        # Example: If target_frequency is '5min', there are 12 steps in an hour.
        # Quantity = Rate / 12.
        freq_str = context['config']['sampling']['target_frequency']
        freq_delta = pd.Timedelta(freq_str)
        samples_per_hour = pd.Timedelta("1h") / freq_delta

        for i, df in enumerate(data_list):
            # Initialize total insulin to 0.0
            total_insulin = 0.0

            # 1. Add Basal Insulin (Convert Rate U/h -> Quantity U)
            if self.basal_col in df.columns:
                # We use fillna(0) to ensure scalar addition works even if some rows are NaN
                # Note: Basal gaps should ideally be filled (e.g., ffill) before this step.
                total_insulin += df[self.basal_col].fillna(0) / samples_per_hour

            # 2. Add Bolus Insulin (Quantity U)
            if self.bolus_col in df.columns:
                total_insulin += df[self.bolus_col].fillna(0)

            # Assign the result to the target column
            df[self.target_col] = total_insulin
            data_list[i] = df

        return data_list