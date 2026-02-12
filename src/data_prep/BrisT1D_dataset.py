import pandas as pd
from typing import Any

from .base_dataset import BaseDataset
from .processors.interface import DataProcessor
from .processors.mapping import ColumnMapper
from .processors.indexing import TimeIndexer
from .processors.glucose_converter import GlucoseUnitConverter


class BrisT1DAligner(DataProcessor):
    """
    Aligns asynchronous device data to a fixed time grid (Smart Snapping).

    Problem Solved:
        In the BrisT1D dataset, devices are asynchronous:
        - CGM might record at 12:03:00
        - Insulin Pump might record at 12:05:00
        This creates sparse rows and misalignment if strictly indexed.

    Solution:
        1. Rounds every timestamp to the nearest grid point (e.g., 5 min).
           Example: 12:03 -> 12:05.
        2. Aggregates colliding records (e.g., if a CGM reading and a Bolus
           end up in the same 12:05 bin).
           - Glucose -> Mean (averages multiple readings if present)
           - Insulin/Carbs -> Sum (accumulates total dose/intake)
    """

    def __init__(self, freq: str = "5min"):
        """
        Args:
            freq: The target frequency string (pandas format, e.g., '5min', '15min').
        """
        self.freq = freq

    def process(self, data_list: list[pd.DataFrame], context: dict[str, Any]) -> list[pd.DataFrame]:
        print(f"   [BrisT1DAligner] Aligning timestamps to a fixed {self.freq} grid...")

        # Retrieve the canonical name of the time column (usually 'timestamp' or 'Date')
        time_col = context.get('time_col', 'timestamp')

        for i, df in enumerate(data_list):
            if df.empty:
                continue

            # --- Step 1: Ensure Timestamp Column is available and typed ---
            if time_col not in df.columns:
                # Fallback: if data is already indexed, move index to column
                df = df.reset_index()

            # Force datetime type to enable .dt accessors
            df[time_col] = pd.to_datetime(df[time_col])

            # --- Step 2: Smart Snap (Rounding) ---
            # We create a temporary column 'grid_time'.
            # .round() is adaptive: 00:02 -> 00:00, 00:03 -> 00:05.
            # This acts as a local dynamic shift, correcting clock drift.
            df['grid_time'] = df[time_col].dt.round(self.freq)

            # --- Step 3: Define Aggregation Rules (Collision Handling) ---
            # If multiple records snap to the same bin, how do we merge them?
            agg_rules = {}

            # A. Physiological Variables
            if 'bg' in df.columns:
                agg_rules['bg'] = 'mean'  # Average multiple sensor readings to reduce noise

            if 'insulin' in df.columns:
                agg_rules['insulin'] = 'sum'  # Sum doses! 0.5U + 2.0U must equal 2.5U

            if 'carbs' in df.columns:
                agg_rules['carbs'] = 'sum'  # Sum carbohydrate intake

            if 'hr' in df.columns:
                agg_rules['hr'] = 'mean'  # Average Heart Rate

            if 'steps' in df.columns:
                agg_rules['steps'] = 'sum'  # Accumulate steps

            # B. Metadata / Static Columns (Device names, IDs, etc.)
            # For any column not explicitly handled above, we keep the 'first' non-null value found.
            # We exclude the time columns from this loop.
            existing_cols = [c for c in df.columns if c not in ['grid_time', time_col]]
            for col in existing_cols:
                if col not in agg_rules:
                    agg_rules[col] = 'first'

            # --- Step 4: Execute Merge ---
            # Group by the snapped time and apply the rules.
            # reset_index() flattens the result back to a DataFrame.
            df_aligned = df.groupby('grid_time').agg(agg_rules).reset_index()

            # --- Step 5: Cleanup ---
            # Rename the grid time back to the standard time column name
            df_aligned = df_aligned.rename(columns={'grid_time': time_col})

            # Update the list in place
            data_list[i] = df_aligned

        return data_list




class BrisT1DDataset(BaseDataset):
    """
    Dataset loader for the BrisT1D Dataset (University of Bristol).

    Key Characteristics:
    - Source Units: mmol/L (requires conversion to mg/dL).
    - Asynchronous timestamps: Devices record at different offsets (e.g., :03 vs :05).
    - Requires specific pipeline ordering to ensure clean indexing.
    """

    name = "brist1d"

    def _init_structure_pipeline(self) -> list[DataProcessor]:
        """
        Override the default structure pipeline to inject Alignment and Conversion logic.

        Execution Order is CRITICAL:
        1. ColumnMapper:
           Renames raw columns (e.g. 'Glucose Level') to standard internal names ('bg', 'insulin').
           This allows subsequent processors to refer to 'bg' safely.

        2. GlucoseUnitConverter:
           Converts 'bg' from mmol/L to mg/dL.
           Must happen before alignment to ensure we aggregate values in the correct scale.

        3. BrisT1DAligner:
           Snaps asynchronous timestamps to the target frequency grid (e.g., 5min).
           Merges rows that fall into the same bin (summing insulin, averaging BG).

        4. TimeIndexer:
           Sets the DataFrame index to the time column and sorts it.
           It receives a clean, regularized time series from the Aligner, avoiding duplicate index errors.
        """

        # Extract target frequency from config (default to 5min if missing)
        target_freq = self.config["sampling"].get("target_frequency", "5min")

        return [
            # Step 1: Standardize Names
            ColumnMapper(),

            # Step 2: Harmonize Units
            GlucoseUnitConverter(
                source_col="glucose",
                target_unit="mg/dL"
            ),

            # Step 3: Fix Time Alignment (The "Smart Snap")
            BrisT1DAligner(
                freq=target_freq
            ),

            # Step 4: Create Index (Now safe)
            TimeIndexer(),
        ]
