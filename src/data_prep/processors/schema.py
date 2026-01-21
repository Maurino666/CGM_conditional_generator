import pandas as pd
from typing import Any

from .interface import DataProcessor


class SchemaStandardizer(DataProcessor):
    """
    Processor responsible for solving the 'Chimera Dataset' problem.

    It assumes column names have ALREADY been mapped to the standard format
    (by ColumnMapper).

    Operations performed:
    1. Checks for the existence of each global conditional feature.
    2. If a feature exists: creates a corresponding '_mask' column set to 1.0.
    3. If a feature is missing: creates the feature column set to 0.0 AND a '_mask' column set to 0.0.
    4. Reorders columns to strictly follow: [Target, Global Features..., Global Masks...].
    """

    def __init__(self, global_cond_cols: list[str]):
        """
        Args:
            global_cond_cols: List of conditional column names expected by the model
                              (e.g., ['basal_rate', 'carbs', 'heart_rate']).
        """
        self.global_cond_cols = global_cond_cols

    def process(self, data_list: list[pd.DataFrame], context: dict[str, Any]) -> list[pd.DataFrame]:
        """
        Applies masking and standardization to the list of DataFrames.
        """
        target_col = context.get("target_col")

        # Safety check
        if not target_col:
            raise ValueError("[SchemaStandardizer] 'target_col' missing in context.")

        print(f"   [SchemaStandardizer] Standardizing {len(data_list)} subjects to global schema...")

        processed_data = []

        for i, df in enumerate(data_list):
            # Work on a copy to avoid SettingWithCopy warnings
            df = df.copy()

            existing_cols = set(df.columns)
            mask_cols_names = []

            # --- 1, 2 & 3: Create Features and Masks ---
            for col in self.global_cond_cols:
                mask_col_name = f"{col}_mask"
                mask_cols_names.append(mask_col_name)

                if col in existing_cols:
                    # Case A: Feature exists (e.g., 'bolus' in AZT)
                    # Set mask to 1.0 (Real Data)
                    df[mask_col_name] = 1.0
                else:
                    # Case B: Feature is missing (e.g., 'heart_rate' in AZT)
                    # Create placeholder feature at 0.0
                    df[col] = 0.0
                    # Set mask to 0.0 (Missing Data)
                    df[mask_col_name] = 0.0

            # --- 4: Enforce Strict Column Order ---
            # Order: [Target] + [Cond Cols] + [Mask Cols]
            # Note: Time is expected to be the Index (handled by TimeIndexer previously)

            final_columns = [target_col] + self.global_cond_cols + mask_cols_names

            # Check if target exists (Critical check)
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' missing in subject {i} (SchemaStandardizer). "
                                 f"Available columns: {list(df.columns)}")

            # Reindex forces the order and drops any extra local columns not in the schema
            df = df[final_columns]

            # Ensure float types for PyTorch compatibility
            # (Features, Targets and Masks should all be float)
            df = df.astype(float)

            processed_data.append(df)

        return processed_data