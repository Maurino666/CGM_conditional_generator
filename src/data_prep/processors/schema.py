from typing import List, Dict, Any
import pandas as pd
import numpy as np

from .interface import DataProcessor


class SchemaStandardizer(DataProcessor):
    """
    Ensures all datasets strictly follow the Global Schema defined in config.

    It performs two critical operations:
    1. 'Chimera' Handling: If a required feature is missing in a dataset,
       it creates the column filled with 0.0 and sets its mask to 0.0.
    2. Row-Level Masking: If a feature exists but has NaN values in specific rows,
       it sets the mask to 0.0 for those rows and fills the value with 0.0.

    The output DataFrame will always have the structure:
    [Target, Feature_1, Feature_2, ..., Feature_1_mask, Feature_2_mask, ...]
    """

    def __init__(self, global_cond_cols: List[str] | None = None):
        """
        Args:
            global_cond_cols: List of conditional features expected by the model.
                              If None, it tries to read from context during process().
        """
        self.global_cond_cols = global_cond_cols

    def process(self, data_list: List[pd.DataFrame], context: Dict[str, Any]) -> List[pd.DataFrame]:
        # 1. Resolve Target and Conditional Columns from Context/Config
        target_col = context.get('target_col')
        if not target_col:
            # Fallback to config if not directly in context root
            target_col = context['config']['schema'].get('target_col', 'glucose')

        # Use passed cols or fetch from global config
        cond_cols = self.global_cond_cols
        if not cond_cols:
            # Try to get from global_config inside context
            # context -> global_config -> schema -> cond_cols
            try:
                cond_cols = context['global_config']['schema']['cond_cols']
            except KeyError:
                # If specifically testing without global config, we might skip or warn
                # For now, we assume it's provided or empty list
                cond_cols = []

        print(f"   [SchemaStandardizer] Standardizing to {len(cond_cols)} features + masks...")

        for i, df in enumerate(data_list):
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' missing in dataset.")

            # We build a list of columns to keep ordered
            # Start with Target
            final_columns = [target_col]

            # Process each required conditional column
            for col in cond_cols:
                mask_col = f"{col}_mask"

                if col in df.columns:
                    # CASE A: Column exists in the dataset.
                    # We must handle partial missingness (NaNs).

                    # 1. Create Mask: 1.0 where data exists, 0.0 where NaN
                    # We cast to float32 immediately
                    df[mask_col] = df[col].notna().astype("float32")

                    # 2. Fill NaNs: Replace NaN with 0.0 in the data column
                    # This ensures the model receives valid numbers even if mask is 0.
                    df[col] = df[col].fillna(0.0)

                else:
                    # CASE B: Column is completely missing (Chimera).
                    # Create the column filled with 0.0
                    df[col] = 0.0
                    # Create the mask filled with 0.0
                    df[mask_col] = 0.0

                # Add to final list in order: Feature, then its Mask
                final_columns.append(col)
                # We append the mask column to the dataframe, but we might want 
                # to order them differently (e.g. all features then all masks).
                # For now, let's just ensure they exist.

            # Construct the list of mask names
            mask_columns = [f"{c}_mask" for c in cond_cols]

            # Enforce strict ordering: [Target, ...Features..., ...Masks...]
            # Any extra columns in the DF (e.g. raw notes) are dropped here.
            ordered_cols = [target_col] + cond_cols + mask_columns

            # Select and reorder
            data_list[i] = df[ordered_cols].copy()

        return data_list