import pandas as pd
from typing import Any
from .interface import DataProcessor


class GlucoseUnitConverter(DataProcessor):
    """
    Converts blood glucose values from mmol/L (standard in UK/Europe)
    to mg/dL (standard in US and for this specific ML model).

    Formula: mg/dL = mmol/L * 18.0182
    """

    def __init__(self, source_col: str = "bg", target_unit: str = "mg/dL"):
        """
        Args:
            source_col: The name of the column containing glucose values.
            target_unit: Descriptive tag for logging (default: "mg/dL").
        """
        self.source_col = source_col
        self.conversion_factor = 18.0182

    def process(self, data_list: list[pd.DataFrame], context: dict[str, Any]) -> list[pd.DataFrame]:
        print(f"   [GlucoseUnitConverter] Converting '{self.source_col}' values to mg/dL...")

        for i, df in enumerate(data_list):
            # Skip if the column doesn't exist (e.g., patient has no CGM data)
            if self.source_col in df.columns:
                # 1. Apply the conversion formula
                df[self.source_col] = df[self.source_col] * self.conversion_factor

                # 2. Rounding
                # mg/dL is usually represented as an integer or with 1 decimal place.
                # We round to 0 decimals to reduce floating point noise, treating it as an integer-like signal.
                df[self.source_col] = df[self.source_col].round(0)

            data_list[i] = df

        return data_list