from typing import Any
import pandas as pd
from .base_dataset import BaseDataset
from .processors.interface import DataProcessor


from .processors.cleaning import TypeAndValueCleaner
from .processors.duplicates import DuplicateRemover
from .processors.gaps import GapFiller
from .processors.insuline_merger import InsulinComponentsMerger


class AZTSpecificCleaner(DataProcessor):
    """
    Processor containing data cleaning logic specific to the AZT1D2025 dataset.

    Operations:
    1. device_mode: Replaces '0' with 'Unknown' (Category).
    2. bolus_type: Replaces '0' with 'None' (Category).
    3. basal_rate: Applies forward-fill to handle missing values (sensor protocol).
    """

    def process(self, data_list: list[pd.DataFrame], context: dict[str, Any]) -> list[pd.DataFrame]:
        print(f"   [AZTSpecificCleaner] Applying AZT-specific fixes...")

        for i, df in enumerate(data_list):
            # 1. Fix device_mode
            if "device_mode" in df.columns:
                df["device_mode"] = df["device_mode"].replace(
                    {"0": "Unknown", 0: "Unknown"}
                ).fillna("Unknown").astype("category")

            # 2. Fix bolus_type
            if "bolus_type" in df.columns:
                df["bolus_type"] = df["bolus_type"].replace(
                    {"0": "None", 0: "None"}
                ).fillna("None").astype("category")

            # 3. Fix basal_rate gaps
            # Critical: Basal rate is a stateful value. If missing, it implies
            # "same as before", so we use ffill().
            if "basal_rate" in df.columns:
                df["basal_rate"] = df["basal_rate"].ffill().fillna(0)

            data_list[i] = df

        return data_list


class AZT1D2025Dataset(BaseDataset):
    """
    Dataset class for AZT1D2025.

    Key Characteristics:
    - High granularity: Separates Basal Rate and Bolus.
    - Requires merging these components into a 'total insulin' column for global compatibility.
    """
    name = "azt1d2025"

    def _init_cleaning_pipeline(self) -> list[DataProcessor]:
        """
        Override the cleaning pipeline to inject Insulin Merging logic.

        Execution Order:
        1. TypeAndValueCleaner: Standard type fixing.
        2. AZTSpecificCleaner: Fills gaps in 'basal_rate' (CRITICAL before merging).
        3. InsulinComponentsMerger: computes 'insulin' = basal + bolus.
        4. DuplicateRemover / GapFiller: Standard cleaning.
        """
        return [
            # 1. Standard Type Cleaning
            TypeAndValueCleaner(),

            # 2. Create Total Insulin Column
            # Requires clean 'basal_rate' from previous step
            InsulinComponentsMerger(
                basal_col="basal_rate",
                bolus_col="bolus_total",
                target_col="total_insulin"
            ),

            # 3. Standard Cleaning
            DuplicateRemover(),
            GapFiller(),

            # 4. Dataset Specific Cleaning
            AZTSpecificCleaner(),
        ]