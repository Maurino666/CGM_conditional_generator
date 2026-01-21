from pathlib import Path
from typing import Any
import pandas as pd

from .base_dataset import BaseDataset
from .processors.interface import DataProcessor


class AZTSpecificCleaner(DataProcessor):
    """
    Processor containing data cleaning logic specific to the AZT1D2025 dataset.

    Operations:
    1. device_mode: Replaces '0' or 0 with 'Unknown' and sets type to category.
    2. bolus_type: Replaces '0' or 0 with 'None' and sets type to category.
    3. basal_rate: Applies forward-fill to handle missing values (sensor protocol).
    """

    def process(self, data_list: list[pd.DataFrame], context: dict[str, Any]) -> list[pd.DataFrame]:
        print(f"   [AZTSpecificCleaner] Applying AZT-specific fixes (device_mode, bolus_type, basal_rate)...")

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
            if "basal_rate" in df.columns:
                df["basal_rate"] = df["basal_rate"].ffill()

            data_list[i] = df

        return data_list


class AZT1D2025Dataset(BaseDataset):
    """
    Dataset class for AZT1D2025.

    It extends BaseDataset by injecting a specific cleaning step
    into the cleaning pipeline.
    """

    def __init__(
            self,
            dataset_root: Path,
            config_file: Path,
            global_config_file: Path | None = None,
            logging_dir: Path | None = None
    ):
        # Initialize the base orchestrator
        super().__init__(dataset_root, config_file, global_config_file, logging_dir)

        # INJECTION: Insert the dataset-specific cleaner into the pipeline.
        # We place it at index 1, immediately after TypeAndValueCleaner (index 0),
        # so we work on typed data before dealing with duplicates or gaps.
        self.cleaning_pipeline.append(AZTSpecificCleaner())