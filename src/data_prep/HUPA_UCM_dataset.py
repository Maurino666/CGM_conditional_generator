from pathlib import Path
from typing import Any
import pandas as pd

from .base_dataset import BaseDataset
from .processors.interface import DataProcessor


class HUPACarbsScaler(DataProcessor):
    """
    Processor containing data cleaning logic specific to the HUPA_UCM dataset.

    Operations:
    1. carbs: Multiplies the carbohydrate values by 10 to correct a unit scaling issue
       in the raw data source.
    """

    def process(self, data_list: list[pd.DataFrame], context: dict[str, Any]) -> list[pd.DataFrame]:
        print(f"   [HUPACarbsScaler] Rescaling carb values (*10)...")

        for i, df in enumerate(data_list):
            if "carbs" in df.columns:
                df["carbs"] = df["carbs"] * 10

            data_list[i] = df

        return data_list


class HUPA_UCMDataset(BaseDataset):
    """
    Dataset class for HUPA_UCM.

    It extends BaseDataset by injecting a specific scaling step
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

        # INJECTION: Insert the scaler into the cleaning pipeline.
        # Placed after TypeAndValueCleaner to ensure 'carbs' is already numeric.
        self.cleaning_pipeline.append(HUPACarbsScaler())