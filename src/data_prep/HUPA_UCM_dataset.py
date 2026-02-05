from pathlib import Path
from typing import Any
import pandas as pd

from .base_dataset import BaseDataset
from .processors.cleaning import TypeAndValueCleaner
from .processors.duplicates import DuplicateRemover
from .processors.gaps import GapFiller
from .processors.insuline_merger import InsulinComponentsMerger
from .processors.interface import DataProcessor


class HUPACarbsScaler(DataProcessor):
    """
    Processor to standardize carbohydrate units in the HUPA-UCM dataset.

    Problem:
    The paper states that 'carb_input' is in servings (1 serving = 10g).
    However, empirical observation shows some files contain values like 130,
    which would imply 1300g of carbs if treated as servings (physiologically impossible).
    This suggests some data might have remained in grams during the authors' preprocessing.

    Solution:
    Apply an adaptive heuristic based on the maximum value found in each patient's file.
    """

    def process(self, data_list: list[pd.DataFrame], context: dict[str, Any]) -> list[pd.DataFrame]:
        print(f"   [HUPACarbsScaler] Adaptive rescaling (Servings vs Grams)...")

        # Thresholds configuration
        # If max carbs < 50, it is likely "servings" (e.g., 5 servings = 50g).
        # If max carbs > 50, it is likely already "grams" (e.g., 60g, 100g).
        THRESHOLD_SERVINGS = 50.0

        # Physiological safety limit to clip extreme outliers (e.g., typos).
        # No single meal should realistically exceed 400g of carbs.
        MAX_PHYSIOLOGICAL_CARBS = 400.0

        for i, df in enumerate(data_list):
            if "carbs" in df.columns:
                # 1. Analyze the current range
                # We check the maximum value to infer the unit of measurement.
                max_val = df["carbs"].max()

                # If the column is empty or all zeros, skip processing
                if max_val == 0:
                    data_list[i] = df
                    continue

                # 2. Adaptive Decision Logic
                if max_val < THRESHOLD_SERVINGS:
                    # Case A: Small values detected (e.g., 4, 8, 12).
                    # Consistent with the paper's claim of "servings"[cite: 63].
                    # Action: Convert to grams (* 10).
                    df["carbs"] = df["carbs"] * 10


                # 3. Safety Clipping
                # Ensure no value exceeds the physiological maximum.
                # This fixes potential outliers (e.g., if a user accidentally typed 1300).
                before_clip_max = df["carbs"].max()
                df["carbs"] = df["carbs"].clip(upper=MAX_PHYSIOLOGICAL_CARBS)

                if before_clip_max > MAX_PHYSIOLOGICAL_CARBS:
                     print(f"     -> Subj {i}: Clipped outlier {before_clip_max} to {MAX_PHYSIOLOGICAL_CARBS}")

            data_list[i] = df

        return data_list


class HUPA_UCMDataset(BaseDataset):
    """
    Dataset class for HUPA_UCM.

    It extends BaseDataset by injecting a specific scaling step
    into the cleaning pipeline.
    """
    name="hupa_ucm"

    def __init__(
            self,
            dataset_root: Path,
            config_file: Path,
            global_config_file: Path | None = None,
            patient_metadata_path: Path | None = None,
            logging_dir: Path | None = None
    ):
        # Initialize the base orchestrator
        super().__init__(
            dataset_root=dataset_root,
            config_file=config_file,
            global_config_file=global_config_file,
            patient_metadata_path=patient_metadata_path,
            logging_dir=logging_dir,
        )



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
                target_col="insulin"
            ),

            # 3. Standard Cleaning
            DuplicateRemover(),
            GapFiller(),

            # 4. Dataset Specific Cleaning
            HUPACarbsScaler(),
        ]