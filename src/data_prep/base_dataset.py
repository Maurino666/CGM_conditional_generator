from pathlib import Path
import pandas as pd
import yaml
import numpy as np
from typing import Any


from .utils import load_dataset, load_dataset_config, print_df_summary

from .processors.interface import DataProcessor
from .processors.mapping import ColumnMapper
from .processors.indexing import TimeIndexer
from .processors.cleaning import TypeAndValueCleaner
from .processors.duplicates import DuplicateRemover
from .processors.gaps import GapFiller
from .processors.schema import SchemaStandardizer
from .processors.augmentation import BaseTimeEventAugmenter



GLOBAL_CONFIG_PATH = "../global_config.yaml"


class BaseDataset:
    """
    Orchestrator class for data preparation.

    It manages the lifecycle of the dataset through two distinct pipelines:

    1. Structure Pipeline (Executed in __init__):
       - Ensures the DataFrame has the correct column names (Mapping).
       - Sets the correct Time Index (Indexing).
       - Enforces the Global Schema, creating masks and filling missing columns (SchemaStandardizer).
       Result: Structurally valid DataFrames (safe to access columns), but potentially 'dirty' values.

    2. Cleaning Pipeline (Executed in clean_data()):
       - Fixes data types and fills defaults (TypeAndValueCleaner).
       - Removes duplicates, fills gaps, and adds features.
       Result: ML-ready DataFrames.
    """

    # State variables
    all_data: list[pd.DataFrame]
    config: dict[str, Any]
    global_config: dict[str, Any]
    context: dict[str, Any]

    # Pipelines
    structure_pipeline: list[DataProcessor]
    cleaning_pipeline: list[DataProcessor]

    # Logging
    logging_dir: Path | None

    def __init__(
            self,
            dataset_root: Path,
            config_file: Path,
            global_config_file: Path | None = None,
            logging_dir: Path | None = None
    ):
        """
        Initializes the dataset, loads raw data, and enforces structural consistency.
        """
        # 1. Setup Paths & Configs
        if global_config_file is None:
            global_config_file = Path(GLOBAL_CONFIG_PATH)

        self.config = load_dataset_config(config_file)
        self.global_config = yaml.safe_load(open(global_config_file))
        self.logging_dir = logging_dir

        if self.logging_dir:
            self.logging_dir.mkdir(parents=True, exist_ok=True)

        # 2. Load Raw Data
        dataset_name = self.config['dataset'].get('name', 'Unnamed')
        print(f"[{dataset_name}] Loading raw data from {dataset_root}...")

        self.all_data = load_dataset(
            dataset_root,
            self.config["dataset"].get("separator", ",")
        )

        # 3. Build Context (Shared memory for Processors)
        self.context = {
            # Configs completi
            'config': self.config,
            'global_config': self.global_config,
            'logging_dir': self.logging_dir,

            # --- Info per Structure Pipeline ---
            'col_mapping': self.config['schema'].get('col_mapping', {}),
            'time_col': self.global_config['schema']['time_col'],
            'target_col': self.global_config['schema']['target_col'],
            'global_cond_cols': self.global_config['schema'].get('cond_cols', []),

            # --- Info per Cleaning Pipeline ---
            'numeric_cols': self._get_numeric_cols_from_config(),
            'defaults': self.config['schema'].get('defaults', {}),
            'impulse_cols': self.global_config['schema'].get('impulse_cols', []),
            'time_steps': pd.Timedelta(self.config["sampling"]["target_frequency"]),
            'max_small_gap': pd.Timedelta(self.global_config["options"].get("max_small_gap", "30min"))
        }

        # 4. Define Pipelines

        # A) Structure Pipeline (Mandatory & Immediate)
        self.structure_pipeline = [
            ColumnMapper(),  # 1. Raw Names -> Standard Names
            TimeIndexer(),  # 2. Set Index & Sort
        ]

        # B) Cleaning Pipeline (Lazy - runs on clean_data())
        self.cleaning_pipeline = [
            TypeAndValueCleaner(),
            DuplicateRemover(),
            GapFiller(),
        ]

        # C) Schema Standardization Pipeline
        self.standardization_pipeline = [
            SchemaStandardizer(  # 3. Create Masks & Enforce Column Order
                global_cond_cols=self.global_config['schema'].get('cond_cols', [])
            )
        ]

        # D) Augmentation Pipeline
        self.augmentation_pipeline = [
            BaseTimeEventAugmenter()
        ]

        # 5. Execute Structure Pipeline Immediately
        self._run_pipeline(self.structure_pipeline, "Initialization (Structure)")

        # Log initial structural summary
        if self.logging_dir:
            print_df_summary(self.all_data, self.logging_dir / "init_structure_summary.txt")

    def clean(self):
        """
        Triggers the heavy data processing: types, duplicates, gaps, and feature engineering.
        Subclasses can inject specific processors into self.cleaning_pipeline before calling this.
        """
        self._run_pipeline(self.cleaning_pipeline, "Data Cleaning")

        # Log final summary
        if self.logging_dir:
            print_df_summary(self.all_data, self.logging_dir / "post_cleaning_summary.txt")

    def standardize(self):
        """
        Adapts the dataset to the Global Schema.
        Adds missing columns (filled with 0) and '_mask' columns.
        Call this BEFORE training if mixing heterogeneous datasets.
        """
        self._run_pipeline(self.standardization_pipeline, "Schema Standardization")
        if self.logging_dir:
            print_df_summary(self.all_data, self.logging_dir / "post_standardization_summary.txt")

    def augment(self):
        """Adds synthetic features (Time encoding, IOB, COB)."""
        self._run_pipeline(self.augmentation_pipeline, "Data Augmentation")
        if self.logging_dir:
            print_df_summary(self.all_data, self.logging_dir / "post_augmentation_summary.txt")

    def split(
            self,
            val_ratio: float,
            split_by: str = "subject",
            random_state: int | None = None,
    ) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[int], list[int]]:
        """
        Splits the dataset into training and validation sets.

        Args:
            val_ratio: Float between 0 and 1.
            split_by: 'subject' (split list of dfs) or 'time' (split each df internally).
            random_state: Seed for reproducibility (only for subject split).
        """
        if not (0.0 < val_ratio < 1.0):
            raise ValueError("val_ratio must be between 0 and 1 (exclusive).")

        num_subjects = len(self.all_data)
        if num_subjects == 0:
            raise ValueError("No subjects available in all_data.")

        if split_by == "subject":
            return self._split_by_subject(val_ratio, random_state)
        elif split_by == "time":
            return self._split_by_time_index(val_ratio)
        else:
            raise ValueError("Attribute split_by must be 'subject' or 'time'.")

    # --- Helper Methods ---

    def _run_pipeline(self, pipeline: list[DataProcessor], phase_name: str):
        """Executes a list of processors sequentially."""
        dataset_name = self.config['dataset'].get('name', 'Unnamed')
        print(f"[{dataset_name}] Starting Phase: {phase_name}")

        for processor in pipeline:
            self.all_data = processor.process(self.all_data, self.context)

        print(f"[{dataset_name}] Phase {phase_name} completed.\n")

    def _get_numeric_cols_from_config(self) -> list[str]:
        """Extracts column names designated as numeric in the local config."""
        dtypes = self.config['schema'].get('dtypes', {})
        numeric_prefixes = ("int", "float")
        return [
            col for col, dtype in dtypes.items()
            if isinstance(dtype, str) and dtype.startswith(numeric_prefixes)
        ]

    def _split_by_subject(self, val_ratio: float, random_state: int | None) -> tuple[
        list[pd.DataFrame], list[pd.DataFrame], list[int], list[int]]:
        rng = np.random.default_rng(random_state)
        indices = np.arange(len(self.all_data))
        rng.shuffle(indices)

        num_val = max(1, int(len(self.all_data) * val_ratio))
        val_idx = indices[:num_val]
        train_idx = indices[num_val:]

        train_data = [self.all_data[i] for i in train_idx]
        val_data = [self.all_data[i] for i in val_idx]

        return train_data, val_data, list(map(int, train_idx)), list(map(int, val_idx))

    def _split_by_time_index(self, val_ratio: float) -> tuple[
        list[pd.DataFrame], list[pd.DataFrame], list[int], list[int]]:
        train_data, val_data = [], []
        train_ids, val_ids = [], []

        for df_idx, df in enumerate(self.all_data):
            if df.empty: continue

            # Assuming df is already sorted by TimeIndexer
            n = len(df)
            split_idx = int((1.0 - val_ratio) * n)

            if split_idx <= 0 or split_idx >= n:
                train_data.append(df)
                train_ids.append(df_idx)
                continue

            train_data.append(df.iloc[:split_idx])
            val_data.append(df.iloc[split_idx:])
            train_ids.append(df_idx)
            val_ids.append(df_idx)

        return train_data, val_data, train_ids, val_ids