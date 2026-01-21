from pathlib import Path

import pandas as pd
import numpy as np
import yaml

from data_prep import print_df_summary, print_duplicate_counts, build_sliding_windows
from data_prep.utils import load_dataset_config
from .utils import load_dataset, clean_duplicates, fill_data


NUMERIC_PREFIXES = ("int", "float")
GLOBAL_CONFIG_PATH = "global_config.yaml"

class BaseDataset:
    """
    BaseDataset class that defines a fixed and overridable pipeline for data preparation of glucose time series.
    Column management is almost entirely based on config files, which are used to define basic, task-specific, operations.
    """

    # list of time series
    all_data : list[pd.DataFrame]

    # path for logs
    logging_dir : Path

    # dataset config information
    global_config : dict
    config : dict
    # time steps of the series
    time_steps : pd.Timedelta


    time_col : str # timestamp col
    target_col : str # target column
    cols : list[str] # other cols
    col_mapping : dict # mapping to standard
    numeric_cols : list[str] # numeric columns
    impulse_cols : list[str] # numeric columns which register events
    category_cols : list[str] # categoric columns


    defaults : dict
    max_small_gap : pd.Timedelta

    # init that loads the dataset and the config parameters
    def __init__(
            self,
            dataset_root : Path,
            config_file : Path,
            global_config_file : Path | None = GLOBAL_CONFIG_PATH,
            logging_dir : Path | None = None
    ):
        """
        Initialization includes dataset loading and dataset and global configs loading
        :param dataset_root: root path of dataset
        :param config_file: config file path
        :param global_config_file: global config file path (if None "global_config.yaml" will be expected from project root)
        :param logging_dir: optional path to logging directory
        """

        # Loading dataset and global configs
        self.config = load_dataset_config(config_file)
        self.global_config = yaml.safe_load(open(global_config_file))

        # Loading data
        self.all_data = load_dataset(
            dataset_root,
            self.config["dataset"].get("separator", ",")
        )

        # Getting references
        options = self.global_config["options"]

        self.max_small_gap = options.get("max_small_gap", pd.Timedelta("5min"))

        self.time_steps = pd.Timedelta(self.config["sampling"]["target_frequency"])

        global_schema = self.global_config["schema"]

        dataset_schema = self.config["schema"]

        self.col_mapping = dataset_schema["col_mapping"]

        # Renaming columns to global schema
        self.rename_cols()

        self.target_col = global_schema["target_col"]

        self.time_col = global_schema["time_col"]

        self.cols = list(col_name for col_name in dataset_schema.get("col_mapping", {}).keys() if col_name != self.time_col)

        self.numeric_cols = [
            col_name
            for col_name, dtype_str in dataset_schema["dtypes"].items()
            if isinstance(dtype_str, str) and dtype_str.startswith(NUMERIC_PREFIXES)
        ]

        self.impulse_cols = [
            col_name
            for col_name in self.cols
            if col_name in global_schema.get("impulse_cols", [])
        ]

        self.category_cols = [
            col_name
            for col_name in self.cols
            if col_name in global_schema.get("category_cols", [])
        ]

        self.defaults = dataset_schema["defaults"]

        # Setting time index
        self.set_time_index()

        self.ensure_cols()

        # Setting logging path
        self.logging_dir = logging_dir

        if self.logging_dir is not None:
            self.logging_dir.mkdir(parents=True, exist_ok=True)

        print_df_summary(self.all_data, self.logging_dir / "init_summary.txt")

    # function that cleans the dataset
    def rename_cols(self):
        inverse_mapping = {raw: standard for standard, raw in self.col_mapping.items()}
        for i, df in enumerate(self.all_data):

            df = df.rename(columns=lambda c: c.strip())  # pulisce spazi
            df = df.rename(columns=inverse_mapping)

            self.all_data[i] = df

    def set_time_index(self):
        """
        Function to set the time index of the dataset.
        Time column is referred to as "time_col".
        :return:
        """
        for i in range(len(self.all_data)):
            df = self.all_data[i]
            df[self.time_col] = pd.to_datetime(df[self.time_col], errors="coerce", utc=True)
            df = df.set_index(self.time_col).sort_index()
            self.all_data[i] = df

    def ensure_cols(self):
        """
        Ensure that all and only known columns are present in the dataset.
        """
        print(f"[{self.config['dataset'].get("name", "")}] forced columns: {self.cols}")
        for i, df in enumerate(self.all_data):
            df = df[self.cols]
            self.all_data[i] = df

    def clean_data(self):
        """
        Function to start the pipeline for data cleaning.
        Functions called are:
            - clean_cols()
            - clean_duplicates()
            - fill_data()
        Logging of the process will be saved in the logging directory if defined.
        """

        # Clean columns
        self.clean_cols()

        # Log and remove duplicate timestamps
        print_duplicate_counts(self.all_data, self.logging_dir / "duplicate_counts.txt")
        self.all_data = clean_duplicates(self.all_data)

        # Log and fill target gaps (interpolate gaps < max_small_gap)
        self.all_data = fill_data(self.all_data, self.time_steps, self.max_small_gap, self.target_col, self.defaults, self.logging_dir / "gaps")

        # Post-cleaning summary
        print_df_summary(self.all_data, self.logging_dir / "post-cleaning.txt")
    # function that sets the time index for the df

    # function that cleans the cols
    def clean_cols(self):
        """
        Core of the data cleaning process.
        This function applies basic data cleaning techniques on the dfs:
            - typing of the columns
            - filling missing values with default values
            - clip impulse_cols to values > 0

        This function is overridable to customize the behavior of the pipeline for data cleaning.
        """
        for i, df in enumerate(self.all_data):
            # Coerce numeric columns
            for c in self.numeric_cols:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            for key, value in self.defaults.items():
                df[key] = df[key].fillna(value)

            df[self.impulse_cols] = df[self.impulse_cols].clip(lower=0.0)

            self.all_data[i] = df

    def to_sequences(
            self,
            feature_cols: list[str],
            seq_len: int,
            step: int,
            max_missing_ratio: float = 0.0,
    ):

        """
        Convert cleaned patient DataFrames into fixed-length sliding window sequences.

        This method builds input sequences for time-series models (e.g. VAE) by
        applying a sliding window over the data of each subject.

        Parameters
        ----------
        seq_len : int
            Length of each window (number of time steps).
        step : int
            Sliding window stride (number of rows between consecutive windows).
        feature_cols : list of str, optional
            Names of feature columns to include in each window. If None, the
            dataset's default feature columns are used.
        max_missing_ratio : float, optional
            Maximum allowed fraction of NaN values inside a window (between 0 and 1).
            Windows with a higher missing ratio are discarded. Default is 0.0.

        Returns
        -------
        np.ndarray
            Array of shape (num_windows, seq_len, num_features) containing all
            valid windows stacked along the first dimension. The dtype is float32.
        """

        return build_sliding_windows(
            self.all_data,
            feature_cols,
            seq_len,
            step,
            max_missing_ratio
        )

    def to_sequence_splits(
            self,
            seq_len: int,
            step: int,
            feature_cols: list[str] | None = None,
            val_ratio: float = 0.2,
            split_by: str = "subject",
            max_missing_ratio: float = 0.0,
            random_state: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:

        num_subjects = len(self.all_data)
        if num_subjects == 0:
            raise ValueError("No subjects available in all_data.")

        # Random split at subject level
        if split_by == "subject":
            train_data, val_data = self._split_by_subject(
                val_ratio,
                random_state,
            )
        elif split_by == "time":
            train_data, val_data = self._split_by_time_index(
                val_ratio,
            )
        else:
            raise ValueError("Attribute split_by must be 'subject' or 'time'.")

        # Build sequences for each split using the existing to_sequences
        X_train = build_sliding_windows(
            train_data,
            feature_cols,
            seq_len,
            step,
            max_missing_ratio
        )

        X_val = build_sliding_windows(
            val_data,
            feature_cols,
            seq_len,
            step,
            max_missing_ratio
        )

        return X_train, X_val


    def _split_by_subject(
            self,
            val_ratio: float,
            random_state: int | None = None,
    ) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
        """
        Split self.all_data at subject level into train/validation sets.
        """
        all_data = self.all_data
        num_subjects = len(all_data)
        if num_subjects == 0:
            raise ValueError("No subjects available in all_data.")

        rng = np.random.default_rng(random_state)
        indices = np.arange(num_subjects)
        rng.shuffle(indices)

        num_val = max(1, int(num_subjects * val_ratio))
        val_idx = indices[:num_val]
        train_idx = indices[num_val:]

        train_data = [all_data[i] for i in train_idx]
        val_data = [all_data[i] for i in val_idx]

        return train_data, val_data


    def _split_by_time_index(
            self,
            val_ratio: float,
    ) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
        """
        Split each subject's DataFrame into train/validation segments along
        the time axis, using the DatetimeIndex ordering.
        """
        if not 0.0 < val_ratio < 1.0:
            raise ValueError("val_ratio must be between 0 and 1 (exclusive).")

        train_data: list[pd.DataFrame] = []
        val_data: list[pd.DataFrame] = []

        for df_idx, df in enumerate(self.all_data):
            if df.empty:
                continue

            # Ensure chronological order by index (DatetimeIndex)
            df_sorted = df.sort_index()

            n = len(df_sorted)
            split_idx = int((1.0 - val_ratio) * n)

            # If split is degenerate, keep everything in train
            if split_idx <= 0 or split_idx >= n:
                train_data.append(df_sorted)
                continue

            # iloc slicing usa la posizione, coerente con l'ordine temporale
            train_df = df_sorted.iloc[:split_idx]
            val_df = df_sorted.iloc[split_idx:]

            train_data.append(train_df)
            val_data.append(val_df)

        return train_data, val_data