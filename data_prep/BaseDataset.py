from pathlib import Path

import pandas as pd
import numpy as np
import yaml

from .utils import load_dataset, load_dataset_config, clean_duplicates, fill_data, \
    print_df_summary, print_duplicate_counts, \
    build_sliding_windows, add_time_of_day_features, add_exponential_decay_feature, build_sliding_windows_conditional

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

    added_cols = []


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
        self._rename_cols()

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
        self._set_time_index()

        self._ensure_cols()

        # Setting logging path
        self.logging_dir = logging_dir

        if self.logging_dir is not None:
            self.logging_dir.mkdir(parents=True, exist_ok=True)

        print_df_summary(self.all_data, self.logging_dir / "init_summary.txt")

    # function that cleans the dataset
    def _rename_cols(self):
        inverse_mapping = {raw: standard for standard, raw in self.col_mapping.items()}
        for i, df in enumerate(self.all_data):

            df = df.rename(columns=lambda c: c.strip())  # pulisce spazi
            df = df.rename(columns=inverse_mapping)

            self.all_data[i] = df

    def _set_time_index(self):
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

    def _ensure_cols(self):
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
        self._clean_cols()

        # Safety check
        if self.logging_dir is not None:
            self.logging_dir.mkdir(parents=True, exist_ok=True)

        # Log and remove duplicate timestamps
        print_duplicate_counts(self.all_data, self.logging_dir / "duplicate_counts.txt")
        self.all_data = clean_duplicates(self.all_data)

        # Log and fill target gaps (interpolate gaps < max_small_gap)
        self.all_data = fill_data(self.all_data, self.time_steps, self.max_small_gap, self.target_col, self.defaults, self.logging_dir / "gaps")

        # Post-cleaning summary
        print_df_summary(self.all_data, self.logging_dir / "post-cleaning.txt")
    # function that sets the time index for the df

    # function that cleans the cols
    def _clean_cols(self):
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

    def add_features(self) -> list[str]:
        """
        Add basic engineered features to each subject DataFrame.

        Currently, this includes:
          - time-of-day sinusoidal features (24h, and optionally 12h if enabled
            inside add_time_of_day_features),
          - exponential decay features for each column in self.impulse_cols.

        Features generated can be changed by overriding the _add_features_to_df function.

        If self.logging_dir is not None, log the names of the added columns
        for each subject to 'added_columns.txt' inside that directory.
        """

        # Optional logging setup
        log_file = None
        if self.logging_dir is not None:
            self.logging_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.logging_dir / "added_columns.txt"
            # Append mode so multiple runs do not overwrite previous logs
            log_file = log_path.open("w", encoding="utf-8")

        try:
            for i, df in enumerate(self.all_data):

                # Update the stored DataFrame
                self.all_data[i], cols = self._add_features_to_df(df)

                # Console feedback
                print(f"Subject {i + 1}: added {len(cols)} columns")

                # Keep a global list of all distinct added columns
                for c in cols:
                    if c not in self.added_cols:
                        self.added_cols.append(c)

                # Optional logging to file
                if log_file is not None:
                    print(f"Subject {i + 1}:", file=log_file)
                    for c in cols:
                        print(f"  - {c}", file=log_file)
                    print("", file=log_file)  # blank line for readability

        finally:
            # Make sure we close the log file if it was opened
            if log_file is not None:
                log_file.close()

        self.cols.extend(self.added_cols)
        return self.added_cols.copy()

    def _add_features_to_df(
            self,
            df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Add basic engineered features to a single subject DataFrame.
        Designed to be overridden by subclasses that need custom features.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame indexed by time.

        Returns
        -------
        df_out : pd.DataFrame
            DataFrame with added time-of-day and exponential-decay features.
        added_cols : list[str]
            Names of the newly created feature columns.
        """
        cols: list[str] = []

        # Time-of-day features (use index as datetime)
        df, tod_cols = add_time_of_day_features(df)
        cols.extend(tod_cols)

        # Exponential decay features for each impulse-like column
        for col in self.impulse_cols:
            df, col_name = add_exponential_decay_feature(
                df,
                col=col,
                sampling_minutes=int(self.time_steps.total_seconds() / 60),
                halflife_min=120,
                past_only=True,
            )
            cols.append(col_name)

        return df, cols


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

        if not val_ratio <= 1 or val_ratio >= 0:
            raise ValueError("val_ratio must be between 0 and 1.")

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

    def to_sequence_splits_conditional(
            self,
            seq_len: int,
            step: int,
            cond_cols: list[str],
            val_ratio: float = 0.2,
            split_by: str = "subject",
            max_missing_ratio: float = 0.0,
            random_state: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build sliding windows for target and conditioning features, then split them
        into train and validation sets.

        This behaves like `to_sequence_splits`, but:
          - uses `self.target_col` as target,
          - uses `cond_cols` for conditioning features,
          - returns 4 arrays: (X_train_y, X_train_c, X_val_y, X_val_c).

        Parameters
        ----------
        seq_len : int
            Length of each sliding window (number of time steps).
        step : int
            Stride between consecutive windows.
        cond_cols : list[str]
            Columns to use as conditioning features.
        val_ratio : float, optional
            Fraction of data to reserve for validation, by default 0.2.
        split_by : str, optional
            Split strategy: "subject" or "time", by default "subject".
        max_missing_ratio : float, optional
            Maximum allowed fraction of missing values in a window,
            passed to `build_sliding_windows_conditional`, by default 0.0.
        random_state : int | None, optional
            Random seed for subject-level split, by default None.

        Returns
        -------
        X_train_y : np.ndarray
            Training target windows, shape (N_train, seq_len, 1).
        X_train_c : np.ndarray
            Training conditioning windows, shape (N_train, seq_len, cond_dim).
        X_val_y : np.ndarray
            Validation target windows, shape (N_val, seq_len, 1).
        X_val_c : np.ndarray
            Validation conditioning windows, shape (N_val, seq_len, cond_dim).
        """
        if not (0.0 < val_ratio < 1.0):
            raise ValueError("val_ratio must be between 0 and 1 (exclusive).")

        num_subjects = len(self.all_data)
        if num_subjects == 0:
            raise ValueError("No subjects available in all_data.")

        # Reuse the same splitting logic as to_sequence_splits
        if split_by == "subject":
            train_data, val_data = self._split_by_subject(
                val_ratio=val_ratio,
                random_state=random_state,
            )
        elif split_by == "time":
            train_data, val_data = self._split_by_time_index(
                val_ratio=val_ratio,
            )
        else:
            raise ValueError("Attribute split_by must be 'subject' or 'time'.")

        # Build conditional windows for train split
        X_train_y, X_train_c = build_sliding_windows_conditional(
            all_data=train_data,
            target_col=self.target_col,
            cond_cols=cond_cols,
            seq_len=seq_len,
            step=step,
            max_missing_ratio=max_missing_ratio,
        )

        # Build conditional windows for validation split
        X_val_y, X_val_c = build_sliding_windows_conditional(
            all_data=val_data,
            target_col=self.target_col,
            cond_cols=cond_cols,
            seq_len=seq_len,
            step=step,
            max_missing_ratio=max_missing_ratio,
        )

        return X_train_y, X_train_c, X_val_y, X_val_c

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