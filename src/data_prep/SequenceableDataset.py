import numpy as np
import pandas as pd

from .BaseDataset import BaseDataset
from .utils import build_sliding_windows, build_sliding_windows_conditional


class SequenceableDataset(BaseDataset):
    def to_sequence_splits(
            self,
            feature_cols: list[str],
            train_seq_len: int,
            train_step: int,
            val_seq_len: int | None = None,
            val_step: int | None = None,
            val_ratio: float = 0.2,
            split_by: str = "subject",
            max_missing_ratio: float = 0.0,
            random_state: int | None = None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        list[tuple[int, int]],
        list[tuple[int, int]]
    ]:

        if not (0.0 < val_ratio < 1.0):
            raise ValueError("val_ratio must be between 0 and 1.")

        if val_seq_len is None: val_seq_len = train_seq_len
        if val_step is None: val_step = train_step

        num_subjects = len(self.all_data)
        if num_subjects == 0:
            raise ValueError("No subjects available in all_data.")

        # Random split at subject level
        if split_by == "subject":
            train_data, val_data, train_subject_ids, val_subject_ids= self._split_by_subject(
                val_ratio,
                random_state,
            )
        elif split_by == "time":
            train_data, val_data, train_subject_ids, val_subject_ids = self._split_by_time_index(
                val_ratio,
            )

        else:
            raise ValueError("Attribute split_by must be 'subject' or 'time'.")

        # Build sequences for each split using the existing to_sequences
        X_train, train_metadata = build_sliding_windows(
            all_data = train_data,
            feature_cols=feature_cols,
            seq_len=train_seq_len,
            step=train_step,
            ids=train_subject_ids,
            max_missing_ratio=max_missing_ratio,
        )

        X_val, val_metadata = build_sliding_windows(
            all_data=val_data,
            feature_cols=feature_cols,
            seq_len=val_seq_len,
            step=val_step,
            ids=val_subject_ids,
            max_missing_ratio=max_missing_ratio,
        )

        return X_train, X_val, train_metadata, val_metadata

    def to_sequence_splits_conditional(
            self,
            train_seq_len: int,
            train_step: int,
            cond_cols: list[str],
            val_seq_len: int | None = None,
            val_step: int | None = None,
            val_ratio: float = 0.2,
            split_by: str = "subject",
            max_missing_ratio: float = 0.0,
            random_state: int | None = None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[tuple[int, int]],
        list[tuple[int, int]],
    ]:
        """
        Build sliding windows for target and conditioning features, then split them
        into train and validation sets.

        This behaves like `to_sequence_splits`, but:
          - uses `self.target_col` as target,
          - uses `cond_cols` for conditioning features,
          - returns 4 arrays: (X_train_y, X_train_c, X_val_y, X_val_c).

        Parameters
        ----------
        train_seq_len : int
            Length of each sliding window (number of time steps) in train split.
        train_step : int
            Stride between consecutive windows in train split.
        cond_cols : list[str]
            Columns to use as conditioning features.
        val_seq_len : int, optional
            Length of each sliding window (number of time steps) in val split.
            If none, train_seq_len is used.
        val_step : int
            Stride between consecutive windows in val split.
            If none, train_step is used.
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

        if val_seq_len is None: val_seq_len = train_seq_len
        if val_step is None: val_step = train_step

        num_subjects = len(self.all_data)
        if num_subjects == 0:
            raise ValueError("No subjects available in all_data.")

        # Reuse the same splitting logic as to_sequence_splits
        if split_by == "subject":
            train_data, val_data, train_subject_ids, val_subject_ids = self._split_by_subject(
                val_ratio=val_ratio,
                random_state=random_state,
            )

        elif split_by == "time":
            train_data, val_data, train_subject_ids, val_subject_ids = self._split_by_time_index(
                val_ratio=val_ratio,
            )

        else:
            raise ValueError("Attribute split_by must be 'subject' or 'time'.")

        # Build conditional windows for train split
        X_train_y, X_train_c, train_metadata = build_sliding_windows_conditional(
            all_data=train_data,
            target_col=self.target_col,
            cond_cols=cond_cols,
            seq_len=train_seq_len,
            step=train_step,
            ids = train_subject_ids,
            max_missing_ratio=max_missing_ratio,
        )

        # Build conditional windows for validation split
        X_val_y, X_val_c, val_metadata = build_sliding_windows_conditional(
            all_data=val_data,
            target_col=self.target_col,
            cond_cols=cond_cols,
            seq_len=val_seq_len,
            step=val_step,
            ids = val_subject_ids,
            max_missing_ratio=max_missing_ratio,
        )

        return X_train_y, X_train_c, X_val_y, X_val_c, train_metadata, val_metadata


    def _split_by_subject(
            self,
            val_ratio: float,
            random_state: int | None = None,
    ) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[int], list[int]]:
        """
        Split self.all_data at subject level into train/validation sets.
        """
        if not 0.0 < val_ratio < 1.0:
            raise ValueError("val_ratio must be between 0 and 1 (exclusive).")

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

        return train_data, val_data, list(map(int, train_idx)), list(map(int, val_idx))


    def _split_by_time_index(
            self,
            val_ratio: float,
    ) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[int], list[int]]:
        """
        Split each subject's DataFrame into train/validation segments along
        the time axis, using the DatetimeIndex ordering.
        """
        if not 0.0 < val_ratio < 1.0:
            raise ValueError("val_ratio must be between 0 and 1 (exclusive).")

        train_data: list[pd.DataFrame] = []
        val_data: list[pd.DataFrame] = []
        train_ids: list[int] = []
        val_ids: list[int] = []

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
                train_ids.append(df_idx)
                continue

            # iloc slicing usa la posizione, coerente con l'ordine temporale
            train_df = df_sorted.iloc[:split_idx]
            val_df = df_sorted.iloc[split_idx:]

            train_data.append(train_df)
            val_data.append(val_df)

            train_ids.append(df_idx)
            val_ids.append(df_idx)


        return train_data, val_data, train_ids, val_ids