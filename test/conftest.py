# tests/conftest.py
import pandas as pd
import numpy as np
import pytest


import pandas as pd
import pytest

from data_prep import BaseDataset


class DummyBaseDataset(BaseDataset):
    """Sottoclasse minimale di BaseDataset per i test."""

    def __init__(self, dfs: list[pd.DataFrame]):
        # NON chiamo super().__init__ per evitare I/O/config veri
        self.all_data = dfs
        self.target_col = "glucose"
        self.time_col = "timestamp"
        self.time_steps = pd.Timedelta(minutes=5)
        self.impulse_cols = ["bolus_total", "carbs"]
        self.added_cols: list[str] = []
        self.logging_dir = None
        self.cols = list(dfs[0].columns)
        self.numeric_cols = list(self.cols)
        self.defaults = {c: 0.0 for c in self.numeric_cols if c != self.target_col}
        self.max_small_gap = pd.Timedelta("20min")

    def split(
            self,
            *,
            val_ratio: float,
            split_by: str = "subject",
            random_state: int | None = None,
    ) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[int], list[int]]:
        """
        Split in list[df] space and return:
          (train_dfs, val_dfs, train_ids, val_ids)

        This is used by the windowing builder layer.
        It is additive and should not break existing tests.
        """
        if not 0.0 < val_ratio < 1.0:
            raise ValueError("val_ratio must be between 0 and 1 (exclusive).")

        n = len(self.all_data)
        if n == 0:
            return [], [], [], []

        if split_by == "subject":
            # Deterministic split with RNG (stable across runs)
            rng = np.random.default_rng(random_state)
            idx = np.arange(n)
            rng.shuffle(idx)

            num_val = max(1, int(n * val_ratio))
            val_idx = idx[:num_val]
            train_idx = idx[num_val:]

            train_dfs = [self.all_data[i] for i in train_idx]
            val_dfs = [self.all_data[i] for i in val_idx]

            return train_dfs, val_dfs, list(map(int, train_idx)), list(map(int, val_idx))

        if split_by == "time":
            # Per-subject time split; ids are the original subject indices
            train_dfs: list[pd.DataFrame] = []
            val_dfs: list[pd.DataFrame] = []
            train_ids: list[int] = []
            val_ids: list[int] = []

            for df_idx, df in enumerate(self.all_data):
                if df.empty:
                    continue
                df_sorted = df.sort_index()
                split_idx = int((1.0 - val_ratio) * len(df_sorted))

                # Degenerate split: keep everything in train
                if split_idx <= 0 or split_idx >= len(df_sorted):
                    train_dfs.append(df_sorted)
                    train_ids.append(int(df_idx))
                    continue

                train_dfs.append(df_sorted.iloc[:split_idx])
                val_dfs.append(df_sorted.iloc[split_idx:])
                train_ids.append(int(df_idx))
                val_ids.append(int(df_idx))

            return train_dfs, val_dfs, train_ids, val_ids

        raise ValueError("split_by must be 'subject' or 'time'.")


@pytest.fixture
def dummy_dataset(toy_subject_df: pd.DataFrame) -> DummyBaseDataset:
    """Istanza di DummyBaseDataset riutilizzabile in tutti i test."""
    dfs = [
        toy_subject_df.assign(_subject_id=0),
        toy_subject_df.assign(_subject_id=1),
    ]
    return DummyBaseDataset(dfs)

@pytest.fixture
def toy_timeseries_df() -> pd.DataFrame:
    """Small single-subject DataFrame for windowing tests."""
    data = {
        "timestamp": pd.date_range("2025-01-01", periods=6, freq="5min"),
        "glucose":   [100, 110, 120, 130, 140, 150],
        "basal_rate": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        "carbs":      [0, 10, 0, 0, 20, 0],
    }
    df = pd.DataFrame(data).set_index("timestamp")
    return df

@pytest.fixture
def toy_subject_df() -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=12, freq="5min")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "glucose": [100, 105, 110, 120, 130, 140, 150, 160, 165, 170, 175, 180],
            "basal_rate": [1.0] * 12,
            "bolus_total": [0, 2, 0, 0, 3, 0, 0, 1, 0, 0, 0, 0],
            "carbs": [0, 0, 10, 0, 0, 0, 20, 0, 0, 0, 0, 0],
        }
    )
    return df.set_index("timestamp")


def _make_subject_df(
    n_rows: int,
    *,
    start: str = "2025-01-01 00:00",
    freq: str = "5min",
    glucose0: float = 100.0,
    basal0: float = 1.0,
) -> pd.DataFrame:
    """Create a single-subject df with deterministic values."""
    idx = pd.date_range(start, periods=n_rows, freq=freq)
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "glucose": glucose0 + np.arange(n_rows, dtype=float),
            "basal_rate": basal0 + 0.01 * np.arange(n_rows, dtype=float),
            # Keep impulse cols available (even if unused by builder)
            "bolus_total": np.zeros(n_rows, dtype=float),
            "carbs": np.zeros(n_rows, dtype=float),
        }
    ).set_index("timestamp")
    return df


@pytest.fixture
def dummy_dataset_three_subjects() -> DummyBaseDataset:
    """3-subject dataset useful for subject split tests (offset, templates)."""
    dfs = [
        _make_subject_df(12, glucose0=100.0).assign(_subject_id=0),
        _make_subject_df(12, glucose0=200.0).assign(_subject_id=1),
        _make_subject_df(12, glucose0=300.0).assign(_subject_id=2),
    ]
    return DummyBaseDataset(dfs)


@pytest.fixture
def dummy_two_datasets_for_offset() -> tuple[DummyBaseDataset, DummyBaseDataset]:
    """
    Two datasets with different number of subjects, for testing global id offsets.
    """
    ds_a = DummyBaseDataset(
        [
            _make_subject_df(10, glucose0=100.0).assign(_subject_id=0),
            _make_subject_df(10, glucose0=200.0).assign(_subject_id=1),
        ]
    )
    ds_b = DummyBaseDataset(
        [
            _make_subject_df(10, glucose0=300.0).assign(_subject_id=0),
            _make_subject_df(10, glucose0=400.0).assign(_subject_id=1),
            _make_subject_df(10, glucose0=500.0).assign(_subject_id=2),
        ]
    )
    return ds_a, ds_b
