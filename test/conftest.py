# tests/conftest.py
import pandas as pd
import numpy as np
import pytest


import pandas as pd
import pytest

from data_prep import BaseDataset  # la tua classe vera


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

@pytest.fixture
def dummy_dataset(toy_subject_df: pd.DataFrame) -> DummyBaseDataset:
    """Istanza di DummyBaseDataset riutilizzabile in tutti i test."""
    dfs = [toy_subject_df, toy_subject_df]
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
