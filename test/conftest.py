import pandas as pd
import numpy as np
import pytest

from data_prep.base_dataset import BaseDataset
from data_prep.processors.cleaning import TypeAndValueCleaner
from data_prep.processors.augmentation import BaseTimeEventAugmenter


class DummyBaseDataset(BaseDataset):
    """
    Sottoclasse di BaseDataset per i test.
    Simula l'inizializzazione senza caricare file dal disco,
    ma configura correttamente context e pipeline.
    """

    def __init__(self, dfs: list[pd.DataFrame]):
        # 1. Setup Config Mock (Risolve l'AttributeError)
        self.config = {
            "dataset": {"name": "DummyDataset"},
            "sampling": {"target_frequency": "5min"},
            "schema": {
                "defaults": {},
                "col_mapping": {},
                "dtypes": {}
            }
        }

        # Non chiamiamo super().__init__ per evitare I/O
        self.all_data = dfs

        self.target_col = "glucose"
        self.time_col = "timestamp"
        self.time_steps = pd.Timedelta(minutes=5)
        self.logging_dir = None

        self.cols = list(dfs[0].columns)
        self.impulse_cols = ["bolus_total", "carbs"]

        self.numeric_cols = [c for c in self.cols if c != "timestamp"]
        self.defaults = {c: 0.0 for c in self.numeric_cols if c != self.target_col}
        self.max_small_gap = pd.Timedelta("20min")

        # --- Costruzione del CONTEXT ---
        self.context = {
            'config': self.config,
            'global_config': {"options": {}},
            'logging_dir': self.logging_dir,
            'target_col': self.target_col,
            'time_col': self.time_col,
            'numeric_cols': self.numeric_cols,
            'defaults': self.defaults,
            'impulse_cols': self.impulse_cols,
            'time_steps': self.time_steps,
            'max_small_gap': self.max_small_gap,
        }

        # --- Definizione Pipeline ---
        self.cleaning_pipeline = [
            TypeAndValueCleaner(),
        ]
        self.augmentation_pipeline = [
            BaseTimeEventAugmenter()
        ]


@pytest.fixture
def toy_subject_df() -> pd.DataFrame:
    """DataFrame singolo realistico per i test."""
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


@pytest.fixture
def toy_timeseries_df() -> pd.DataFrame:
    """Fixture mancante richiesta dai test di windowing."""
    data = {
        "timestamp": pd.date_range("2025-01-01", periods=6, freq="5min"),
        "glucose": [100, 110, 120, 130, 140, 150],
        "basal_rate": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        "carbs": [0, 10, 0, 0, 20, 0],
    }
    df = pd.DataFrame(data).set_index("timestamp")
    return df


@pytest.fixture
def dummy_dataset(toy_subject_df: pd.DataFrame) -> DummyBaseDataset:
    dfs = [
        toy_subject_df.copy().assign(_subject_id=0),
        toy_subject_df.copy().assign(_subject_id=1),
    ]
    return DummyBaseDataset(dfs)


# --- Helper per creare DF deterministic ---
def _make_subject_df(
        n_rows: int,
        *,
        start: str = "2025-01-01 00:00",
        freq: str = "5min",
        glucose0: float = 100.0,
        basal0: float = 1.0,
) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n_rows, freq=freq)
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "glucose": glucose0 + np.arange(n_rows, dtype=float),
            "basal_rate": basal0 + 0.01 * np.arange(n_rows, dtype=float),
            "bolus_total": np.zeros(n_rows, dtype=float),
            "carbs": np.zeros(n_rows, dtype=float),
        }
    ).set_index("timestamp")
    return df


@pytest.fixture
def dummy_dataset_three_subjects() -> DummyBaseDataset:
    dfs = [
        _make_subject_df(12, glucose0=100.0).assign(_subject_id=0),
        _make_subject_df(12, glucose0=200.0).assign(_subject_id=1),
        _make_subject_df(12, glucose0=300.0).assign(_subject_id=2),
    ]
    return DummyBaseDataset(dfs)


@pytest.fixture
def dummy_two_datasets_for_offset() -> tuple[DummyBaseDataset, DummyBaseDataset]:
    """Fixture mancante richiesta dai test di offset."""
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