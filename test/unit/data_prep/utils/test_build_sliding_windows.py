# tests/unit/test_build_sliding_windows.py

import numpy as np
import pandas as pd
import pytest

from data_prep.utils import build_sliding_windows


def test_build_sliding_windows_basic() -> None:
    """Costruisce finestre sovrapposte con la forma attesa."""
    idx = pd.date_range("2020-01-01 00:00", periods=5, freq="5T")
    df1 = pd.DataFrame(
        {
            "glucose": [1, 2, 3, 4, 5],
            "basal": [0, 0, 0, 0, 0],
        },
        index=idx,
    )

    all_data = [df1, df1]

    X = build_sliding_windows(
        all_data=all_data,
        feature_cols=["glucose", "basal"],
        seq_len=3,
        step=1,
        max_missing_ratio=0.0,
    )

    # Per ogni df, con 5 righe e seq_len=3, step=1, ci sono 3 finestre.
    # Con 2 df otteniamo 6 finestre.
    assert X.shape == (6, 3, 2)
    assert X.dtype == np.float32


def test_build_sliding_windows_missing_features() -> None:
    """Se mancano feature richieste viene sollevato KeyError."""
    df = pd.DataFrame({"glucose": [1, 2, 3]})
    with pytest.raises(KeyError):
        build_sliding_windows(
            all_data=[df],
            feature_cols=["glucose", "basal"],
            seq_len=2,
            step=1,
        )


def test_build_sliding_windows_filters_missing_windows() -> None:
    """Finestre con troppi NaN vengono scartate in base a max_missing_ratio."""
    df = pd.DataFrame(
        {
            "glucose": [1.0, np.nan, np.nan, 4.0],
        }
    )
    # seq_len=2, step=1 produce finestre:
    # [1, nan], [nan, nan], [nan, 4]
    # missing_ratio per finestra: 0.5, 1.0, 0.5
    X = build_sliding_windows(
        all_data=[df],
        feature_cols=["glucose"],
        seq_len=2,
        step=1,
        max_missing_ratio=0.5,
    )
    # La finestra con 1.0 di NaN viene scartata, ne restano 2
    assert X.shape == (2, 2, 1)


def test_build_sliding_windows_invalid_parameters() -> None:
    """Parametri non validi producono ValueError."""
    df = pd.DataFrame({"x": [1, 2, 3]})

    with pytest.raises(ValueError):
        build_sliding_windows([df], feature_cols=["x"], seq_len=0, step=1)

    with pytest.raises(ValueError):
        build_sliding_windows([df], feature_cols=["x"], seq_len=2, step=0)

    with pytest.raises(ValueError):
        build_sliding_windows([df], feature_cols=["x"], seq_len=2, step=1, max_missing_ratio=1.5)
