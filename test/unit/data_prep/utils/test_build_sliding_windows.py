# tests/unit/test_build_sliding_windows.py

import numpy as np
import pandas as pd
import pytest

from data_prep.utils import build_sliding_windows


def test_build_sliding_windows_basic() -> None:
    """Builds overlapping windows with the expected shape and dtype."""
    idx = pd.date_range("2020-01-01 00:00", periods=5, freq="5T")
    df1 = pd.DataFrame(
        {
            "glucose": [1, 2, 3, 4, 5],
            "basal": [0, 0, 0, 0, 0],
        },
        index=idx,
    )

    all_data = [df1, df1]

    X, metadata = build_sliding_windows(
        all_data=all_data,
        feature_cols=["glucose", "basal"],
        seq_len=3,
        step=1,
        max_missing_ratio=0.0,
    )

    # For each df, with 5 rows, seq_len=3, step=1, we get 3 windows.
    # With 2 dfs we obtain 6 windows.
    assert X.shape == (6, 3, 2)
    assert X.dtype == np.float32

    # Metadata length matches number of windows
    assert len(metadata) == 6

    # Without explicit ids, subject ids should be 0 and 1
    subject_ids = {m[0] for m in metadata}
    assert subject_ids == {0, 1}


def test_build_sliding_windows_with_explicit_ids() -> None:
    """Explicit subject IDs are propagated into metadata."""
    df = pd.DataFrame(
        {
            "glucose": [1, 2, 3, 4],
        }
    )

    all_data = [df, df]
    ids = [10, 20]

    X, metadata = build_sliding_windows(
        all_data=all_data,
        feature_cols=["glucose"],
        seq_len=2,
        step=1,
        ids=ids,
        max_missing_ratio=0.0,
    )

    # Each df: 4 rows, seq_len=2, step=1 => 3 windows per subject => 6 total
    assert X.shape == (6, 2, 1)
    assert len(metadata) == 6

    subject_ids = {m[0] for m in metadata}
    assert subject_ids == {10, 20}


def test_build_sliding_windows_missing_features() -> None:
    """Missing required feature columns raises KeyError."""
    df = pd.DataFrame({"glucose": [1, 2, 3]})
    with pytest.raises(KeyError):
        _ = build_sliding_windows(
            all_data=[df],
            feature_cols=["glucose", "basal"],
            seq_len=2,
            step=1,
        )


def test_build_sliding_windows_filters_missing_windows() -> None:
    """Windows with too many NaNs are discarded according to max_missing_ratio."""
    df = pd.DataFrame(
        {
            "glucose": [1.0, np.nan, np.nan, 4.0],
        }
    )
    # seq_len=2, step=1 produces windows:
    # [1, nan], [nan, nan], [nan, 4]
    # missing_ratio per window: 0.5, 1.0, 0.5
    X, metadata = build_sliding_windows(
        all_data=[df],
        feature_cols=["glucose"],
        seq_len=2,
        step=1,
        max_missing_ratio=0.5,
    )
    # The window with missing_ratio=1.0 is discarded, 2 remain
    assert X.shape == (2, 2, 1)
    assert len(metadata) == 2

    # Start indices should be 0 and 2
    starts = sorted(m[1] for m in metadata)
    assert starts == [0, 2]


def test_build_sliding_windows_invalid_parameters() -> None:
    """Invalid parameters produce ValueError."""
    df = pd.DataFrame({"x": [1, 2, 3]})

    with pytest.raises(ValueError):
        _ = build_sliding_windows([df], feature_cols=["x"], seq_len=0, step=1)

    with pytest.raises(ValueError):
        _ = build_sliding_windows([df], feature_cols=["x"], seq_len=2, step=0)

    with pytest.raises(ValueError):
        _ = build_sliding_windows(
            [df],
            feature_cols=["x"],
            seq_len=2,
            step=1,
            max_missing_ratio=1.5,
        )
