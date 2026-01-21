import numpy as np
import pandas as pd
import pytest

from windowing.utils import build_sliding_windows_conditional


def test_build_sliding_windows_conditional_basic() -> None:
    """Windows are correctly split into target and conditioning arrays."""
    idx = pd.date_range("2020-01-01 00:00", periods=4, freq="5T")
    df = pd.DataFrame(
        {
            "glucose": [1.0, 2.0, 3.0, 4.0],
            "basal": [0.1, 0.2, 0.3, 0.4],
            "carbs": [0.0, 10.0, 0.0, 0.0],
        },
        index=idx,
    )

    all_data = [df]

    X_target, X_cond, metadata = build_sliding_windows_conditional(
        all_data=all_data,
        seq_len=2,
        step=1,
        target_col="glucose",
        cond_cols=["basal", "carbs"],
        max_missing_ratio=0.0,
    )

    # With 4 rows, seq_len=2, step=1 => 3 windows
    assert X_target.shape == (3, 2, 1)
    assert X_cond.shape == (3, 2, 2)
    assert len(metadata) == 3

    # Target of first window corresponds to glucose[0:2]
    expected_first_window_target = df["glucose"].iloc[0:2].to_numpy()
    assert np.allclose(X_target[0, :, 0], expected_first_window_target)

    # Cond of first window corresponds to [basal, carbs][0:2]
    expected_first_window_cond = df[["basal", "carbs"]].iloc[0:2].to_numpy()
    assert np.allclose(X_cond[0], expected_first_window_cond)

    # Only one subject and no explicit ids => all global ids should be 0
    assert {m[0] for m in metadata} == {0}


def test_build_sliding_windows_conditional_missing_columns() -> None:
    """Missing required columns raises KeyError."""
    df = pd.DataFrame({"glucose": [1, 2, 3]})
    with pytest.raises(KeyError):
        _ = build_sliding_windows_conditional(
            all_data=[df],
            seq_len=2,
            step=1,
            target_col="glucose",
            cond_cols=["basal"],
        )
