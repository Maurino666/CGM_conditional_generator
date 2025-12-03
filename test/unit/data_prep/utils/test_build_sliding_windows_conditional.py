# tests/unit/test_build_sliding_windows_conditional.py

import numpy as np
import pandas as pd
import pytest

from data_prep.utils import build_sliding_windows_conditional


def test_build_sliding_windows_conditional_basic() -> None:
    """Le finestre vengono separate correttamente in target e condizionamento."""
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

    X_target, X_cond = build_sliding_windows_conditional(
        all_data=all_data,
        seq_len=2,
        step=1,
        target_col="glucose",
        cond_cols=["basal", "carbs"],
        max_missing_ratio=0.0,
    )

    # Con 4 righe, seq_len=2, step=1 => 3 finestre
    assert X_target.shape == (3, 2, 1)
    assert X_cond.shape == (3, 2, 2)

    # Controllo che il target corrisponda alla prima feature
    expected_first_window_target = df["glucose"].iloc[0:2].to_numpy()
    assert np.allclose(X_target[0, :, 0], expected_first_window_target)

    # Controllo che le condizioni corrispondano alle altre colonne
    expected_first_window_cond = df[["basal", "carbs"]].iloc[0:2].to_numpy()
    assert np.allclose(X_cond[0], expected_first_window_cond)


def test_build_sliding_windows_conditional_missing_columns() -> None:
    """Se mancano colonne richieste viene sollevato KeyError."""
    df = pd.DataFrame({"glucose": [1, 2, 3]})
    with pytest.raises(KeyError):
        build_sliding_windows_conditional(
            all_data=[df],
            seq_len=2,
            step=1,
            target_col="glucose",
            cond_cols=["basal"],
        )
