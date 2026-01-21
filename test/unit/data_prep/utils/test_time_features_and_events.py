# tests/unit/test_time_features_and_events.py

import numpy as np
import pandas as pd
import pytest

from data_prep.utils import (
    add_target_lags,
    add_time_of_day_features,
    add_event_present_indicator,
    add_exponential_decay_feature,
)


def test_add_target_lags_basic() -> None:
    """add_target_lags aggiunge colonne lag corrette e shiftate."""
    df = pd.DataFrame({"y": [10, 20, 30, 40, 50]})
    # sampling 5 min, lag 10 min => shift di 2 step
    df_lag = add_target_lags(df, target_col="y", lag_minutes=(10,), sampling_minutes=5)

    assert "y_lag_10m" in df_lag.columns
    # Le prime 2 righe devono essere NaN
    assert df_lag["y_lag_10m"].iloc[0] != df_lag["y_lag_10m"].iloc[0]  # NaN check
    assert df_lag["y_lag_10m"].iloc[1] != df_lag["y_lag_10m"].iloc[1]
    # Il resto è shiftato
    assert df_lag["y_lag_10m"].iloc[2] == 10
    assert df_lag["y_lag_10m"].iloc[3] == 20
    assert df_lag["y_lag_10m"].iloc[4] == 30


def test_add_target_lags_invalid_multiple() -> None:
    """Se lag_minutes non è multiplo del sampling e enforce_multiple=True, solleva ValueError."""
    df = pd.DataFrame({"y": [1, 2, 3]})
    with pytest.raises(ValueError):
        add_target_lags(df, "y", lag_minutes=(7,), sampling_minutes=5, enforce_multiple=True)


def test_add_time_of_day_features_with_index() -> None:
    """add_time_of_day_features usa il DatetimeIndex se datetime_col è None."""
    idx = pd.date_range("2020-01-01", periods=4, freq="6H")
    df = pd.DataFrame({"y": [1, 2, 3, 4]}, index=idx)

    df_out, cols = add_time_of_day_features(df, include_12h=True)

    for c in ["tod_sin_24h", "tod_cos_24h", "tod_sin_12h", "tod_cos_12h"]:
        assert c in df_out.columns
        assert c in cols
        assert df_out[c].between(-1.0, 1.0).all()


def test_add_time_of_day_features_with_bad_index() -> None:
    """Se non c'è DatetimeIndex e datetime_col è None, viene sollevato ValueError."""
    df = pd.DataFrame({"y": [1, 2, 3]})
    with pytest.raises(ValueError):
        add_time_of_day_features(df)


def test_add_event_present_indicator_basic() -> None:
    """add_event_present_indicator crea una colonna binaria basata sulla soglia."""
    df = pd.DataFrame({"carbs": [0.0, 5.0, None, 0.1]})

    out = add_event_present_indicator(
        df,
        col="carbs",
        threshold=0.0,
        strictly_greater=True,
        out_name="carbs_present",
    )

    assert "carbs_present" in out.columns
    # NaN diventa 0, >0 diventa 1
    assert list(out["carbs_present"]) == [0, 1, 0, 1]


def test_add_exponential_decay_feature_past_only() -> None:
    """L'exponential decay cresce quando ci sono eventi e decade negli step successivi."""
    idx = pd.date_range("2020-01-01 00:00", periods=4, freq="5T")
    df = pd.DataFrame({"event": [1.0, 0.0, 0.0, 0.0]}, index=idx)

    df_out, name = add_exponential_decay_feature(
        df,
        col="event",
        sampling_minutes=5,
        halflife_min=5.0,  # half-life = 5 min => alpha=0.5 per step
        past_only=True,
    )

    assert name in df_out.columns
    r = df_out[name].to_numpy()

    # Calcolo atteso:
    # t0: r0 = 0.5*0 + 0 (x_{-1}=0) = 0
    # t1: r1 = 0.5*r0 + x0 = 0 + 1 = 1
    # t2: r2 = 0.5*r1 + x1 = 0.5*1 + 0 = 0.5
    # t3: r3 = 0.5*r2 + x2 = 0.5*0.5 + 0 = 0.25
    assert np.allclose(r, [0.0, 1.0, 0.5, 0.25], atol=1e-6)
