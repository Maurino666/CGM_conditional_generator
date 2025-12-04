import numpy as np
import pandas as pd
import pytest


def test_clean_cols_coerces_numeric_fills_defaults_and_clips_impulses(dummy_dataset):
    # partiamo da una copia del primo soggetto
    df = dummy_dataset.all_data[0]

    # 1) valore non numerico in glucose → deve diventare NaN
    df.loc[df.index[0], "glucose"] = "not_a_number"

    # 2) NaN in basal_rate → deve essere riempito col default
    df.loc[df.index[1], "basal_rate"] = np.nan
    dummy_dataset.defaults["basal_rate"] = 1.23

    # 3) valori negativi nelle impulse_cols → devono essere clippati a 0
    df.loc[df.index[2], "bolus_total"] = -5.0
    df.loc[df.index[3], "carbs"] = -10.0

    dummy_dataset.all_data[0] = df

    dummy_dataset._clean_cols()

    cleaned = dummy_dataset.all_data[0]

    # tutte le numeric_cols hanno dtype numerico
    for col in dummy_dataset.numeric_cols:
        assert pd.api.types.is_numeric_dtype(cleaned[col])

    # il valore non numerico è diventato NaN
    assert np.isnan(cleaned["glucose"].iloc[0])

    # basal_rate riempito col default
    assert cleaned["basal_rate"].iloc[1] == pytest.approx(1.23)

    # colonne di impulso clippate a >= 0
    assert (cleaned["bolus_total"] >= 0).all()
    assert (cleaned["carbs"] >= 0).all()
