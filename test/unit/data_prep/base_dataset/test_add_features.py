import pandas as pd
import pytest


def test_add_features_to_df_adds_expected_columns(dummy_dataset, toy_subject_df):
    original = toy_subject_df

    new_df, added_cols = dummy_dataset._add_features_to_df(toy_subject_df)

    # l'input non è modificato in-place
    pd.testing.assert_frame_equal(toy_subject_df, original)

    expected_cols = {
        "tod_sin_24h",
        "tod_cos_24h",
        "bolus_total_expdecay_120m",
        "carbs_expdecay_120m",
    }
    assert expected_cols.issubset(set(new_df.columns))
    assert set(added_cols) == expected_cols

    # qualche check semplice sui valori (devono essere >= 0 per le expdecay)
    assert (new_df["bolus_total_expdecay_120m"] >= 0).all()
    assert new_df["bolus_total_expdecay_120m"].max() > 0
    assert new_df["carbs_expdecay_120m"].max() > 0


def test_add_features_updates_all_data_and_cols(dummy_dataset):
    initial_cols = list(dummy_dataset.cols)

    added = dummy_dataset.add_features()

    # ritorno e attributo added_cols coerenti
    assert added
    assert set(added) == set(dummy_dataset.added_cols)

    # ogni df ha le nuove colonne
    for df in dummy_dataset.all_data:
        for col in added:
            assert col in df.columns

    # self.cols deve contenere sia le vecchie che le nuove
    for col in initial_cols:
        assert col in dummy_dataset.cols
    for col in added:
        assert col in dummy_dataset.cols


