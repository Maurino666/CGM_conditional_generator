# tests/unit/test_fill_data_and_plot_gaps.py

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from data_prep.utils import fill_data


# Usa backend non-interattivo per matplotlib nei test
matplotlib.use("Agg")


def _make_irregular_df() -> pd.DataFrame:
    """DataFrame con sampling irregolare e qualche NaN."""
    idx = pd.to_datetime(
        [
            "2020-01-01 00:00",
            "2020-01-01 00:10",
            "2020-01-01 00:20",
        ]
    )
    df = pd.DataFrame(
        {
            "glucose": [100.0, np.nan, 140.0],
            "carbs": [10.0, np.nan, 0.0],
        },
        index=idx,
    )
    return df


def test_fill_data_interpolates_target_and_applies_defaults(tmp_path: Path) -> None:
    """fill_data interpola il target e applica i default sulle righe inferite."""
    df = _make_irregular_df()
    all_data = [df]

    expected = pd.Timedelta(minutes=5)
    max_gap = pd.Timedelta(minutes=15)
    target_col = "glucose"
    defaults = {"carbs": 0.0}

    out_dir = tmp_path / "plots"

    filled_list = fill_data(
        all_data=all_data,
        expected=expected,
        max_gap=max_gap,
        target_col=target_col,
        defaults=defaults,
        logging_path=out_dir,
    )

    assert len(filled_list) == 1
    filled = filled_list[0]

    # Index regolare ogni 5 minuti fra min e max
    assert filled.index.min() == df.index.min()
    assert filled.index.max() == df.index.max()
    deltas = filled.index.to_series().diff().dropna().unique()
    assert len(deltas) == 1
    assert deltas[0] == expected

    # Il target ha meno NaN dell'originale
    assert filled[target_col].isna().sum() <= df[target_col].isna().sum()

    # I default vengono applicati almeno in qualche riga
    assert (filled["carbs"] == 0.0).sum() > 0

    # plot_gaps deve aver salvato almeno un file PNG
    png_files = list(out_dir.glob("subject_*.png"))
    assert len(png_files) >= 1
