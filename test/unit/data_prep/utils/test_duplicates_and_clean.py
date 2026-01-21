# tests/unit/test_duplicates_and_clean.py

from pathlib import Path

import pandas as pd

from data_prep.utils import clean_duplicates

def test_clean_duplicates_removes_duplicated_index() -> None:
    """clean_duplicates rimuove le righe con timestamp duplicati (keep='first')."""
    idx = pd.to_datetime(
        [
            "2020-01-01 00:00",
            "2020-01-01 00:00",
            "2020-01-01 00:05",
            "2020-01-01 00:05",
        ]
    )
    df = pd.DataFrame({"a": [1, 2, 3, 4]}, index=idx)

    cleaned_list = clean_duplicates([df])
    assert len(cleaned_list) == 1

    cleaned = cleaned_list[0]
    assert len(cleaned) == 2  # una sola riga per timestamp
    assert cleaned.index.is_unique
    # Mantiene la prima occorrenza per ciascun timestamp
    assert cleaned.loc["2020-01-01 00:00", "a"] == 1
    assert cleaned.loc["2020-01-01 00:05", "a"] == 3