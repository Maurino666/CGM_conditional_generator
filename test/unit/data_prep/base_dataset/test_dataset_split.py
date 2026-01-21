# tests/unit/data_prep/base_dataset/test_dataset_split.py

import pytest


def test_split_by_subject_basic(dummy_dataset) -> None:
    """
    _split_by_subject must split the list of DataFrames into
    two non-overlapping sets whose union covers all subjects.
    """
    val_ratio = 0.5
    random_state = 123

    train_data, val_data, train_ids, val_ids = dummy_dataset._split_by_subject(
        val_ratio=val_ratio,
        random_state=random_state,
    )

    # Controllo che non ci siano sovrapposizioni negli ID
    assert set(train_ids).isdisjoint(set(val_ids))

    # E che la loro unione copra tutti i soggetti non vuoti
    n_subjects = len(dummy_dataset.all_data)
    assert set(train_ids) | set(val_ids) == set(range(n_subjects))

    # Sanity check: il numero di dfs in train e val è coerente con gli ID
    assert len(train_data) == len(train_ids)
    assert len(val_data) == len(val_ids)


def test_split_by_subject_val_ratio_bounds(dummy_dataset) -> None:
    """_split_by_subject must raise for invalid val_ratio."""
    for bad in (-0.1, 0.0, 1.0, 1.1):
        with pytest.raises(ValueError):
            _ = dummy_dataset._split_by_subject(
                val_ratio=bad,
                random_state=42,
            )


def test_split_by_time_index_basic(dummy_dataset) -> None:
    """
    _split_by_time_index must return train and val segments per subject.

    Each subject in dummy_dataset ha 12 righe → con val_ratio=0.5
    ci aspettiamo ~6 punti in train e ~6 in val per i soggetti con split non degenerato.
    """
    val_ratio = 0.5

    train_data, val_data, train_ids, val_ids = dummy_dataset._split_by_time_index(
        val_ratio=val_ratio,
    )

    # Tutti gli ID in val devono essere contenuti in quelli di train
    # (perché per ogni soggetto che ha val, ha anche una porzione di train)
    assert set(val_ids).issubset(set(train_ids))

    # train_ids e train_data devono avere stessa lunghezza
    assert len(train_data) == len(train_ids)
    assert len(val_data) == len(val_ids)

    # Per ogni soggetto che appare anche in val, la somma delle lunghezze
    # train + val per quel soggetto deve corrispondere alla lunghezza originale
    for df_idx in val_ids:
        original_len = len(dummy_dataset.all_data[df_idx])

        # Trova il segmento train corrispondente
        train_segments = [
            df for sid, df in zip(train_ids, train_data) if sid == df_idx
        ]
        val_segments = [
            df for sid, df in zip(val_ids, val_data) if sid == df_idx
        ]

        # Dovrebbe esserci esattamente un segmento train e uno val per quel subject
        assert len(train_segments) == 1
        assert len(val_segments) == 1

        train_len = len(train_segments[0])
        val_len = len(val_segments[0])

        assert train_len + val_len == original_len


def test_split_by_time_index_val_ratio_bounds(dummy_dataset) -> None:
    """_split_by_time_index must raise for invalid val_ratio."""
    for bad in (-0.1, 0.0, 1.0, 1.1):
        with pytest.raises(ValueError):
            _ = dummy_dataset._split_by_time_index(val_ratio=bad)
