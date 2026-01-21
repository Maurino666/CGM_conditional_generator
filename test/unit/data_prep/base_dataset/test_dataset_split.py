import pytest

# Nota: Questi test ora chiamano i metodi ereditati da BaseDataset reale.

def test_split_by_subject_basic(dummy_dataset) -> None:
    val_ratio = 0.5
    # random_state serve perché split_by_subject usa numpy shuffle
    random_state = 123

    # _split_by_subject è un metodo "protetto", ma accessibile per i test
    train_data, val_data, train_ids, val_ids = dummy_dataset._split_by_subject(
        val_ratio=val_ratio,
        random_state=random_state,
    )

    # Checks invariati
    assert set(train_ids).isdisjoint(set(val_ids))
    n_subjects = len(dummy_dataset.all_data)
    assert set(train_ids) | set(val_ids) == set(range(n_subjects))
    assert len(train_data) == len(train_ids)
    assert len(val_data) == len(val_ids)


def test_split_by_subject_val_ratio_bounds(dummy_dataset) -> None:
    for bad in (-0.1, 0.0, 1.0, 1.1):
        with pytest.raises(ValueError):
            dummy_dataset.split(val_ratio=bad, split_by="subject")


def test_split_by_time_index_basic(dummy_dataset) -> None:
    val_ratio = 0.5

    # Nota: DummyDataset ha 2 soggetti di 12 righe ciascuno.
    train_data, val_data, train_ids, val_ids = dummy_dataset._split_by_time_index(
        val_ratio=val_ratio,
    )

    # Verifica Logica
    assert set(val_ids).issubset(set(train_ids))
    assert len(train_data) == len(train_ids)

    # Verifica lunghezze temporali
    for df_idx in val_ids:
        original_len = len(dummy_dataset.all_data[df_idx])

        # Filter segments
        train_segments = [df for sid, df in zip(train_ids, train_data) if sid == df_idx]
        val_segments = [df for sid, df in zip(val_ids, val_data) if sid == df_idx]

        assert len(train_segments) == 1
        assert len(val_segments) == 1

        # Con 12 righe e ratio 0.5 -> 6 train, 6 val
        assert len(train_segments[0]) == 6
        assert len(val_segments[0]) == 6
        assert len(train_segments[0]) + len(val_segments[0]) == original_len


def test_split_public_api(dummy_dataset):
    """Testa che il metodo pubblico .split() deleghi correttamente."""
    # Test subject split delegation
    tr, va, tr_id, va_id = dummy_dataset.split(val_ratio=0.5, split_by="subject", random_state=42)
    assert len(tr) > 0

    # Test time split delegation
    tr_t, va_t, tr_id_t, va_id_t = dummy_dataset.split(val_ratio=0.5, split_by="time")
    assert len(tr_t) == 2  # 2 soggetti, entrambi splittati