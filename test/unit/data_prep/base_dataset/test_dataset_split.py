import pytest


def test_split_by_subject_basic(dummy_dataset):
    train_data, val_data = dummy_dataset._split_by_subject(
        val_ratio=0.5,
        random_state=123,
    )

    # 2 soggetti → 1 train, 1 val
    assert len(train_data) + len(val_data) == len(dummy_dataset.all_data)
    assert len(train_data) == 1
    assert len(val_data) == 1

    # nessun soggetto perso o duplicato
    original_ids = {id(df) for df in dummy_dataset.all_data}
    split_ids = {id(df) for df in train_data + val_data}
    assert original_ids == split_ids
    assert not set(map(id, train_data)) & set(map(id, val_data))


def test_split_by_subject_empty_raises(dummy_dataset):
    dummy_dataset.all_data = []
    with pytest.raises(ValueError, match="No subjects available in all_data"):
        dummy_dataset._split_by_subject(val_ratio=0.2, random_state=0)


def test_split_by_time_index_basic(dummy_dataset):
    # ogni soggetto ha 12 righe → con val_ratio=0.5 ci aspettiamo 6/6
    train_data, val_data = dummy_dataset._split_by_time_index(val_ratio=0.5)

    # stessa numerosità di soggetti
    assert len(train_data) == len(dummy_dataset.all_data)
    assert len(val_data) == len(dummy_dataset.all_data)

    for original_df, train_df, val_df in zip(
        dummy_dataset.all_data, train_data, val_data
    ):
        assert len(train_df) + len(val_df) == len(original_df)
        # partizione temporale corretta
        assert train_df.index.max() < val_df.index.min()


def test_split_by_time_index_invalid_val_ratio(dummy_dataset):
    for bad in (0.0, 1.0, -0.1, 1.1):
        with pytest.raises(ValueError, match="val_ratio must be between 0 and 1"):
            dummy_dataset._split_by_time_index(val_ratio=bad)
