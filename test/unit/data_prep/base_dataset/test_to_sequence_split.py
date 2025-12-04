import numpy as np
import pytest

from data_prep.utils import build_sliding_windows


def test_to_sequence_splits_subject_matches_direct_windows(dummy_dataset):
    feature_cols = ["glucose", "basal_rate"]
    seq_len = 3
    step = 1
    val_ratio = 0.5
    random_state = 42

    X_train, X_val = dummy_dataset.to_sequence_splits(
        seq_len=seq_len,
        step=step,
        feature_cols=feature_cols,
        val_ratio=val_ratio,
        split_by="subject",
        max_missing_ratio=0.0,
        random_state=random_state,
    )

    train_data, val_data = dummy_dataset._split_by_subject(
        val_ratio=val_ratio,
        random_state=random_state,
    )

    X_train_ref = build_sliding_windows(
        all_data=train_data,
        feature_cols=feature_cols,
        seq_len=seq_len,
        step=step,
        max_missing_ratio=0.0,
    )
    X_val_ref = build_sliding_windows(
        all_data=val_data,
        feature_cols=feature_cols,
        seq_len=seq_len,
        step=step,
        max_missing_ratio=0.0,
    )

    assert np.array_equal(X_train, X_train_ref)
    assert np.array_equal(X_val, X_val_ref)


def test_to_sequence_splits_time_matches_direct_windows(dummy_dataset):
    feature_cols = ["glucose", "basal_rate"]
    seq_len = 3
    step = 1
    val_ratio = 0.5

    X_train, X_val = dummy_dataset.to_sequence_splits(
        seq_len=seq_len,
        step=step,
        feature_cols=feature_cols,
        val_ratio=val_ratio,
        split_by="time",
        max_missing_ratio=0.0,
        random_state=None,
    )

    train_data, val_data = dummy_dataset._split_by_time_index(val_ratio=val_ratio)

    X_train_ref = build_sliding_windows(
        all_data=train_data,
        feature_cols=feature_cols,
        seq_len=seq_len,
        step=step,
        max_missing_ratio=0.0,
    )
    X_val_ref = build_sliding_windows(
        all_data=val_data,
        feature_cols=feature_cols,
        seq_len=seq_len,
        step=step,
        max_missing_ratio=0.0,
    )

    assert np.array_equal(X_train, X_train_ref)
    assert np.array_equal(X_val, X_val_ref)


def test_to_sequence_splits_invalid_val_ratio(dummy_dataset):
    feature_cols = ["glucose"]
    for bad in (-0.1, 0.0, 1.0, 1.1):
        with pytest.raises(ValueError, match="val_ratio must be between 0 and 1"):
            dummy_dataset.to_sequence_splits(
                seq_len=3,
                step=1,
                feature_cols=feature_cols,
                val_ratio=bad,
                split_by="subject",
            )
