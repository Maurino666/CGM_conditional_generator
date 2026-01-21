# tests/unit/test_to_sequence_splits.py

import numpy as np
import pytest

from data_prep.utils import build_sliding_windows


def test_to_sequence_splits_subject_matches_direct_windows(dummy_dataset) -> None:
    """to_sequence_splits (split_by='subject') matches direct build_sliding_windows."""
    feature_cols = ["glucose", "basal_rate"]
    train_seq_len = 3
    train_step = 1
    val_ratio = 0.5
    random_state = 42

    X_train, X_val, train_metadata, val_metadata = dummy_dataset.to_sequence_splits(
        feature_cols=feature_cols,
        train_seq_len=train_seq_len,
        train_step=train_step,
        val_seq_len=train_seq_len,
        val_step=train_step,
        val_ratio=val_ratio,
        split_by="subject",
        max_missing_ratio=0.0,
        random_state=random_state,
    )

    train_data, val_data, _, _ = dummy_dataset._split_by_subject(
        val_ratio=val_ratio,
        random_state=random_state,
    )

    X_train_ref, _ = build_sliding_windows(
        all_data=train_data,
        feature_cols=feature_cols,
        seq_len=train_seq_len,
        step=train_step,
        max_missing_ratio=0.0,
    )
    X_val_ref, _ = build_sliding_windows(
        all_data=val_data,
        feature_cols=feature_cols,
        seq_len=train_seq_len,
        step=train_step,
        max_missing_ratio=0.0,
    )

    assert np.array_equal(X_train, X_train_ref)
    assert np.array_equal(X_val, X_val_ref)
    # metadata length must match number of windows
    assert len(train_metadata) == X_train.shape[0]
    assert len(val_metadata) == X_val.shape[0]


def test_to_sequence_splits_time_matches_direct_windows(dummy_dataset) -> None:
    """to_sequence_splits (split_by='time') matches direct build_sliding_windows."""
    feature_cols = ["glucose", "basal_rate"]
    train_seq_len = 3
    train_step = 1
    val_ratio = 0.5

    X_train, X_val, train_metadata, val_metadata = dummy_dataset.to_sequence_splits(
        feature_cols=feature_cols,
        train_seq_len=train_seq_len,
        train_step=train_step,
        val_seq_len=train_seq_len,
        val_step=train_step,
        val_ratio=val_ratio,
        split_by="time",
        max_missing_ratio=0.0,
        random_state=None,
    )

    train_data, val_data, _, _ = dummy_dataset._split_by_time_index(
        val_ratio=val_ratio,
    )

    X_train_ref, _ = build_sliding_windows(
        all_data=train_data,
        feature_cols=feature_cols,
        seq_len=train_seq_len,
        step=train_step,
        max_missing_ratio=0.0,
    )
    X_val_ref, _ = build_sliding_windows(
        all_data=val_data,
        feature_cols=feature_cols,
        seq_len=train_seq_len,
        step=train_step,
        max_missing_ratio=0.0,
    )

    assert np.array_equal(X_train, X_train_ref)
    assert np.array_equal(X_val, X_val_ref)
    assert len(train_metadata) == X_train.shape[0]
    assert len(val_metadata) == X_val.shape[0]


def test_to_sequence_splits_invalid_val_ratio(dummy_dataset) -> None:
    """Invalid val_ratio raises ValueError."""
    feature_cols = ["glucose"]
    for bad in (-0.1, 0.0, 1.0, 1.1):
        with pytest.raises(ValueError, match="val_ratio must be between 0 and 1"):
            _ = dummy_dataset.to_sequence_splits(
                feature_cols=feature_cols,
                train_seq_len=3,
                train_step=1,
                val_ratio=bad,
                split_by="subject",
            )
