# tests/unit/test_to_sequence_splits_conditional.py

import numpy as np
import pytest

from data_prep.utils import build_sliding_windows_conditional


def test_to_sequence_splits_conditional_subject_matches_utils(dummy_dataset) -> None:
    """Conditional to_sequence_splits (split_by='subject') matches direct utils."""
    cond_cols = ["basal_rate", "carbs"]
    train_seq_len = 3
    train_step = 1
    val_ratio = 0.5
    random_state = 42

    (
        X_train_y,
        X_train_c,
        X_val_y,
        X_val_c,
        train_metadata,
        val_metadata,
    ) = dummy_dataset.to_sequence_splits_conditional(
        train_seq_len=train_seq_len,
        train_step=train_step,
        cond_cols=cond_cols,
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

    y_train_ref, c_train_ref, _ = build_sliding_windows_conditional(
        all_data=train_data,
        seq_len=train_seq_len,
        step=train_step,
        target_col=dummy_dataset.target_col,
        cond_cols=cond_cols,
        max_missing_ratio=0.0,
    )
    y_val_ref, c_val_ref, _ = build_sliding_windows_conditional(
        all_data=val_data,
        seq_len=train_seq_len,
        step=train_step,
        target_col=dummy_dataset.target_col,
        cond_cols=cond_cols,
        max_missing_ratio=0.0,
    )

    assert np.array_equal(X_train_y, y_train_ref)
    assert np.array_equal(X_train_c, c_train_ref)
    assert np.array_equal(X_val_y, y_val_ref)
    assert np.array_equal(X_val_c, c_val_ref)
    assert len(train_metadata) == X_train_y.shape[0]
    assert len(val_metadata) == X_val_y.shape[0]


def test_to_sequence_splits_conditional_time_matches_utils(dummy_dataset) -> None:
    """Conditional to_sequence_splits (split_by='time') matches direct utils."""
    cond_cols = ["basal_rate", "carbs"]
    train_seq_len = 3
    train_step = 1
    val_ratio = 0.5

    (
        X_train_y,
        X_train_c,
        X_val_y,
        X_val_c,
        train_metadata,
        val_metadata,
    ) = dummy_dataset.to_sequence_splits_conditional(
        train_seq_len=train_seq_len,
        train_step=train_step,
        cond_cols=cond_cols,
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

    y_train_ref, c_train_ref, _ = build_sliding_windows_conditional(
        all_data=train_data,
        seq_len=train_seq_len,
        step=train_step,
        target_col=dummy_dataset.target_col,
        cond_cols=cond_cols,
        max_missing_ratio=0.0,
    )
    y_val_ref, c_val_ref, _ = build_sliding_windows_conditional(
        all_data=val_data,
        seq_len=train_seq_len,
        step=train_step,
        target_col=dummy_dataset.target_col,
        cond_cols=cond_cols,
        max_missing_ratio=0.0,
    )

    assert np.array_equal(X_train_y, y_train_ref)
    assert np.array_equal(X_train_c, c_train_ref)
    assert np.array_equal(X_val_y, y_val_ref)
    assert np.array_equal(X_val_c, c_val_ref)
    assert len(train_metadata) == X_train_y.shape[0]
    assert len(val_metadata) == X_val_y.shape[0]


def test_to_sequence_splits_conditional_invalid_val_ratio(dummy_dataset) -> None:
    """Invalid val_ratio raises ValueError for conditional splits."""
    cond_cols = ["basal_rate"]
    for bad in (-0.1, 0.0, 1.0, 1.1):
        with pytest.raises(ValueError, match="val_ratio must be between 0 and 1"):
            _ = dummy_dataset.to_sequence_splits_conditional(
                train_seq_len=3,
                train_step=1,
                cond_cols=cond_cols,
                val_ratio=bad,
                split_by="subject",
            )
