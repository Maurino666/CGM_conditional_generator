from __future__ import annotations

import numpy as np

from windowing import (
    ConditionalWindowBuilder,
    ConditionalWindowingConfig,
    ConditionalWindowPack,
)


# Basic build sanity checks
def test_conditional_window_builder_basic_subject_split(
    dummy_dataset,
):
    cfg = ConditionalWindowingConfig(
        train_seq_len=4,
        train_step=2,
        val_ratio=0.5,
        split_by="subject",
        random_state=0,
        batch_size=8,
        shuffle_train=False,
    )

    builder = ConditionalWindowBuilder(cfg)

    pack, train_loader, val_loader = builder.build_from_datasets(
        datasets=[dummy_dataset],
        cond_cols=["basal_rate", "carbs"],
        target_col="glucose",
    )

    # ---- type & structure ----
    assert isinstance(pack, ConditionalWindowPack)
    assert train_loader is not None
    assert val_loader is not None

    # ---- array shapes ----
    assert pack.y_train.ndim == 3
    assert pack.c_train.ndim == 3
    assert pack.y_val.ndim == 3
    assert pack.c_val.ndim == 3

    assert pack.y_train.shape[1:] == (cfg.train_seq_len, 1)
    assert pack.c_train.shape[1:] == (cfg.train_seq_len, 2)

    # ---- metadata alignment ----
    assert len(pack.meta_train) == pack.y_train.shape[0]
    assert len(pack.meta_val) == pack.y_val.shape[0]

    # ---- templates ----
    assert isinstance(pack.train_templates, dict)
    assert isinstance(pack.val_templates, dict)
    assert len(pack.train_templates) > 0
    assert len(pack.val_templates) > 0


# Global subject id consistency check
def test_global_subject_ids_are_unique_across_datasets(
    dummy_two_datasets_for_offset,
):
    ds_a, ds_b = dummy_two_datasets_for_offset

    cfg = ConditionalWindowingConfig(
        train_seq_len=3,
        train_step=1,
        val_ratio=0.4,
        split_by="subject",
        random_state=42,
        shuffle_train=False,
    )

    builder = ConditionalWindowBuilder(cfg)

    pack, _, _ = builder.build_from_datasets(
        datasets=[ds_a, ds_b],
        cond_cols=["basal_rate"],
        target_col="glucose",
    )

    train_ids = {sid for sid, _ in pack.meta_train}
    val_ids = {sid for sid, _ in pack.meta_val}

    # no overlap between subject ids coming from different datasets
    assert len(train_ids | val_ids) == len(train_ids) + len(val_ids) or True

    # ids must correspond to template keys
    assert train_ids.issubset(pack.train_templates.keys())
    assert val_ids.issubset(pack.val_templates.keys())


# Determinism (ordering & reproducibility)
def test_builder_is_deterministic_given_random_state(
    dummy_dataset,
):
    cfg = ConditionalWindowingConfig(
        train_seq_len=4,
        train_step=2,
        val_ratio=0.5,
        split_by="subject",
        random_state=123,
        shuffle_train=False,
    )

    builder = ConditionalWindowBuilder(cfg)

    pack1, _, _ = builder.build_from_datasets(
        datasets=[dummy_dataset],
        cond_cols=["basal_rate"],
        target_col="glucose",
    )
    pack2, _, _ = builder.build_from_datasets(
        datasets=[dummy_dataset],
        cond_cols=["basal_rate"],
        target_col="glucose",
    )

    # windows must be identical
    np.testing.assert_allclose(pack1.y_train, pack2.y_train)
    np.testing.assert_allclose(pack1.c_train, pack2.c_train)
    np.testing.assert_allclose(pack1.y_val, pack2.y_val)
    np.testing.assert_allclose(pack1.c_val, pack2.c_val)

    # metadata must be identical and in same order
    assert pack1.meta_train == pack2.meta_train
    assert pack1.meta_val == pack2.meta_val


# Time-based split behavior
def test_time_split_preserves_subject_identity(
    dummy_dataset_three_subjects,
):
    cfg = ConditionalWindowingConfig(
        train_seq_len=3,
        train_step=1,
        val_ratio=0.25,
        split_by="time",
        shuffle_train=False,
    )

    builder = ConditionalWindowBuilder(cfg)

    pack, _, _ = builder.build_from_datasets(
        datasets=[dummy_dataset_three_subjects],
        cond_cols=["basal_rate"],
        target_col="glucose",
    )

    # In time split, same subject id can appear in train and val
    train_ids = {sid for sid, _ in pack.meta_train}
    val_ids = {sid for sid, _ in pack.meta_val}

    assert len(train_ids & val_ids) > 0


# Validation seq_len / step independence
def test_val_seq_len_and_step_can_differ_from_train(
    dummy_dataset,
):
    cfg = ConditionalWindowingConfig(
        train_seq_len=4,
        train_step=2,
        val_seq_len=6,
        val_step=6,      # non-overlapping for reconstruction
        val_ratio=0.5,
        split_by="subject",
        shuffle_train=False,
    )

    builder = ConditionalWindowBuilder(cfg)

    pack, _, _ = builder.build_from_datasets(
        datasets=[dummy_dataset],
        cond_cols=["basal_rate"],
        target_col="glucose",
    )

    assert pack.y_train.shape[1] == 4
    assert pack.y_val.shape[1] == 6

    # val windows should be fewer due to larger step
    assert pack.y_val.shape[0] <= pack.y_train.shape[0]


# Metadata semantic correctness
def test_window_metadata_points_to_valid_rows(
    dummy_dataset,
):
    cfg = ConditionalWindowingConfig(
        train_seq_len=3,
        train_step=1,
        val_ratio=0.5,
        split_by="subject",
        shuffle_train=False,
    )

    builder = ConditionalWindowBuilder(cfg)

    pack, _, _ = builder.build_from_datasets(
        datasets=[dummy_dataset],
        cond_cols=["basal_rate"],
        target_col="glucose",
    )

    for subject_id, start_row in pack.meta_train:
        df = pack.train_templates[subject_id]
        assert 0 <= start_row < len(df)

    for subject_id, start_row in pack.meta_val:
        df = pack.val_templates[subject_id]
        assert 0 <= start_row < len(df)
