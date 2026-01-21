from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reconstruction import ReconstructionConfig, WindowReconstructor
from reconstruction.strategies import NonOverlapStrategy


# NonOverlapStrategy tests
def test_non_overlap_strategy_place_basic() -> None:
    """Placing a window should overwrite the correct slice."""
    strategy = NonOverlapStrategy()
    buf = np.full((6, 1), np.nan, dtype=np.float32)

    win = np.array([[1.0], [2.0]], dtype=np.float32)
    strategy.place(buf, win, start_row=2)

    assert np.isnan(buf[0, 0])
    assert np.isnan(buf[1, 0])
    assert buf[2, 0] == 1.0
    assert buf[3, 0] == 2.0
    assert np.isnan(buf[4, 0])
    assert np.isnan(buf[5, 0])


# WindowReconstructor tests
def test_reconstruct_single_subject_full_coverage_non_overlap(
    toy_timeseries_df: pd.DataFrame,
) -> None:
    """
    Reconstruct a single subject where windows cover the whole timeline
    (non-overlapping windows with step == seq_len).
    """
    cfg = ReconstructionConfig(
        target_col="glucose",
        cond_cols=["basal_rate", "carbs"],
        synth_col="glucose_synth",
        include_true_target=True,
    )
    recon = WindowReconstructor(cfg)

    templates = {0: toy_timeseries_df}  # length = 6
    # seq_len = 2, start rows = 0, 2, 4 -> full coverage
    meta = [(0, 0), (0, 2), (0, 4)]

    c_windows = np.zeros((3, 2, 2), dtype=np.float32)  # values not used by reconstructor
    y_hat_windows = np.array(
        [
            [[1.0], [2.0]],
            [[3.0], [4.0]],
            [[5.0], [6.0]],
        ],
        dtype=np.float32,
    )

    out = recon.reconstruct_subject_dfs(
        templates=templates,
        meta=meta,
        c_windows=c_windows,
        y_hat_windows=y_hat_windows,
    )

    assert len(out) == 1
    df_out = out[0]

    # index preserved
    assert df_out.index.equals(toy_timeseries_df.index)

    # required columns
    assert "basal_rate" in df_out.columns
    assert "carbs" in df_out.columns
    assert "glucose" in df_out.columns
    assert "glucose_synth" in df_out.columns

    # cond copied from template
    assert np.allclose(df_out["basal_rate"].to_numpy(), toy_timeseries_df["basal_rate"].to_numpy())
    assert np.allclose(df_out["carbs"].to_numpy(), toy_timeseries_df["carbs"].to_numpy())

    # true target copied from template
    assert np.allclose(df_out["glucose"].to_numpy(), toy_timeseries_df["glucose"].to_numpy())

    # synthetic reconstructed
    assert np.allclose(df_out["glucose_synth"].to_numpy(), np.array([1, 2, 3, 4, 5, 6], dtype=float))


def test_reconstruct_single_subject_partial_coverage_keeps_nans(
    toy_timeseries_df: pd.DataFrame,
) -> None:
    """If windows do not cover the full timeline, uncovered rows should remain NaN."""
    cfg = ReconstructionConfig(
        target_col="glucose",
        cond_cols=["basal_rate"],
        synth_col="glucose_synth",
        include_true_target=True,
    )
    recon = WindowReconstructor(cfg)

    templates = {0: toy_timeseries_df}  # length = 6

    # Only first window placed, rest should be NaN
    meta = [(0, 0)]
    c_windows = np.zeros((1, 2, 1), dtype=np.float32)
    y_hat_windows = np.array([[[7.0], [8.0]]], dtype=np.float32)

    out = recon.reconstruct_subject_dfs(
        templates=templates,
        meta=meta,
        c_windows=c_windows,
        y_hat_windows=y_hat_windows,
    )

    df_out = out[0]
    synth = df_out["glucose_synth"].to_numpy()

    assert np.allclose(synth[:2], np.array([7.0, 8.0]))
    assert np.isnan(synth[2:]).all()


def test_reconstruct_multiple_subjects_sorted_by_subject_id(
    toy_timeseries_df: pd.DataFrame,
) -> None:
    """Output must be deterministic and sorted by subject_id."""
    cfg = ReconstructionConfig(
        target_col="glucose",
        cond_cols=["basal_rate"],
        synth_col="glucose_synth",
        include_true_target=True,
    )
    recon = WindowReconstructor(cfg)

    # create two subjects with same structure but different glucose
    df0 = toy_timeseries_df.copy()
    df1 = toy_timeseries_df.copy()
    df1["glucose"] = df1["glucose"] + 100

    # templates inserted in reverse order on purpose
    templates = {2: df1, 1: df0}

    meta = [
        (2, 0),
        (1, 0),
    ]
    c_windows = np.zeros((2, 2, 1), dtype=np.float32)
    y_hat_windows = np.array(
        [
            [[10.0], [11.0]],  # subject 2
            [[20.0], [21.0]],  # subject 1
        ],
        dtype=np.float32,
    )

    out = recon.reconstruct_subject_dfs(
        templates=templates,
        meta=meta,
        c_windows=c_windows,
        y_hat_windows=y_hat_windows,
    )

    # must be sorted by sid: [1, 2]
    assert len(out) == 2
    assert np.allclose(out[0]["glucose_synth"].to_numpy()[:2], np.array([20.0, 21.0]))
    assert np.allclose(out[1]["glucose_synth"].to_numpy()[:2], np.array([10.0, 11.0]))


def test_reconstruct_include_true_target_false_does_not_add_target_col(
    toy_timeseries_df: pd.DataFrame,
) -> None:
    """If include_true_target=False, output should not contain the true target column."""
    cfg = ReconstructionConfig(
        target_col="glucose",
        cond_cols=["basal_rate"],
        synth_col="glucose_synth",
        include_true_target=False,
    )
    recon = WindowReconstructor(cfg)

    templates = {0: toy_timeseries_df}
    meta = [(0, 0)]
    c_windows = np.zeros((1, 2, 1), dtype=np.float32)
    y_hat_windows = np.array([[[1.0], [2.0]]], dtype=np.float32)

    out = recon.reconstruct_subject_dfs(
        templates=templates,
        meta=meta,
        c_windows=c_windows,
        y_hat_windows=y_hat_windows,
    )

    df_out = out[0]
    assert "glucose" not in df_out.columns
    assert "glucose_synth" in df_out.columns


def test_reconstruct_missing_cond_col_fills_nan(
    toy_timeseries_df: pd.DataFrame,
) -> None:
    """If a conditional column is missing in the template, the output column should be NaN."""
    cfg = ReconstructionConfig(
        target_col="glucose",
        cond_cols=["basal_rate", "does_not_exist"],
        synth_col="glucose_synth",
        include_true_target=True,
    )
    recon = WindowReconstructor(cfg)

    templates = {0: toy_timeseries_df}
    meta = [(0, 0)]
    c_windows = np.zeros((1, 2, 2), dtype=np.float32)
    y_hat_windows = np.array([[[1.0], [2.0]]], dtype=np.float32)

    out = recon.reconstruct_subject_dfs(
        templates=templates,
        meta=meta,
        c_windows=c_windows,
        y_hat_windows=y_hat_windows,
    )

    df_out = out[0]
    assert "does_not_exist" in df_out.columns
    assert df_out["does_not_exist"].isna().all()


def test_reconstruct_raises_on_meta_y_misalignment(
    toy_timeseries_df: pd.DataFrame,
) -> None:
    """meta length must match number of windows for both y_hat and c_windows."""
    cfg = ReconstructionConfig(
        target_col="glucose",
        cond_cols=["basal_rate"],
        synth_col="glucose_synth",
        include_true_target=True,
    )
    recon = WindowReconstructor(cfg)

    templates = {0: toy_timeseries_df}
    meta = [(0, 0), (0, 2)]  # 2 windows

    c_windows = np.zeros((2, 2, 1), dtype=np.float32)
    y_hat_windows = np.zeros((1, 2, 1), dtype=np.float32)  # mismatched (1)

    with pytest.raises(ValueError, match="meta and y_hat_windows must be aligned"):
        recon.reconstruct_subject_dfs(
            templates=templates,
            meta=meta,
            c_windows=c_windows,
            y_hat_windows=y_hat_windows,
        )


def test_reconstruct_raises_on_meta_c_misalignment(
    toy_timeseries_df: pd.DataFrame,
) -> None:
    cfg = ReconstructionConfig(
        target_col="glucose",
        cond_cols=["basal_rate"],
        synth_col="glucose_synth",
        include_true_target=True,
    )
    recon = WindowReconstructor(cfg)

    templates = {0: toy_timeseries_df}
    meta = [(0, 0), (0, 2)]  # 2 windows

    c_windows = np.zeros((1, 2, 1), dtype=np.float32)      # mismatched (1)
    y_hat_windows = np.zeros((2, 2, 1), dtype=np.float32)  # ok (2)

    with pytest.raises(ValueError, match="meta and c_windows must be aligned"):
        recon.reconstruct_subject_dfs(
            templates=templates,
            meta=meta,
            c_windows=c_windows,
            y_hat_windows=y_hat_windows,
        )


def test_reconstruct_raises_if_subject_id_not_in_templates(
    toy_timeseries_df: pd.DataFrame,
) -> None:
    """If metadata contains a subject id not present in templates, raise KeyError."""
    cfg = ReconstructionConfig(
        target_col="glucose",
        cond_cols=["basal_rate"],
        synth_col="glucose_synth",
        include_true_target=True,
    )
    recon = WindowReconstructor(cfg)

    templates = {0: toy_timeseries_df}
    meta = [(999, 0)]  # invalid subject id

    c_windows = np.zeros((1, 2, 1), dtype=np.float32)
    y_hat_windows = np.zeros((1, 2, 1), dtype=np.float32)

    with pytest.raises(KeyError, match="Subject id 999 not found in templates"):
        recon.reconstruct_subject_dfs(
            templates=templates,
            meta=meta,
            c_windows=c_windows,
            y_hat_windows=y_hat_windows,
        )
