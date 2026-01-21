from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from windowing import WindowMetadata
from .strategies import NonOverlapStrategy


@dataclass(frozen=True)
class ReconstructionConfig:
    """
    Configuration for windows -> list[df] reconstruction.
    """
    target_col: str
    cond_cols: list[str]
    synth_col: str = "glucose_synth"
    include_true_target: bool = True


class WindowReconstructor:
    """
    Reconstruct per-subject DataFrames from generated target windows + conditional windows + metadata.

    The reconstructor does NOT assume access to DataLoader (which may shuffle).
    It expects windows and metadata to be aligned and deterministic (from the WindowPack).
    """

    def __init__(
        self,
        cfg: ReconstructionConfig,
        *,
        strategy: NonOverlapStrategy | None = None,
    ) -> None:
        self.cfg = cfg
        self.strategy = strategy if strategy is not None else NonOverlapStrategy()

    def reconstruct_subject_dfs(
        self,
        *,
        templates: dict[int, pd.DataFrame],
        meta: list[WindowMetadata],
        c_windows: np.ndarray,
        y_hat_windows: np.ndarray,
    ) -> list[pd.DataFrame]:
        """
        Reconstruct a list of DataFrames (one per subject) using templates as backbone.

        Output columns:
          - cond columns (copied from template)
          - true target column (optional, copied from template)
          - synth target column (always present)

        Parameters
        ----------
        templates:
            dict[subject_id -> df] representing the split segment.
        meta:
            list[(subject_id, start_row)] aligned with windows.
        c_windows:
            Conditional windows aligned with meta. Shape (N, seq_len, cond_dim).
        y_hat_windows:
            Generated target windows aligned with meta. Shape (N, seq_len, 1).

        Returns
        -------
        list[pd.DataFrame]
            Sorted by subject_id for determinism.
        """
        if len(meta) != int(y_hat_windows.shape[0]):
            raise ValueError("meta and y_hat_windows must be aligned (same number of windows).")
        if len(meta) != int(c_windows.shape[0]):
            raise ValueError("meta and c_windows must be aligned (same number of windows).")

        # Build per-subject buffers for the synthetic target
        synth_buffers: dict[int, np.ndarray] = {}
        for sid, df in templates.items():
            # Synthetic target buffer: one column
            synth_buffers[sid] = np.full((len(df), 1), np.nan, dtype=np.float32)

        # Place each window into its subject buffer
        for i, (sid, start_row) in enumerate(meta):
            if sid not in synth_buffers:
                # This can happen if templates were not built consistently with metadata.
                raise KeyError(f"Subject id {sid} not found in templates.")

            y_win = y_hat_windows[i]  # (seq_len, 1)
            self.strategy.place(synth_buffers[sid], y_win, start_row=start_row)

        # Build final df per subject
        out: list[pd.DataFrame] = []
        for sid in sorted(templates.keys()):
            template = templates[sid]

            # Start from a copy to avoid mutating templates
            df_out = pd.DataFrame(index=template.index)

            # Copy conditional columns if present
            for col in self.cfg.cond_cols:
                if col in template.columns:
                    df_out[col] = template[col]
                else:
                    # Keep it explicit: templates should contain cond cols for evaluation.
                    df_out[col] = np.nan

            # Optional true target
            if self.cfg.include_true_target:
                if self.cfg.target_col in template.columns:
                    df_out[self.cfg.target_col] = template[self.cfg.target_col]
                else:
                    df_out[self.cfg.target_col] = np.nan

            # Synthetic target
            df_out[self.cfg.synth_col] = synth_buffers[sid][:, 0]

            out.append(df_out)

        return out
