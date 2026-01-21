from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NonOverlapStrategy:
    """
    Non-overlapping reconstruction strategy.

    Assumption:
      - Windows are generated to be non-overlapping by construction (e.g. step == seq_len),
        OR we simply place each window at [start_row : start_row+seq_len] and accept overwrite-free layout.
    """

    def place(
        self,
        buffer: np.ndarray,
        window_values: np.ndarray,
        *,
        start_row: int,
    ) -> None:
        """
        Place a window into a per-subject buffer.

        Parameters
        ----------
        buffer:
            Array shape (T, 1) where T = len(subject_df).
        window_values:
            Array shape (seq_len, 1) for target.
        start_row:
            Starting row index (iloc-based) in the subject timeline.
        """
        end = start_row + int(window_values.shape[0])
        buffer[start_row:end, :] = window_values
