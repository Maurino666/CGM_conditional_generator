from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .utils import WindowMetadata

@dataclass(frozen=True)
class ConditionalWindowPack:
    """
    Frozen container with deterministic-order windows and reconstruction info.
    This object is meant to be the single source of truth for:
      - training loaders
      - generation inputs (typically val split)
      - reconstruction back to list[pd.DataFrame]
    """

    # Windows (numpy) in deterministic order
    y_train: np.ndarray
    c_train: np.ndarray
    y_val: np.ndarray
    c_val: np.ndarray

    # Per-window metadata aligned 1:1 with the arrays above
    meta_train: list[WindowMetadata]
    meta_val: list[WindowMetadata]

    # Templates for reconstruction (per subject, split-specific)
    # Each df is expected to be time-indexed and contain conditional columns and true target.
    train_templates: dict[int, pd.DataFrame]
    val_templates: dict[int, pd.DataFrame]

    # Schema / bookkeeping
    target_col: str
    cond_cols: list[str]
    freq_minutes: int
    split_by: str

    # Optional extra debug fields
    extra: dict[str, Any] | None = None

    def num_train_windows(self) -> int:
        return int(self.y_train.shape[0])

    def num_val_windows(self) -> int:
        return int(self.y_val.shape[0])
