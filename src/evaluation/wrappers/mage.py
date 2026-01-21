from __future__ import annotations

import numpy as np
import pandas as pd

from ..types import EvaluationConfig, Metric, MetricOutput
from evaluation.metrics_core import compute_mage


class MageMetric(Metric):
    """
    Wrapper for the MAGE (Mean Amplitude of Glycemic Excursions) scalar metric.

    Design choice:
    - MAGE is a single global scalar per subject.
    - No plotting is produced (artifacts are always empty).
    """

    def __init__(
        self,
        name: str = "mage",
        *,
        dropna: bool = True,
        value_range: tuple[float, float] = (40.0, 400.0),
        sd_threshold: float = 1.0,
        smooth_window: int = 3,
        min_separation: int = 1,
    ) -> None:
        self.name = name
        self._dropna = bool(dropna)
        self._value_range = value_range
        self._sd_threshold = float(sd_threshold)
        self._smooth_window = int(smooth_window)
        self._min_separation = int(min_separation)

    def compute(self, subject_id: int, df: pd.DataFrame, cfg: EvaluationConfig) -> MetricOutput:
        if cfg.target_col not in df.columns:
            raise ValueError(f"{self.name}: cfg.target_col='{cfg.target_col}' not found in DataFrame columns.")

        # Extract target series (ordering assumed correct by Evaluator)
        values = pd.to_numeric(df[cfg.target_col], errors="coerce")

        mage_value = compute_mage(
            values,
            dropna=self._dropna,
            value_range=self._value_range,
            sd_threshold=self._sd_threshold,
            smooth_window=self._smooth_window,
            min_separation=self._min_separation,
        )

        scalars = {
            self.name: float(mage_value) if mage_value is not None and np.isfinite(mage_value) else np.nan
        }

        return MetricOutput(
            scalars=scalars,
            tables={},
            artifacts={},  # intentionally empty
        )
