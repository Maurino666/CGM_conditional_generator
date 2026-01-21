from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from ..types import EvaluationConfig, Metric, MetricOutput
from evaluation.metrics_core import compute_clinical_stats


@dataclass(frozen=True)
class ClinicalStatsParams:
    hypo_threshold: float = 70.0
    hyper_threshold: float = 180.0


class ClinicalStatsMetric(Metric):
    """
    Computes basic descriptive and clinical statistics (Mean, TIR, GMI, etc.)
    strictly on the configured 'target_col'.

    This metric is oblivious to any comparison column. It characterizes
    the distribution of the primary target series only.
    """
    name: str = "basic_stats"
    requires_synthetic: bool = False

    def __init__(self, params: ClinicalStatsParams = ClinicalStatsParams()) -> None:
        self.params = params

    def compute(self, subject_id: int, df: pd.DataFrame, cfg: EvaluationConfig) -> MetricOutput:
        # Strict adherence: use ONLY target_col
        if cfg.target_col not in df.columns:
            # Fail gracefully or raise error depending on policy
            return MetricOutput(scalars={}, tables={}, artifacts={})

        stats = compute_clinical_stats(
            df[cfg.target_col],
            hypo_threshold=self.params.hypo_threshold,
            hyper_threshold=self.params.hyper_threshold
        )

        # Flatten dictionary with metric name prefix
        # e.g., "basic_stats__mean", "basic_stats__tir"
        scalars = {f"{self.name}__{k}": v for k, v in stats.items()}

        return MetricOutput(
            scalars=scalars,
            tables={},
            artifacts={}
        )