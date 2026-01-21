from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from ..types import EvaluationConfig, Metric, MetricOutput
from evaluation.metrics_core import compute_distributional_distance


@dataclass(frozen=True)
class SimilarityParams:
    method: str = "wasserstein"


class SimilarityMetric(Metric):
    """
    Compares the distribution of Real vs. Synthetic target values.
    Requires 'synth_col' to be present in the dataframe.
    """
    name: str = "similarity"
    requires_synthetic: bool = True  # This flags the Evaluator!

    def __init__(self, params: SimilarityParams = SimilarityParams()) -> None:
        self.params = params

    def compute(self, subject_id: int, df: pd.DataFrame, cfg: EvaluationConfig) -> MetricOutput:
        if cfg.synth_col is None:
            raise ValueError("SimilarityMetric requires 'synth_col' in config.")

        if cfg.synth_col not in df.columns:
            # Skip gracefully if synth column missing
            return MetricOutput(scalars={}, tables={}, artifacts={})

        res = compute_distributional_distance(
            df,
            target_col=cfg.target_col,
            synth_col=cfg.synth_col,
            method=self.params.method
        )

        # Flatten for the report
        scalars = {
            f"{self.name}__{k}": v for k, v in res.items()
        }

        return MetricOutput(scalars=scalars, tables={}, artifacts={})