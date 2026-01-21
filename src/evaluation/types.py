from pathlib import Path

import pandas as pd

from dataclasses import dataclass



@dataclass(frozen=True)
class EvaluationConfig:
    target_col: str
    cond_cols: list[str]
    time_col: str | None = None  # If None, assume the index represents time
    subject_id_col: str | None = None  # If present, assume it is constant within each subject DataFrame

    feature_groups: dict[str, list[str]] | None = None

    # Shared derived features
    lag_minutes: list[int] | None = None  # Target lags to ensure exist (e.g., [5, 10, 15, 30, 60])
    ensure_time_of_day: bool = True  # Ensure tod_sin_24h and tod_cos_24h exist

    # Plot/export controls
    output_dir: Path | None = None
    per_subject_plots: bool = False

@dataclass
class EvaluationArtifacts:
    # per_subject[subject_id][metric_name] -> dict[str, str]
    per_subject: dict[str, dict[str, dict[str, str]]]

    # cohort-level artifacts, if you generate them elsewhere
    cohort: dict[str, str]


@dataclass
class EvaluationTables:
    # per_subject[subject_id][metric_name] -> dict[str, pd.DataFrame]
    per_subject: dict[str, dict[str, dict[str, pd.DataFrame]]]


@dataclass
class EvaluationResult:
    per_subject: pd.DataFrame
    summary: pd.DataFrame
    artifacts: EvaluationArtifacts
    tables: EvaluationTables
    metadata: dict[str, object]



@dataclass
class MetricOutput:
    # Scalar values that will be stored in per-subject tables
    scalars: dict[str, float]

    # Optional tabular outputs (e.g., AGP curve) keyed by a short name
    tables: dict[str, pd.DataFrame]

    # Optional artifacts (e.g., plot file paths) keyed by a short name
    artifacts: dict[str, str]



class Metric:
    """
    A metric computes outputs for a single subject DataFrame.

    - scalars: go into per-subject evaluation tables
    - tables: optional richer outputs (e.g., time-of-day curves)
    - artifacts: optional saved files (e.g., PNG plots), usually controlled by cfg
    """
    name: str

    def compute(self, subject_id: int, df: pd.DataFrame, cfg: EvaluationConfig) -> MetricOutput:
        raise NotImplementedError


class ScalarCallableMetric(Metric):
    """
    Adapter for metrics that only produce scalar outputs.
    """
    def __init__(self, name: str, fn):
        # fn signature: (df: pd.DataFrame, cfg: EvaluationConfig, subject_id: int) -> dict[str, float]
        self.name = name
        self._fn = fn

    def compute(self, subject_id: int, df: pd.DataFrame, cfg: EvaluationConfig) -> MetricOutput:
        scalars = self._fn(subject_id, df, cfg)
        return MetricOutput(scalars=scalars, tables={}, artifacts={})