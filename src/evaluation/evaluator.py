from __future__ import annotations

import numpy as np
import pandas as pd

from .types import (
    EvaluationConfig,
    EvaluationResult,
    EvaluationTables,
    EvaluationArtifacts,
    Metric,
)
from .wrappers.utils import restore_nans_from_masks_global


class Evaluator:
    def __init__(self, cfg: EvaluationConfig, metrics: list[Metric], verbose: bool = True):
        self.cfg = cfg
        self.metrics = metrics
        self.verbose = verbose

    def evaluate(self, series: list[pd.DataFrame]) -> EvaluationResult:
        self._validate_inputs(series)

        artifacts = EvaluationArtifacts(per_subject={}, cohort={})
        tables = EvaluationTables(per_subject={})

        per_subject = self._compute_per_subject(series, artifacts=artifacts, tables=tables)
        summary = self._summarize(per_subject)

        metadata = {
            "n_subjects": len(series),
            "metrics": [m.name for m in self.metrics],
            "per_subject_plots": bool(self.cfg.per_subject_plots),
            "output_dir": str(self.cfg.output_dir) if self.cfg.output_dir is not None else None,
            "lag_minutes": list(self.cfg.lag_minutes) if self.cfg.lag_minutes is not None else None,
            "ensure_time_of_day": bool(self.cfg.ensure_time_of_day),
            "feature_groups": self.cfg.feature_groups,
        }

        return EvaluationResult(
            per_subject=per_subject,
            summary=summary,
            artifacts=artifacts,
            tables=tables,
            metadata=metadata,
        )

    def _validate_inputs(self, series: list[pd.DataFrame]) -> None:
        if len(series) == 0:
            raise ValueError("series must be non-empty.")

    def _compute_per_subject(
        self,
        series: list[pd.DataFrame],
        artifacts: EvaluationArtifacts,
        tables: EvaluationTables,
    ) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        n_subjects = len(series)

        for i, df in enumerate(series):
            subject_id = i if self.cfg.subject_id_col is None else self._get_subject_id_as_int(df, fallback=i)

            if self.verbose:
                print(f"[Evaluator] Subject {i + 1}/{n_subjects} (id={subject_id})...")

            df = restore_nans_from_masks_global(df, self.cfg)

            # Ensure shared derived features exist once per subject
            df_feat = self._ensure_derived_features(df)

            row: dict[str, object] = {"subject_id": subject_id}

            artifacts.per_subject.setdefault(str(subject_id), {})
            tables.per_subject.setdefault(str(subject_id), {})

            for j, metric in enumerate(self.metrics):
                # Filter single metrics
                if metric.requires_synth:
                    if self.cfg.comparison_target_col is None:
                        # Skip metric that requires synthetic
                        continue
                    if self.cfg.comparison_target_col not in df.columns:
                        # Defensive Skip: synth column not in df
                        if self.verbose:
                            print(f"  [Skip] {metric.name} requires '{self.cfg.comparison_target_col}' (missing).")
                        continue

                if self.verbose:
                    print(f"  - ({j + 1}/{len(self.metrics)}) {metric.name}... ")

                try:
                    output = metric.compute(subject_id, df_feat, self.cfg)
                    row.update(output.scalars)

                    if output.tables:
                        tables.per_subject[str(subject_id)][metric.name] = output.tables
                    if output.artifacts:
                        artifacts.per_subject[str(subject_id)][metric.name] = output.artifacts
                except Exception as e:
                    # Fail-safe: keep the pipeline running
                    row[f"{metric.name}__error"] = 1.0
                    # Store the exception message (compact)
                    row[f"{metric.name}__error_msg"] = f"{type(e).__name__}: {e}"

                    if self.verbose:
                        print(f"FAIL ({type(e).__name__}: {e})")

            rows.append(row)

        return pd.DataFrame(rows)

    def _get_subject_id_as_int(self, df: pd.DataFrame, fallback: int) -> int:
        if self.cfg.subject_id_col and self.cfg.subject_id_col in df.columns:
            val = df[self.cfg.subject_id_col].iloc[0]
            try:
                return int(val)
            except Exception:
                return fallback
        return fallback

    def _ensure_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Work on a copy to avoid mutating the original dataset object
        out = df.copy()

        # Sort by time and get a DatetimeIndex for feature construction
        out, time_index = self._sort_and_get_time_index(out)

        # Ensure target lags exist
        if self.cfg.lag_minutes:
            out = self._ensure_target_lags(out, time_index)

        # Ensure time-of-day features exist
        if self.cfg.ensure_time_of_day:
            out = self._ensure_time_of_day(out, time_index)

        return out

    def _sort_and_get_time_index(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
        if self.cfg.time_col is not None:
            if self.cfg.time_col not in df.columns:
                raise ValueError(f"cfg.time_col='{self.cfg.time_col}' not found in DataFrame columns.")
            t = pd.to_datetime(df[self.cfg.time_col])
            if t.isna().any():
                raise ValueError("time_col contains invalid datetimes.")
            df = df.sort_values(self.cfg.time_col)
            return df, pd.DatetimeIndex(pd.to_datetime(df[self.cfg.time_col]))

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("A DatetimeIndex is required when cfg.time_col is None.")
        df = df.sort_index()
        return df, df.index

    def _infer_dt_minutes(self, time_index: pd.DatetimeIndex) -> float:
        dt = pd.Series(time_index).diff().dt.total_seconds().div(60.0).bfill()
        dt_min = float(dt.median())
        if not np.isfinite(dt_min) or dt_min <= 0:
            raise ValueError("Cannot infer a positive sampling step.")
        return dt_min

    def _ensure_target_lags(self, df: pd.DataFrame, time_index: pd.DatetimeIndex) -> pd.DataFrame:
        dt_min = self._infer_dt_minutes(time_index)

        # List of columns "to lag"
        cols_to_lag = [self.cfg.target_col]

        # Add synthetic col only if present
        if self.cfg.comparison_target_col is not None and self.cfg.comparison_target_col in df.columns:
            cols_to_lag.append(self.cfg.comparison_target_col)

        for base_col in cols_to_lag:
            for m in self.cfg.lag_minutes or []:
                col_name = f"{base_col}_lag_{int(m)}m"
                if col_name in df.columns:
                    continue

                steps = max(1, int(round(float(m) / dt_min)))
                df[col_name] = df[base_col].shift(steps)

        return df

    def _ensure_time_of_day(self, df: pd.DataFrame, time_index: pd.DatetimeIndex) -> pd.DataFrame:
        need_sin = "tod_sin_24h" not in df.columns
        need_cos = "tod_cos_24h" not in df.columns
        if not (need_sin or need_cos):
            return df

        dates = time_index.normalize()
        tod_seconds = (time_index - dates).total_seconds()
        angle = 2.0 * np.pi * (tod_seconds / (24.0 * 3600.0))

        if need_sin:
            df["tod_sin_24h"] = np.sin(angle)
        if need_cos:
            df["tod_cos_24h"] = np.cos(angle)

        return df

    def _summarize(self, per_subject: pd.DataFrame) -> pd.DataFrame:
        numeric = per_subject.select_dtypes(include=[np.number])

        if numeric.empty:
            return pd.DataFrame(columns=["metric", "mean", "median", "std"])

        summary = numeric.agg(["mean", "median", "std"]).T
        summary = summary.reset_index().rename(columns={"index": "metric"})
        return summary
