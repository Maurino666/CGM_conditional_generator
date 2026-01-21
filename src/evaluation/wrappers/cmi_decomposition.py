from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from evaluation.metrics_core import compute_cmi_ksg_decomposition  # adjust import path
from ..types import EvaluationConfig, Metric, MetricOutput
from evaluation.wrappers.utils import resolve_feature_group, filter_valid_features


@dataclass(frozen=True)
class CmiKsgDecompositionParams:
    horizons_min: list[int]
    freq_min: int = 5

    # cfg.feature_groups keys
    group_a_key: str = "A"
    group_b_key: str = "B"

    add_time_of_day: bool = True
    k: int = 10
    metric: str = "chebyshev"
    jitter: float = 1e-6
    random_state: int = 0
    min_samples: int = 100
    clip_at_zero: bool = True

    # outputs
    plot: bool = True
    plot_kind: str = "decomp"  # "decomp" or "total_only"


class CmiKsgDecompositionMetric(Metric):
    """
    Wrapper for compute_cmi_ksg_decomposition (multi-horizon KSG-CMI AB decomposition).

    - base_cols: target lags (ensured by Evaluator)
    - features_A/B: resolved from cfg.feature_groups
    - returns:
        scalars: flattened per horizon
        tables: {'by_horizon': df}
        artifacts: optional plot path when cfg.per_subject_plots=True
    """
    name = "cmi_ksg_ab"

    def __init__(self, params: CmiKsgDecompositionParams) -> None:
        if not params.horizons_min:
            raise ValueError("CmiKsgDecompositionParams.horizons_min must be non-empty.")
        self.params = params

    def compute(self, subject_id: int, df: pd.DataFrame, cfg: EvaluationConfig) -> MetricOutput:
        base_cols = self._resolve_base_cols(df, cfg)

        features_a = resolve_feature_group(df, cfg, self.params.group_a_key)
        features_b = resolve_feature_group(df, cfg, self.params.group_b_key)

        if len(features_a) == 0 or len(features_b) == 0:
            raise ValueError(
                f"{self.name}: empty feature groups after filtering. "
                f"A({self.params.group_a_key})={features_a}, "
                f"B({self.params.group_b_key})={features_b}"
            )

        if cfg.masked_dataframes:
            features_a = filter_valid_features(df, features_a)
            features_b = filter_valid_features(df, features_b)

            if len(features_a) == 0 or len(features_b) == 0:
                return MetricOutput(
                    scalars={f"{self.name}__skipped_missing_group": 1.0},
                    tables={}, artifacts={}
                )

        by_h = compute_cmi_ksg_decomposition(
            df,
            target_col=cfg.target_col,
            base_cols=base_cols,
            features_A=features_a,
            features_B=features_b,
            horizons_min=list(self.params.horizons_min),
            freq_min=int(self.params.freq_min),
            add_time_of_day=bool(self.params.add_time_of_day),
            k=int(self.params.k),
            metric=str(self.params.metric),
            jitter=float(self.params.jitter),
            random_state=int(self.params.random_state),
            min_samples=int(self.params.min_samples),
            clip_at_zero=bool(self.params.clip_at_zero),
        )

        tables = {"by_horizon": by_h}
        scalars = self._flatten_scalars(by_h)

        artifacts: dict[str, str] = {}
        if (
            self.params.plot
            and cfg.per_subject_plots
            and cfg.output_dir is not None
        ):
            p = self._plot(subject_id, by_h, cfg.output_dir)
            artifacts["cmi_ab_plot"] = str(p)

        return MetricOutput(scalars=scalars, tables=tables, artifacts=artifacts)

    def _resolve_base_cols(self, df: pd.DataFrame, cfg: EvaluationConfig) -> list[str]:
        if not cfg.lag_minutes:
            raise ValueError(f"{self.name}: cfg.lag_minutes must be provided (baseline requires target lags).")

        cols: list[str] = []
        for m in cfg.lag_minutes:
            c = f"{cfg.target_col}_lag_{int(m)}m"
            if c in df.columns:
                cols.append(c)

        if not cols:
            raise ValueError(
                f"{self.name}: no baseline lag columns found. "
                f"Expected columns like '{cfg.target_col}_lag_15m' etc."
            )
        return cols

    def _flatten_scalars(self, by_h: pd.DataFrame) -> dict[str, float]:
        """
        Flatten by-horizon outputs into scalars:
          <metric>__<field>__h<minutes>m
        """
        fields = ["n", "I_total", "unique_A", "unique_B", "shared_raw", "synergy", "overlap"]

        scalars: dict[str, float] = {}
        for _, row in by_h.iterrows():
            h = int(row["horizon_min"])
            for f in fields:
                key = f"{self.name}__{f}__h{h}m"
                v = row.get(f, np.nan)
                try:
                    scalars[key] = float(v)
                except Exception:
                    scalars[key] = float("nan")

        # handy summaries
        if "I_total" in by_h.columns:
            vals = by_h["I_total"].to_numpy(dtype=float)
            scalars[f"{self.name}__I_total__mean_over_horizons"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
        if "synergy" in by_h.columns:
            vals = by_h["synergy"].to_numpy(dtype=float)
            scalars[f"{self.name}__synergy__mean_over_horizons"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan

        return scalars

    def _plot(self, subject_id: int, by_h: pd.DataFrame, output_dir: Path) -> Path:
        subj_dir = Path(output_dir) / "artifacts" / f"subject_{subject_id}"
        subj_dir.mkdir(parents=True, exist_ok=True)

        kind = self.params.plot_kind
        path = subj_dir / f"{self.name}__{kind}.png"

        x = by_h["horizon_min"].to_numpy(dtype=int)

        plt.figure()

        if kind == "total_only":
            plt.plot(x, by_h["I_total"].to_numpy(dtype=float), marker="o", label="I_total")
            plt.ylabel("CMI (bits)")
            plt.title(f"{self.name} I_total - subject {subject_id}")
        else:
            for col in ["I_total", "unique_A", "unique_B", "synergy", "overlap"]:
                if col in by_h.columns:
                    plt.plot(x, by_h[col].to_numpy(dtype=float), marker="o", label=col)
            plt.ylabel("CMI (bits)")
            plt.title(f"{self.name} decomposition - subject {subject_id}")

        plt.xlabel("Horizon (minutes)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

        return path
