from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from metrics_core import compute_delta_r2_nonlinear_ab_decomposition
from metrics_core import TemporalCVSpec, RegressorFactory  # adjust import path
from ..types import EvaluationConfig, Metric, MetricOutput
from evaluation.wrappers.utils import resolve_feature_group


@dataclass(frozen=True)
class NonlinearDeltaR2ABParams:
    horizons_min: list[int]
    freq_min: int = 5

    # feature_groups keys in cfg.feature_groups
    group_a_key: str = "A"
    group_b_key: str = "B"

    add_time_of_day: bool = True
    cv_spec: TemporalCVSpec = TemporalCVSpec()
    random_state: int = 42
    min_samples: int | None = None
    clip_shared_at_zero: bool = True

    # model
    regressor_factory: RegressorFactory = None  # must be provided

    # outputs
    store_details_tables: bool = False
    plot_kind: str = "delta"  # "delta" or "r2"


class NonlinearDeltaR2ABDecompositionMetric(Metric):
    """
    Wrapper for compute_delta_r2_nonlinear_ab_decomposition (multi-horizon).

    - Baseline: target lags (created by Evaluator) (+ TOD optional inside core)
    - A/B blocks: resolved from cfg.feature_groups[group_a_key/group_b_key]
    - Outputs:
        scalars: flattened per horizon
        tables: by_horizon (+ optional details dump)
        artifacts: optional plot per subject
    """
    name = "delta_r2_nl_ab"

    def __init__(self, params: NonlinearDeltaR2ABParams) -> None:
        if params.regressor_factory is None:
            raise ValueError("NonlinearDeltaR2ABParams.regressor_factory must be provided.")
        if not params.horizons_min:
            raise ValueError("NonlinearDeltaR2ABParams.horizons_min must be non-empty.")
        self.params = params

    def compute(self, subject_id: int, df: pd.DataFrame, cfg: EvaluationConfig) -> MetricOutput:
        # baseline columns from lags ensured by Evaluator
        base_cols = self._resolve_base_cols(df, cfg)

        # Resolve A/B from cfg.feature_groups
        A_cols = resolve_feature_group(df, cfg, self.params.group_a_key)
        B_cols = resolve_feature_group(df, cfg, self.params.group_b_key)

        if len(A_cols) == 0 or len(B_cols) == 0:
            raise ValueError(
                f"{self.name}: empty feature groups after filtering. "
                f"A({self.params.group_a_key})={A_cols}, "
                f"B({self.params.group_b_key})={B_cols}"
            )

        out = compute_delta_r2_nonlinear_ab_decomposition(
            df,
            target_col=cfg.target_col,
            base_cols=base_cols,
            features_A=A_cols,
            features_B=B_cols,
            horizons_min=list(self.params.horizons_min),
            freq_min=int(self.params.freq_min),
            add_time_of_day=bool(self.params.add_time_of_day),
            cv_spec=self.params.cv_spec,
            regressor_factory=self.params.regressor_factory,
            random_state=int(self.params.random_state),
            min_samples=self.params.min_samples,
            clip_shared_at_zero=bool(self.params.clip_shared_at_zero),
        )

        by_horizon: pd.DataFrame = out["by_horizon"].copy()

        scalars = self._flatten_scalars(by_horizon)
        tables: dict[str, pd.DataFrame] = {"by_horizon": by_horizon}

        if self.params.store_details_tables:
            # Details is a nested dict; easiest is to store as a "long" table of keys/values
            details = out.get("details", {})
            tables["details_index"] = self._details_index_table(details)

        artifacts: dict[str, str] = {}
        if cfg.per_subject_plots and cfg.output_dir is not None:
            p = self._plot(subject_id, by_horizon, cfg.output_dir)
            artifacts["ab_plot"] = str(p)

        return MetricOutput(scalars=scalars, tables=tables, artifacts=artifacts)

    def _resolve_base_cols(self, df: pd.DataFrame, cfg: EvaluationConfig) -> list[str]:
        if cfg.lag_minutes is None or len(cfg.lag_minutes) == 0:
            raise ValueError(f"{self.name}: cfg.lag_minutes must be provided (baseline requires target lags).")

        cols: list[str] = []
        for m in cfg.lag_minutes:
            c = f"{cfg.target_col}_lag_{int(m)}m"
            if c in df.columns:
                cols.append(c)

        if len(cols) == 0:
            raise ValueError(
                f"{self.name}: no baseline lag columns found. "
                f"Expected columns like '{cfg.target_col}_lag_15m' etc."
            )
        return cols

    def _flatten_scalars(self, by_horizon: pd.DataFrame) -> dict[str, float]:
        """
        Flatten by-horizon outputs into scalars.

        Keys:
          <metric>__<field>__h<minutes>m
        """
        fields = [
            "n",
            "n_folds",
            "r2_base",
            "r2_baseA",
            "r2_baseB",
            "r2_baseAB",
            "delta_total",
            "unique_A",
            "unique_B",
            "shared",
        ]

        scalars: dict[str, float] = {}
        for _, row in by_horizon.iterrows():
            h = int(row["horizon_min"])
            for f in fields:
                key = f"{self.name}__{f}__h{h}m"
                v = row.get(f, np.nan)
                try:
                    scalars[key] = float(v)
                except Exception:
                    scalars[key] = float("nan")

        # compact summaries
        if "delta_total" in by_horizon.columns:
            vals = by_horizon["delta_total"].to_numpy(dtype=float)
            scalars[f"{self.name}__delta_total__mean_over_horizons"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan

        return scalars

    def _details_index_table(self, details: dict[int, dict[str, object]]) -> pd.DataFrame:
        """
        Turn nested details dict into a compact table.
        We avoid exploding huge nested objects; this is mainly a manifest.
        """
        rows: list[dict[str, object]] = []
        for h, d in details.items():
            rows.append(
                {
                    "horizon_min": int(h),
                    "has_details_A": bool(d.get("details_A")),
                    "has_details_B": bool(d.get("details_B")),
                    "has_details_AB": bool(d.get("details_AB")),
                    "used_cols_baseline": str(d.get("used_cols_baseline", "")),
                    "used_cols_A": str(d.get("used_cols_A", "")),
                    "used_cols_B": str(d.get("used_cols_B", "")),
                    "used_cols_AB": str(d.get("used_cols_AB", "")),
                }
            )
        return pd.DataFrame(rows).sort_values("horizon_min").reset_index(drop=True)

    def _plot(self, subject_id: int, by_horizon: pd.DataFrame, output_dir: Path) -> Path:
        subj_dir = Path(output_dir) / "artifacts" / f"subject_{subject_id}"
        subj_dir.mkdir(parents=True, exist_ok=True)

        kind = self.params.plot_kind
        path = subj_dir / f"{self.name}__{kind}.png"

        x = by_horizon["horizon_min"].to_numpy(dtype=int)

        plt.figure()

        if kind == "r2":
            for col in ["r2_base", "r2_baseA", "r2_baseB", "r2_baseAB"]:
                if col in by_horizon.columns:
                    plt.plot(x, by_horizon[col].to_numpy(dtype=float), marker="o", label=col)
            plt.ylabel("R² (walk-forward CV mean)")
            plt.title(f"{self.name} R² - subject {subject_id}")
        else:
            for col in ["delta_total", "unique_A", "unique_B", "shared"]:
                if col in by_horizon.columns:
                    plt.plot(x, by_horizon[col].to_numpy(dtype=float), marker="o", label=col)
            plt.ylabel("ΔR² (out-of-sample)")
            plt.title(f"{self.name} decomposition - subject {subject_id}")

        plt.xlabel("Horizon (minutes)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

        return path
