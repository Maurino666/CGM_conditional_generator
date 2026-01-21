from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from metrics_core import compute_arx_delta_r2_linear_ab_decomposition
from ..types import EvaluationConfig, Metric, MetricOutput
from evaluation.wrappers.utils import resolve_feature_group


@dataclass(frozen=True)
class ArxDeltaR2LinearABParams:
    """
    Parameters for ARX ΔR² linear AB decomposition wrapper.
    """
    horizons_min: list[int]

    # Feature group keys inside cfg.feature_groups
    group_a_key: str = "A"
    group_b_key: str = "B"

    # Core options
    add_time_of_day: bool = True
    n_splits: int = 5
    alpha: float = 1.0
    min_samples: int = 200
    clip_shared_at_zero: bool = True

    # Output controls
    store_details_table: bool = False  # If True, store a long table with some per-horizon details


class ArxDeltaR2LinearABMetric(Metric):
    """
    Wrapper for compute_arx_delta_r2_linear_ab_decomposition.

    Responsibilities:
    - Resolve feature groups A/B from cfg.feature_groups (filtered to df columns)
    - Call the multi-horizon core
    - Flatten key outputs into scalars (per horizon)
    - Optionally produce a per-subject plot (under cfg.output_dir) if enabled
    """
    name = "arx_delta_r2_linear_ab"

    def __init__(self, params: ArxDeltaR2LinearABParams) -> None:
        self.params = params

    def compute(self, subject_id: int, df: pd.DataFrame, cfg: EvaluationConfig) -> MetricOutput:
        # Resolve A/B columns from config groups and keep only columns present in df
        features_a = resolve_feature_group(df, cfg, self.params.group_a_key)
        features_b = resolve_feature_group(df, cfg, self.params.group_b_key)

        if len(features_a) == 0 or len(features_b) == 0:
            raise ValueError(
                f"{self.name}: empty feature groups after filtering. "
                f"A({self.params.group_a_key})={features_a}, "
                f"B({self.params.group_b_key})={features_b}"
            )

        if cfg.lag_minutes is None or len(cfg.lag_minutes) == 0:
            raise ValueError(f"{self.name}: cfg.lag_minutes must be provided for baseline construction.")

        out = compute_arx_delta_r2_linear_ab_decomposition(
            df,
            target_col=cfg.target_col,
            lag_minutes=list(cfg.lag_minutes),
            horizons_min=list(self.params.horizons_min),
            features_A=features_a,
            features_B=features_b,
            add_time_of_day=bool(self.params.add_time_of_day),
            time_col=cfg.time_col,
            n_splits=int(self.params.n_splits),
            alpha=float(self.params.alpha),
            min_samples=int(self.params.min_samples),
            clip_shared_at_zero=bool(self.params.clip_shared_at_zero),
        )

        by_horizon = out["by_horizon"].copy()

        scalars = self._flatten_scalars(by_horizon)
        tables: dict[str, pd.DataFrame] = {"by_horizon": by_horizon}

        if self.params.store_details_table:
            details_table = self._build_details_table(out.get("details", {}))
            tables["details"] = details_table

        artifacts: dict[str, str] = {}
        if cfg.per_subject_plots and cfg.output_dir is not None:
            plot_path = self._plot_decomposition(subject_id, by_horizon, cfg.output_dir)
            artifacts["decomposition_plot"] = str(plot_path)

        return MetricOutput(scalars=scalars, tables=tables, artifacts=artifacts)

    def _flatten_scalars(self, by_horizon: pd.DataFrame) -> dict[str, float]:
        """
        Flatten key columns per horizon into scalar outputs.

        Output naming convention:
          <metric>__<field>__h<minutes>
        """
        fields = [
            "delta_total",
            "unique_A",
            "unique_B",
            "shared",
            "n_rows_common",
            "dA_mean",
            "dB_mean",
            "dAB_mean",
        ]

        scalars: dict[str, float] = {}

        for _, row in by_horizon.iterrows():
            h = int(row["horizon_min"])
            for f in fields:
                key = f"{self.name}__{f}__h{h}m"
                val = row.get(f, np.nan)
                try:
                    scalars[key] = float(val)
                except Exception:
                    scalars[key] = float("nan")

        # Optional compact summaries across horizons (handy for quick inspection)
        if "delta_total" in by_horizon.columns:
            vals = by_horizon["delta_total"].to_numpy(dtype=float)
            scalars[f"{self.name}__delta_total__mean_over_horizons"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
            scalars[f"{self.name}__delta_total__min_over_horizons"] = float(np.nanmin(vals)) if np.isfinite(vals).any() else np.nan
            scalars[f"{self.name}__delta_total__max_over_horizons"] = float(np.nanmax(vals)) if np.isfinite(vals).any() else np.nan

        return scalars

    def _build_details_table(self, details: dict[int, dict[str, object]]) -> pd.DataFrame:
        """
        Convert the 'details' dict into a light long-form table.

        This is intentionally conservative: it avoids storing fold-level arrays unless you want to.
        """
        rows: list[dict[str, object]] = []
        for h, pack in details.items():
            rows.append(
                {
                    "horizon_min": int(h),
                    "n_rows_common": int(pack.get("n_rows_common", 0)),
                    "used_cols_baseline": str(pack.get("used_cols_baseline", [])),
                    "used_cols_A": str(pack.get("used_cols_A", [])),
                    "used_cols_B": str(pack.get("used_cols_B", [])),
                    "used_cols_AB": str(pack.get("used_cols_AB", [])),
                }
            )
        if not rows:
            return pd.DataFrame(columns=["horizon_min", "n_rows_common", "used_cols_baseline", "used_cols_A", "used_cols_B", "used_cols_AB"])
        return pd.DataFrame(rows).sort_values("horizon_min").reset_index(drop=True)

    def _plot_decomposition(self, subject_id: int, by_horizon: pd.DataFrame, output_dir: Path) -> Path:
        """
        Save a simple decomposition plot per subject:
          delta_total, unique_A, unique_B, shared vs horizon.
        """
        subj_dir = Path(output_dir) / "artifacts" / f"subject_{subject_id}"
        subj_dir.mkdir(parents=True, exist_ok=True)

        path = subj_dir / f"{self.name}__decomposition.png"

        x = by_horizon["horizon_min"].to_numpy(dtype=int)

        plt.figure()
        for col in ["delta_total", "unique_A", "unique_B", "shared"]:
            if col in by_horizon.columns:
                y = by_horizon[col].to_numpy(dtype=float)
                plt.plot(x, y, marker="o", label=col)

        plt.xlabel("Horizon (minutes)")
        plt.ylabel("ΔR² contribution")
        plt.title(f"{self.name} - subject {subject_id}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

        return path
