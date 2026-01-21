from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from evaluation.metrics_core import compute_arx_delta_r2_linear
from ..types import EvaluationConfig, Metric, MetricOutput


@dataclass(frozen=True)
class ArxDeltaR2LinearParams:
    horizons_min: list[int]
    n_splits: int = 5
    alpha: float = 1.0
    min_samples: int = 200
    add_time_of_day: bool = True

    # Scalar flattening policy
    flatten_all_horizons: bool = True
    key_horizons_min: list[int] | None = None  # used only if flatten_all_horizons=False


class ArxDeltaR2LinearMetric(Metric):
    """
    Wrapper for ARX ΔR² (linear), multi-horizon.

    - Calls compute_arx_delta_r2_linear() which returns:
        by_horizon: DataFrame (one row per horizon)
        folds: DataFrame (long format)
    - Produces:
        tables["by_horizon"], tables["folds"]
        scalars flattened per horizon (all horizons or a chosen subset)
        artifacts["plot"] with a saved PNG if enabled by cfg
    """

    def __init__(
        self,
        *,
        params: ArxDeltaR2LinearParams,
        candidate_cols: list[str] | None = None,
        name: str = "arx_delta_r2_linear",
    ) -> None:
        self.name = name
        self.params = params
        self.candidate_cols = candidate_cols  # if None, fallback to cfg.cond_cols

    def compute(self, subject_id: int, df: pd.DataFrame, cfg: EvaluationConfig) -> MetricOutput:
        if cfg.lag_minutes is None or len(cfg.lag_minutes) == 0:
            raise ValueError(
                "ARX ΔR² requires cfg.lag_minutes to be set (target lag columns are prerequisites)."
            )

        candidates = self.candidate_cols if self.candidate_cols is not None else list(cfg.cond_cols)

        res = compute_arx_delta_r2_linear(
            df,
            target_col=cfg.target_col,
            candidate_cols=candidates,
            lag_minutes=list(cfg.lag_minutes),
            horizons_min=list(self.params.horizons_min),
            add_time_of_day=bool(self.params.add_time_of_day),
            time_col=cfg.time_col,
            n_splits=int(self.params.n_splits),
            alpha=float(self.params.alpha),
            min_samples=int(self.params.min_samples),
        )

        by_horizon = self._ensure_dataframe(res.get("by_horizon"), name="by_horizon")
        folds = self._ensure_dataframe(res.get("folds"), name="folds")

        tables = {"by_horizon": by_horizon, "folds": folds}
        scalars = self._flatten_scalars(by_horizon)

        artifacts: dict[str, str] = {}
        plot_path = self._maybe_save_plot(subject_id=subject_id, by_horizon=by_horizon, cfg=cfg)
        if plot_path is not None:
            artifacts["plot"] = str(plot_path)

        return MetricOutput(scalars=scalars, tables=tables, artifacts=artifacts)

    def _ensure_dataframe(self, obj: object, *, name: str) -> pd.DataFrame:
        if isinstance(obj, pd.DataFrame):
            return obj
        if obj is None:
            return pd.DataFrame()
        raise TypeError(f"{self.name}: expected '{name}' to be a DataFrame, got {type(obj).__name__}.")

    def _flatten_scalars(self, by_horizon: pd.DataFrame) -> dict[str, float]:
        """
        Flatten per-horizon rows into the per-subject scalar table.

        Policy:
        - If params.flatten_all_horizons=True: flatten every horizon.
        - Otherwise: flatten only params.key_horizons_min.
        - Always provide a couple of aggregate stats across horizons (mean/max of delta).
        """
        prefix = f"{self.name}__"
        out: dict[str, float] = {}

        if by_horizon.empty:
            out[f"{prefix}empty"] = 1.0
            return out

        # Decide which horizons to flatten
        if self.params.flatten_all_horizons:
            selected = by_horizon
        else:
            key = set(self.params.key_horizons_min or [])
            selected = by_horizon[by_horizon["horizon_min"].isin(key)]

        for _, row in selected.iterrows():
            h = int(row["horizon_min"])
            hprefix = f"{prefix}h{h}m__"

            out[f"{hprefix}r2_base_mean"] = float(row.get("r2_base_mean", np.nan))
            out[f"{hprefix}r2_aug_mean"] = float(row.get("r2_aug_mean", np.nan))
            out[f"{hprefix}delta_r2_mean"] = float(row.get("delta_r2_mean", np.nan))
            out[f"{hprefix}n_samples"] = float(row.get("n_samples", np.nan))

        # Aggregate scalars across horizons (useful for summary)
        deltas = pd.to_numeric(by_horizon.get("delta_r2_mean"), errors="coerce").to_numpy(dtype=float)
        r2b = pd.to_numeric(by_horizon.get("r2_base_mean"), errors="coerce").to_numpy(dtype=float)
        r2a = pd.to_numeric(by_horizon.get("r2_aug_mean"), errors="coerce").to_numpy(dtype=float)

        out[f"{prefix}delta_r2_mean__avg_over_horizons"] = float(np.nanmean(deltas)) if deltas.size else float("nan")
        out[f"{prefix}delta_r2_mean__max_over_horizons"] = float(np.nanmax(deltas)) if deltas.size else float("nan")
        out[f"{prefix}r2_base_mean__avg_over_horizons"] = float(np.nanmean(r2b)) if r2b.size else float("nan")
        out[f"{prefix}r2_aug_mean__avg_over_horizons"] = float(np.nanmean(r2a)) if r2a.size else float("nan")

        return out

    def _maybe_save_plot(
        self,
        *,
        subject_id: int,
        by_horizon: pd.DataFrame,
        cfg: EvaluationConfig,
    ) -> Path | None:
        if not cfg.per_subject_plots:
            return None
        if cfg.output_dir is None:
            # Fail-soft: plotting requested but no output_dir configured.
            return None
        if by_horizon.empty:
            return None

        out_dir = Path(cfg.output_dir) / self.name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{self.name}__subject_{subject_id}.png"

        fig = self._build_figure(subject_id=subject_id, by_horizon=by_horizon)
        try:
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
        finally:
            plt.close(fig)

        return out_path

    def _build_figure(self, *, subject_id: int, by_horizon: pd.DataFrame) -> plt.Figure:
        h = pd.to_numeric(by_horizon["horizon_min"], errors="coerce").to_numpy(dtype=float)
        r2b = pd.to_numeric(by_horizon.get("r2_base_mean"), errors="coerce").to_numpy(dtype=float)
        r2a = pd.to_numeric(by_horizon.get("r2_aug_mean"), errors="coerce").to_numpy(dtype=float)
        dlt = pd.to_numeric(by_horizon.get("delta_r2_mean"), errors="coerce").to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(9, 3.5))

        ax.plot(h, r2b, marker="o", label="R² baseline (mean)")
        ax.plot(h, r2a, marker="o", label="R² augmented (mean)")
        ax.plot(h, dlt, marker="o", label="ΔR² mean")

        ax.axhline(0.0, linewidth=1.0)
        ax.set_title(f"{self.name} – subject {subject_id}")
        ax.set_xlabel("Horizon (minutes)")
        ax.set_ylabel("Score")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        return fig
