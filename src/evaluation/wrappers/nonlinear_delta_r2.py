from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from evaluation.metrics_core import TemporalCVSpec, RegressorFactory, compute_delta_r2_nonlinear
from .utils import filter_valid_features
from ..types import EvaluationConfig, Metric, MetricOutput


@dataclass(frozen=True)
class DeltaR2NonlinearParams:
    horizons_min: list[int]
    freq_min: int = 5
    add_time_of_day: bool = True
    cv_spec: TemporalCVSpec = TemporalCVSpec()
    regressor_factory: RegressorFactory | None = None
    random_state: int = 42
    min_samples: int | None = None

    # Scalar flattening policy
    flatten_all_horizons: bool = True
    key_horizons_min: list[int] | None = None  # used only if flatten_all_horizons=False


class DeltaR2NonlinearMetric(Metric):
    """
    Wrapper for nonlinear out-of-sample ΔR² over multiple horizons.

    Baseline:
      y(t+h) ~ base_cols (+TOD)
    Augmented:
      y(t+h) ~ baseline + candidate_cols

    Responsibilities:
    - Resolve baseline columns (usually target lags) from cfg / df.
    - Resolve candidate columns (usually cfg.cond_cols) from cfg / df.
    - Call compute_delta_r2_nonlinear(...) which returns by_horizon + folds.
    - Map to MetricOutput:
        scalars: flattened per-horizon + aggregates
        tables: by_horizon, folds
        artifacts: optional plot (cfg.per_subject_plots + cfg.output_dir)

    Notes:
    - This wrapper assumes the evaluator already ensured:
        * correct time ordering
        * creation of target lag columns named f"{target}_lag_{m}m"
        * (optionally) TOD columns
    """

    def __init__(
        self,
        *,
        params: DeltaR2NonlinearParams,
        base_cols: list[str] | None = None,
        candidate_cols: list[str] | None = None,
        name: str = "delta_r2_nonlinear",
    ) -> None:
        self.name = name
        self.params = params

        # Optional overrides
        self._base_cols_override = base_cols
        self._candidate_cols_override = candidate_cols

    def compute(self, subject_id: int, df: pd.DataFrame, cfg: EvaluationConfig) -> MetricOutput:
        base_cols = self._resolve_base_cols(df=df, cfg=cfg)
        candidate_cols = self._resolve_candidate_cols(df=df, cfg=cfg)

        if cfg.masked_dataframes:
            candidate_cols = filter_valid_features(df, candidate_cols)
            if not candidate_cols:
                return MetricOutput(
                    scalars={f"{self.name}__skipped": 1.0},
                    tables={},
                    artifacts={}
                )

        res = compute_delta_r2_nonlinear(
            df,
            target_col=cfg.target_col,
            base_cols=base_cols,
            candidate_cols=candidate_cols,
            horizons_min=list(self.params.horizons_min),
            freq_min=int(self.params.freq_min),
            add_time_of_day=bool(self.params.add_time_of_day),
            cv_spec=self.params.cv_spec,
            regressor_factory=self.params.regressor_factory,
            random_state=int(self.params.random_state),
            min_samples=self.params.min_samples,
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

    # ---------------- Column resolution ----------------
    def _resolve_base_cols(self, *, df: pd.DataFrame, cfg: EvaluationConfig) -> list[str]:
        if self._base_cols_override is not None:
            cols = [c for c in self._base_cols_override if c in df.columns]
            if len(cols) == 0:
                raise ValueError(f"{self.name}: base_cols override provided but none are present in df.")
            return cols

        if cfg.lag_minutes is None or len(cfg.lag_minutes) == 0:
            raise ValueError(
                f"{self.name}: requires cfg.lag_minutes to be set (baseline lag features are prerequisites)."
            )

        cols: list[str] = []
        for m in cfg.lag_minutes:
            col = f"{cfg.target_col}_lag_{int(m)}m"
            if col in df.columns:
                cols.append(col)

        if len(cols) == 0:
            raise ValueError(f"{self.name}: no baseline lag columns found in df.")
        return cols

    def _resolve_candidate_cols(self, *, df: pd.DataFrame, cfg: EvaluationConfig) -> list[str]:
        raw = self._candidate_cols_override if self._candidate_cols_override is not None else list(cfg.cond_cols)

        # IMPORTANT: for nonlinear core we keep it strict; we pre-filter to existing cols to avoid hard errors.
        cols = [c for c in raw if c in df.columns]
        if len(cols) == 0:
            raise ValueError(f"{self.name}: no candidate columns found in df.")
        return cols

    # ---------------- Output mapping ----------------
    def _ensure_dataframe(self, obj: object, *, name: str) -> pd.DataFrame:
        if isinstance(obj, pd.DataFrame):
            return obj
        if obj is None:
            return pd.DataFrame()
        raise TypeError(f"{self.name}: expected '{name}' to be a DataFrame, got {type(obj).__name__}.")

    def _flatten_scalars(self, by_horizon: pd.DataFrame) -> dict[str, float]:
        """
        Flatten multi-horizon results into per-subject scalar outputs.

        We export for each selected horizon:
          - r2_base_mean
          - r2_aug_mean
          - delta_r2_mean
          - n
          - n_folds

        Plus aggregates across horizons (avg/max delta, avg r2).
        """
        prefix = f"{self.name}__"
        out: dict[str, float] = {}

        if by_horizon.empty:
            out[f"{prefix}empty"] = 1.0
            return out

        # Select horizons for flattening
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
            out[f"{hprefix}n"] = float(row.get("n", np.nan))
            out[f"{hprefix}n_folds"] = float(row.get("n_folds", np.nan))

        # Aggregates
        dlt = pd.to_numeric(by_horizon.get("delta_r2_mean"), errors="coerce").to_numpy(dtype=float)
        r2b = pd.to_numeric(by_horizon.get("r2_base_mean"), errors="coerce").to_numpy(dtype=float)
        r2a = pd.to_numeric(by_horizon.get("r2_aug_mean"), errors="coerce").to_numpy(dtype=float)

        out[f"{prefix}delta_r2_mean__avg_over_horizons"] = float(np.nanmean(dlt)) if dlt.size else float("nan")
        out[f"{prefix}delta_r2_mean__max_over_horizons"] = float(np.nanmax(dlt)) if dlt.size else float("nan")
        out[f"{prefix}r2_base_mean__avg_over_horizons"] = float(np.nanmean(r2b)) if r2b.size else float("nan")
        out[f"{prefix}r2_aug_mean__avg_over_horizons"] = float(np.nanmean(r2a)) if r2a.size else float("nan")

        return out

    # ---------------- Plotting ----------------
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

        ax.plot(h, r2b, marker="o", label="R² baseline (oos mean)")
        ax.plot(h, r2a, marker="o", label="R² augmented (oos mean)")
        ax.plot(h, dlt, marker="o", label="ΔR² (oos mean)")

        ax.axhline(0.0, linewidth=1.0)
        ax.set_title(f"{self.name} – subject {subject_id}")
        ax.set_xlabel("Horizon (minutes)")
        ax.set_ylabel("Score")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        return fig
