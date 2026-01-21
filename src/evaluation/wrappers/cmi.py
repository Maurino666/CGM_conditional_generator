from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from metrics_core import compute_cmi_ksg
from ..types import EvaluationConfig, Metric, MetricOutput


@dataclass(frozen=True)
class CmiKsgParams:
    horizons_min: list[int]
    freq_min: int = 5
    add_time_of_day: bool = True

    k: int = 10
    metric: str = "chebyshev"
    jitter: float = 1e-6
    random_state: int = 0
    min_samples: int = 100
    clip_at_zero: bool = False

    # Scalar flattening policy
    flatten_all_horizons: bool = True
    key_horizons_min: list[int] | None = None  # used only if flatten_all_horizons=False


class CmiKsgMetric(Metric):
    """
    Wrapper for KSG Conditional Mutual Information over multiple horizons:

        I(X_t ; Y_{t+h} | Z_t)

    Where:
      - X_t = candidate columns (typically cfg.cond_cols)
      - Z_t = baseline conditioning columns (typically target lags from cfg.lag_minutes [+ TOD])
      - Y_{t+h} = future target (cfg.target_col shifted by step)

    Responsibilities:
    - Resolve X (candidate cols) and Z (base cols) from cfg/df.
    - Call compute_cmi_ksg(...) which returns a by-horizon DataFrame.
    - Map to MetricOutput:
        scalars: flattened per-horizon + aggregates
        tables: by_horizon
        artifacts: optional plot (cfg.per_subject_plots + cfg.output_dir)

    Notes:
    - Assumes evaluator already ensured:
        * correct time ordering
        * target lag columns exist as f"{target}_lag_{m}m"
        * (optional) tod_sin_24h / tod_cos_24h exist if desired
    """

    def __init__(
        self,
        *,
        params: CmiKsgParams,
        base_cols: list[str] | None = None,
        candidate_cols: list[str] | None = None,
        name: str = "cmi_ksg",
    ) -> None:
        self.name = name
        self.params = params
        self._base_cols_override = base_cols
        self._candidate_cols_override = candidate_cols

    def compute(self, subject_id: int, df: pd.DataFrame, cfg: EvaluationConfig) -> MetricOutput:
        base_cols = self._resolve_base_cols(df=df, cfg=cfg)
        candidate_cols = self._resolve_candidate_cols(df=df, cfg=cfg)

        by_horizon = compute_cmi_ksg(
            df,
            target_col=cfg.target_col,
            candidate_cols=candidate_cols,
            base_cols=base_cols,
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

        tables = {"by_horizon": by_horizon}
        scalars = self._flatten_scalars(by_horizon)

        artifacts: dict[str, str] = {}
        plot_path = self._maybe_save_plot(subject_id=subject_id, by_horizon=by_horizon, cfg=cfg)
        if plot_path is not None:
            artifacts["plot"] = str(plot_path)

        return MetricOutput(scalars=scalars, tables=tables, artifacts=artifacts)

    # ---------------- Column resolution ----------------
    def _resolve_base_cols(self, *, df: pd.DataFrame, cfg: EvaluationConfig) -> list[str]:
        """
        Z columns for conditioning.

        Default policy:
        - Use target lags derived from cfg.lag_minutes: f"{target}_lag_{m}m"
        - Do NOT add TOD here (the core handles TOD inclusion if add_time_of_day=True and columns exist).
        """
        if self._base_cols_override is not None:
            cols = [c for c in self._base_cols_override if c in df.columns]
            if len(cols) == 0:
                raise ValueError(f"{self.name}: base_cols override provided but none are present in df.")
            return cols

        if cfg.lag_minutes is None or len(cfg.lag_minutes) == 0:
            raise ValueError(f"{self.name}: requires cfg.lag_minutes to define conditioning lags (base_cols).")

        cols: list[str] = []
        for m in cfg.lag_minutes:
            col = f"{cfg.target_col}_lag_{int(m)}m"
            if col in df.columns:
                cols.append(col)

        if len(cols) == 0:
            raise ValueError(f"{self.name}: no baseline lag columns found in df.")
        return cols

    def _resolve_candidate_cols(self, *, df: pd.DataFrame, cfg: EvaluationConfig) -> list[str]:
        """
        X columns.

        Default policy:
        - Use cfg.cond_cols.
        - Pre-filter to existing df columns to keep the wrapper robust.
          (If you want strict behavior, remove the filtering and raise on missing.)
        """
        raw = self._candidate_cols_override if self._candidate_cols_override is not None else list(cfg.cond_cols)
        cols = [c for c in raw if c in df.columns]
        if len(cols) == 0:
            raise ValueError(f"{self.name}: no candidate columns found in df.")
        return cols

    # ---------------- Output mapping ----------------
    def _flatten_scalars(self, by_horizon: pd.DataFrame) -> dict[str, float]:
        """
        Flatten by-horizon CMI into per-subject scalars.

        Per horizon:
          - cmi_bits
          - n

        Aggregates:
          - mean and max CMI across horizons
          - min n across horizons (weakest coverage)
        """
        prefix = f"{self.name}__"
        out: dict[str, float] = {}

        if by_horizon.empty:
            out[f"{prefix}empty"] = 1.0
            return out

        if self.params.flatten_all_horizons:
            selected = by_horizon
        else:
            key = set(self.params.key_horizons_min or [])
            selected = by_horizon[by_horizon["horizon_min"].isin(key)]

        for _, row in selected.iterrows():
            h = int(row["horizon_min"])
            hprefix = f"{prefix}h{h}m__"
            out[f"{hprefix}cmi_bits"] = float(row.get("cmi_bits", np.nan))
            out[f"{hprefix}n"] = float(row.get("n", np.nan))

        cmi = pd.to_numeric(by_horizon.get("cmi_bits"), errors="coerce").to_numpy(dtype=float)
        n = pd.to_numeric(by_horizon.get("n"), errors="coerce").to_numpy(dtype=float)

        out[f"{prefix}cmi_bits__avg_over_horizons"] = float(np.nanmean(cmi)) if cmi.size else float("nan")
        out[f"{prefix}cmi_bits__max_over_horizons"] = float(np.nanmax(cmi)) if cmi.size else float("nan")
        out[f"{prefix}n__min_over_horizons"] = float(np.nanmin(n)) if n.size else float("nan")

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
        cmi = pd.to_numeric(by_horizon.get("cmi_bits"), errors="coerce").to_numpy(dtype=float)
        n = pd.to_numeric(by_horizon.get("n"), errors="coerce").to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(9, 3.5))

        ax.plot(h, cmi, marker="o", label="CMI (bits)")
        ax.axhline(0.0, linewidth=1.0)

        ax.set_title(f"{self.name} – subject {subject_id}")
        ax.set_xlabel("Horizon (minutes)")
        ax.set_ylabel("I(X;Y|Z) [bits]")
        ax.grid(True, alpha=0.3)

        # Second axis to visualize coverage
        ax2 = ax.twinx()
        ax2.plot(h, n, marker="o", label="n (valid rows)")
        ax2.set_ylabel("n")

        # Merge legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="best")

        return fig
