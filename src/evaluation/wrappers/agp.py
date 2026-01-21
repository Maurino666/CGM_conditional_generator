from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from evaluation.metrics_core import compute_agp
from ..types import Metric, MetricOutput, EvaluationConfig


@dataclass(frozen=True)
class AgpParams:
    freq: str = "5min"
    min_days_per_bin: int = 5
    clamp_range: tuple[float, float] | None = (39.0, 401.0)
    round_timestamps: bool = True
    aggfunc: str = "mean"


class AgpMetric(Metric):
    """
    AGP metric wrapper.

    - Computes AGP table via agp_core()
    - Provides scalar summaries
    - Optionally saves a per-subject plot if cfg.per_subject_plots is True and cfg.output_dir is set
    """

    def __init__(self, params: AgpParams | None = None, name: str = "agp") -> None:
        self.name = name
        self.params = params or AgpParams()

    def compute(self, subject_id: int, df: pd.DataFrame, cfg: EvaluationConfig) -> MetricOutput:
        cgm = self._extract_cgm_series(df, cfg)

        agp = compute_agp(
            cgm=cgm,
            freq=self.params.freq,
            min_days_per_bin=self.params.min_days_per_bin,
            clamp_range=self.params.clamp_range,
            round_timestamps=self.params.round_timestamps,
            aggfunc=self.params.aggfunc,
        )

        scalars = self._summarize_agp(agp)
        tables = {"by_time_of_day": agp}

        artifacts: dict[str, str] = {}
        plot_path = self._maybe_save_plot(subject_id=subject_id, agp=agp, cfg=cfg)
        if plot_path is not None:
            artifacts["plot"] = str(plot_path)

        return MetricOutput(scalars=scalars, tables=tables, artifacts=artifacts)

    def _extract_cgm_series(self, df: pd.DataFrame, cfg: EvaluationConfig) -> pd.Series:
        if cfg.target_col not in df.columns:
            raise ValueError(f"AGP: cfg.target_col='{cfg.target_col}' not found in DataFrame columns.")

        if cfg.time_col is not None:
            if cfg.time_col not in df.columns:
                raise ValueError(f"AGP: cfg.time_col='{cfg.time_col}' not found in DataFrame columns.")

            time_index = pd.to_datetime(df[cfg.time_col], errors="coerce")
            if time_index.isna().any():
                raise ValueError("AGP: cfg.time_col contains invalid datetimes.")

            values = pd.to_numeric(df[cfg.target_col], errors="coerce").to_numpy(dtype=float)
            cgm = pd.Series(values, index=pd.DatetimeIndex(time_index))
            return cgm.sort_index()

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("AGP requires a DatetimeIndex when cfg.time_col is None.")

        values = pd.to_numeric(df[cfg.target_col], errors="coerce")
        return values.sort_index()

    def _summarize_agp(self, agp: pd.DataFrame) -> dict[str, float]:
        p50 = agp["p50"].to_numpy(dtype=float) if "p50" in agp.columns else np.array([], dtype=float)
        iqr = agp["iqr"].to_numpy(dtype=float) if "iqr" in agp.columns else np.array([], dtype=float)

        valid = np.isfinite(p50)
        valid_ratio = float(np.mean(valid)) if valid.size > 0 else float("nan")

        iqr_valid = iqr[valid] if iqr.size == valid.size else iqr[np.isfinite(iqr)]
        p50_valid = p50[valid]

        prefix = f"{self.name}__"
        return {
            f"{prefix}valid_bin_ratio": valid_ratio,
            f"{prefix}iqr_mean": float(np.nanmean(iqr_valid)) if iqr_valid.size > 0 else float("nan"),
            f"{prefix}iqr_p95": float(np.nanpercentile(iqr_valid, 95)) if iqr_valid.size > 0 else float("nan"),
            f"{prefix}median_mean": float(np.nanmean(p50_valid)) if p50_valid.size > 0 else float("nan"),
        }

    def _maybe_save_plot(self, *, subject_id: int, agp: pd.DataFrame, cfg: EvaluationConfig) -> Path | None:
        """
        Save a per-subject AGP plot if enabled by cfg.
        Returns the saved path, or None if plotting is disabled / not possible.
        """
        if not cfg.per_subject_plots:
            return None
        if cfg.output_dir is None:
            # Fail-soft: plotting requested but no output_dir configured.
            return None

        out_dir = Path(cfg.output_dir) / self.name
        out_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{self.name}__subject_{subject_id}.png"
        out_path = out_dir / filename

        fig = self._build_agp_figure(agp=agp, subject_id=subject_id)
        try:
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
        finally:
            plt.close(fig)

        return out_path

    def _build_agp_figure(self, *, agp: pd.DataFrame, subject_id: int) -> plt.Figure:
        """
        Create a standard AGP plot:
        - median curve (p50)
        - shaded IQR (p25-p75)
        - optional p10-p90 band (light)
        """
        # Convert TimedeltaIndex to hours for a clean x-axis
        tod = agp.index
        hours = tod.total_seconds() / 3600.0

        p10 = agp["p10"].to_numpy(dtype=float)
        p25 = agp["p25"].to_numpy(dtype=float)
        p50 = agp["p50"].to_numpy(dtype=float)
        p75 = agp["p75"].to_numpy(dtype=float)
        p90 = agp["p90"].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(10, 4))

        # Bands: nan-aware via masks
        ax.fill_between(hours, p10, p90, alpha=0.15, label="p10–p90")
        ax.fill_between(hours, p25, p75, alpha=0.30, label="p25–p75 (IQR)")
        ax.plot(hours, p50, linewidth=2.0, label="median (p50)")

        ax.set_title(f"AGP – subject {subject_id}")
        ax.set_xlabel("Time of day (hours)")
        ax.set_ylabel("Glucose")

        ax.set_xlim(0.0, 24.0)
        ax.set_xticks([0, 6, 12, 18, 24])

        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        return fig