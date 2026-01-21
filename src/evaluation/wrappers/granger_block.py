from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from metrics_core import compute_granger_block
from ..types import EvaluationConfig, Metric, MetricOutput


@dataclass(frozen=True)
class GrangerBlockParams:
    horizons_min: list[int]
    freq_min: int = 5  # expected sampling grid for shift-based horizon
    add_time_of_day: bool = True
    match_n: bool = True
    min_samples: int = 200
    clip_partial_r2_at_zero: bool = True

    # Flattening policy
    flatten_all_horizons: bool = True
    key_horizons_min: list[int] | None = None  # used only if flatten_all_horizons=False


class GrangerBlockFTestMetric(Metric):
    """
    Wrapper for Granger-style in-sample block F-test (multi-horizon).

    Reduced model:
      y(t+h) ~ base_cols (+TOD if enabled)
    Full model:
      y(t+h) ~ base_cols (+TOD) + block_cols

    Notes:
    - The Evaluator is responsible for ensuring time ordering, target lags, and TOD features.
    - This wrapper only selects columns, calls the core, and maps outputs + plotting.
    """

    def __init__(
        self,
        *,
        params: GrangerBlockParams,
        base_cols: list[str] | None = None,
        block_cols: list[str] | None = None,
        name: str = "granger_block_f_test",
    ) -> None:
        self.name = name
        self.params = params

        # If base_cols is None, we build it from cfg.lag_minutes at compute-time.
        self._base_cols_override = base_cols

        # If block_cols is None, we use cfg.cond_cols at compute-time.
        self._block_cols_override = block_cols

    def compute(self, subject_id: int, df: pd.DataFrame, cfg: EvaluationConfig) -> MetricOutput:
        base_cols = self._resolve_base_cols(df=df, cfg=cfg)
        block_cols = self._resolve_block_cols(df=df, cfg=cfg)

        res = compute_granger_block(
            df,
            target_col=cfg.target_col,
            base_cols=base_cols,
            block_cols=block_cols,
            horizons_min=list(self.params.horizons_min),
            freq_min=int(self.params.freq_min),
            add_time_of_day=bool(self.params.add_time_of_day),
            match_n=bool(self.params.match_n),
            min_samples=int(self.params.min_samples),
            clip_partial_r2_at_zero=bool(self.params.clip_partial_r2_at_zero),
        )

        by_horizon = self._ensure_dataframe(res.get("by_horizon"), name="by_horizon")

        tables = {"by_horizon": by_horizon}
        scalars = self._flatten_scalars(by_horizon)

        artifacts: dict[str, str] = {}
        plot_path = self._maybe_save_plot(subject_id=subject_id, by_horizon=by_horizon, cfg=cfg)
        if plot_path is not None:
            artifacts["plot"] = str(plot_path)

        return MetricOutput(scalars=scalars, tables=tables, artifacts=artifacts)

    def _resolve_base_cols(self, *, df: pd.DataFrame, cfg: EvaluationConfig) -> list[str]:
        if self._base_cols_override is not None:
            return [c for c in self._base_cols_override if c in df.columns]

        if cfg.lag_minutes is None or len(cfg.lag_minutes) == 0:
            raise ValueError(
                "Granger block test requires baseline lag features. "
                "Set cfg.lag_minutes (and ensure lag columns exist)."
            )

        cols: list[str] = []
        for m in cfg.lag_minutes:
            col = f"{cfg.target_col}_lag_{int(m)}m"
            if col in df.columns:
                cols.append(col)

        if len(cols) == 0:
            raise ValueError("No baseline lag columns found in DataFrame for Granger block test.")

        return cols

    def _resolve_block_cols(self, *, df: pd.DataFrame, cfg: EvaluationConfig) -> list[str]:
        block = self._block_cols_override if self._block_cols_override is not None else list(cfg.cond_cols)
        block_present = [c for c in block if c in df.columns]
        if len(block_present) == 0:
            raise ValueError("No candidate block columns found in DataFrame for Granger block test.")
        return block_present

    def _ensure_dataframe(self, obj: object, *, name: str) -> pd.DataFrame:
        if isinstance(obj, pd.DataFrame):
            return obj
        if obj is None:
            return pd.DataFrame()
        raise TypeError(f"{self.name}: expected '{name}' to be a DataFrame, got {type(obj).__name__}.")

    def _flatten_scalars(self, by_horizon: pd.DataFrame) -> dict[str, float]:
        """
        Flatten per-horizon key values into per-subject scalars.

        Suggested scalars to expose:
        - partial_r2_is (effect size-like)
        - pval (significance)
        - F (test statistic)
        - n (samples)
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

            out[f"{hprefix}partial_r2_is"] = float(row.get("partial_r2_is", np.nan))
            out[f"{hprefix}pval"] = float(row.get("pval", np.nan))
            out[f"{hprefix}F"] = float(row.get("F", np.nan))
            out[f"{hprefix}n"] = float(row.get("n", np.nan))

        # Aggregate summaries across horizons (useful for cohort-level summaries)
        pr2 = pd.to_numeric(by_horizon.get("partial_r2_is"), errors="coerce").to_numpy(dtype=float)
        pval = pd.to_numeric(by_horizon.get("pval"), errors="coerce").to_numpy(dtype=float)

        out[f"{prefix}partial_r2_is__avg_over_horizons"] = float(np.nanmean(pr2)) if pr2.size else float("nan")
        out[f"{prefix}partial_r2_is__max_over_horizons"] = float(np.nanmax(pr2)) if pr2.size else float("nan")

        # A common summary: minimum p-value across horizons (strongest evidence)
        out[f"{prefix}pval__min_over_horizons"] = float(np.nanmin(pval)) if pval.size else float("nan")

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
        pr2 = pd.to_numeric(by_horizon.get("partial_r2_is"), errors="coerce").to_numpy(dtype=float)
        pval = pd.to_numeric(by_horizon.get("pval"), errors="coerce").to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(9, 3.5))

        # Plot effect size (partial R²) vs horizon
        ax.plot(h, pr2, marker="o", label="partial R² (in-sample)")

        ax.set_title(f"{self.name} – subject {subject_id}")
        ax.set_xlabel("Horizon (minutes)")
        ax.set_ylabel("partial R² (in-sample)")
        ax.grid(True, alpha=0.3)

        # Add a second y-axis for -log10(p) to visualize significance (optional but handy)
        ax2 = ax.twinx()
        with np.errstate(divide="ignore", invalid="ignore"):
            neglogp = -np.log10(pval)
        ax2.plot(h, neglogp, marker="o", label="-log10(p-value)")
        ax2.set_ylabel("-log10(p-value)")

        # Merge legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="best")

        return fig
