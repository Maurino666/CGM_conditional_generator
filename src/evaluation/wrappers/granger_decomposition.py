from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from evaluation.metrics_core import compute_granger_ab_decomposition
from ..types import EvaluationConfig, Metric, MetricOutput
from evaluation.wrappers.utils import resolve_feature_group, filter_valid_features


@dataclass(frozen=True)
class GrangerABDecompositionParams:
    """
    Parameters for Granger block F-test AB decomposition wrapper.
    """
    horizons_min: list[int]
    freq_min: int = 5

    # Feature group keys inside cfg.feature_groups
    group_a_key: str = "A"
    group_b_key: str = "B"

    # Core options
    add_time_of_day: bool = True
    min_samples: int = 200
    match_n: bool = True
    clip_shared_at_zero: bool = True
    clip_partial_r2_at_zero: bool = True

    # Output controls
    store_used_cols_table: bool = False  # optional debug table
    plot_kind: str = "partial_r2"         # "partial_r2" or "decomposition"


class GrangerABDecompositionMetric(Metric):
    """
    Wrapper for compute_granger_ab_decomposition.

    Responsibilities:
    - Resolve feature groups A/B from cfg.feature_groups (filtered to df columns)
    - Build baseline regressors from cfg.lag_minutes (Evaluator already created lag columns)
    - Call the multi-horizon core
    - Flatten outputs into MetricOutput.scalars
    - Optionally produce a per-subject plot
    """
    name = "granger_ab"

    def __init__(self, params: GrangerABDecompositionParams) -> None:
        self.params = params

    def compute(self, subject_id: int, df: pd.DataFrame, cfg: EvaluationConfig) -> MetricOutput:
        if cfg.lag_minutes is None or len(cfg.lag_minutes) == 0:
            raise ValueError(f"{self.name}: cfg.lag_minutes must be provided (baseline requires target lags).")

        # Baseline columns are the target lag columns ensured by the Evaluator
        base_cols = self._resolve_base_cols(df, cfg)

        # Resolve A/B from feature groups
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

        out = compute_granger_ab_decomposition(
            df,
            target_col=cfg.target_col,
            base_cols=base_cols,
            features_A=features_a,
            features_B=features_b,
            horizons_min=list(self.params.horizons_min),
            freq_min=int(self.params.freq_min),
            add_time_of_day=bool(self.params.add_time_of_day),
            min_samples=int(self.params.min_samples),
            match_n=bool(self.params.match_n),
            clip_shared_at_zero=bool(self.params.clip_shared_at_zero),
            clip_partial_r2_at_zero=bool(self.params.clip_partial_r2_at_zero),
        )

        by_horizon = out["by_horizon"].copy()

        scalars = self._flatten_scalars(by_horizon)
        tables: dict[str, pd.DataFrame] = {"by_horizon": by_horizon}

        if self.params.store_used_cols_table:
            tables["used_cols"] = pd.DataFrame(
                {
                    "base_cols": [str(base_cols)],
                    "A_cols": [str(features_a)],
                    "B_cols": [str(features_b)],
                }
            )

        artifacts: dict[str, str] = {}
        if cfg.per_subject_plots and cfg.output_dir is not None:
            plot_path = self._plot(subject_id, by_horizon, cfg.output_dir)
            artifacts["ab_plot"] = str(plot_path)

        return MetricOutput(scalars=scalars, tables=tables, artifacts=artifacts)

    def _resolve_base_cols(self, df: pd.DataFrame, cfg: EvaluationConfig) -> list[str]:
        """
        Baseline for Granger: target lag features already created upstream.

        We keep only columns present in df to be robust, but we also want to fail early
        if none exist.
        """
        cols: list[str] = []
        for m in cfg.lag_minutes or []:
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
        Flatten core outputs per horizon into scalar keys.

        Naming convention:
          <metric>__<field>__h<minutes>m
        """
        # Most useful columns from the core output
        fields = [
            "n",
            "partial_r2_A",
            "partial_r2_B",
            "partial_r2_AB",
            "unique_A",
            "unique_B",
            "shared",
            "unique_A_delta",
            "unique_B_delta",
            "shared_delta",
            "F_A",
            "p_A",
            "df_num_A",
            "df_den_A",
            "F_B",
            "p_B",
            "df_num_B",
            "df_den_B",
            "F_AB",
            "p_AB",
            "df_num_AB",
            "df_den_AB",
            "F_A_given_B",
            "p_A_given_B",
            "F_B_given_A",
            "p_B_given_A",
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

        # Optional compact summaries across horizons
        if "partial_r2_AB" in by_horizon.columns:
            vals = by_horizon["partial_r2_AB"].to_numpy(dtype=float)
            scalars[f"{self.name}__partial_r2_AB__mean_over_horizons"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan

        return scalars

    def _plot(self, subject_id: int, by_horizon: pd.DataFrame, output_dir: Path) -> Path:
        """
        Save a per-subject plot. Two modes:
        - partial_r2: partial_r2_A/B/AB over horizons
        - decomposition: unique_A/unique_B/shared over horizons
        """
        subj_dir = Path(output_dir) / "artifacts" / f"subject_{subject_id}"
        subj_dir.mkdir(parents=True, exist_ok=True)

        path = subj_dir / f"{self.name}__{self.params.plot_kind}.png"

        x = by_horizon["horizon_min"].to_numpy(dtype=int)

        plt.figure()

        if self.params.plot_kind == "decomposition":
            for col in ["unique_A", "unique_B", "shared"]:
                if col in by_horizon.columns:
                    plt.plot(x, by_horizon[col].to_numpy(dtype=float), marker="o", label=col)
            plt.ylabel("Decomposition (partial R² units)")
            plt.title(f"{self.name} decomposition - subject {subject_id}")
        else:
            for col in ["partial_r2_A", "partial_r2_B", "partial_r2_AB"]:
                if col in by_horizon.columns:
                    plt.plot(x, by_horizon[col].to_numpy(dtype=float), marker="o", label=col)
            plt.ylabel("Partial R² (in-sample)")
            plt.title(f"{self.name} partial R² - subject {subject_id}")

        plt.xlabel("Horizon (minutes)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

        return path
