# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import pandas as pd
from data_prep import AZT1D2025Dataset

from evaluation import *
from evaluation.wrappers import CmiKsgDecompositionParams

# -----------------------------
# Fallback column names (EDIT)
# -----------------------------
DEFAULT_TARGET_COL = "cgm"
DEFAULT_TIME_COL = "timestamp"
DEFAULT_SUBJECT_ID_COL = "subject_id"

# If you do not have dataset.cond_cols, set them here:
DEFAULT_COND_COLS: list[str] = []


def main() -> None:
    dataset = AZT1D2025Dataset(
        Path("../datasets/AZT1D2025/CGM Records"),
        Path("../datasets/AZT1D2025/CGM Records/azt1d2025.yaml"),
        logging_dir=Path("../datasets/AZT1D2025/prep_logs"),
    )
    dataset.clean_data()

    # Prefer your dataset API if available; here you used dataset.all_data already
    series: list[pd.DataFrame] = dataset.all_data

    # -----------------------------
    # Resolve columns
    # -----------------------------
    target_col = getattr(dataset, "target_col", DEFAULT_TARGET_COL)

    # Example: single candidate feature
    cond_cols = [
        "basal_rate",
        "bolus_total",
        "bolus_correction",
        "bolus_meal",
        "carbs"
    ]

    cond_a = [
        "basal_rate",
    ]

    cond_b = [
        "bolus_total",
        "bolus_correction",
        "bolus_meal",
        "carbs",
    ]

    # -----------------------------
    # Configure evaluation
    # -----------------------------
    out_dir = Path("../reports") / "azt1d2025_eval_debug"
    out_dir.mkdir(parents=True, exist_ok=True)

    horizons = [15, 30, 45, 60, 75, 90, 105, 120]

    cfg = EvaluationConfig(
        target_col=target_col,
        cond_cols=list(cond_cols),
        time_col=None,
        subject_id_col=None,
        lag_minutes=horizons,
        ensure_time_of_day=True,
        output_dir=out_dir,
        per_subject_plots=True,
        feature_groups={
            "A": cond_a,  # block A
            "B": cond_b,  # block B
        },
    )

    # -----------------------------
    # Build metrics
    # -----------------------------
    metrics: list[Metric] = [
        # Mage
        MageMetric(),

        # AGP (wrapper handles plotting when cfg.per_subject_plots=True)
        AgpMetric(AgpParams(freq="5min", min_days_per_bin=5)),

        # ARX ΔR² linear (multi-horizon)
        ArxDeltaR2LinearMetric(
            params=ArxDeltaR2LinearParams(
                horizons_min=horizons,
                add_time_of_day=True,
                n_splits=5,
                alpha=1.0,
                min_samples=200,
                # Optional: flattening policy if you implemented it like others
                flatten_all_horizons=True,
            ),
            # Optional overrides:
            # candidate_cols=None -> uses cfg.cond_cols inside wrapper
        ),

        ArxDeltaR2LinearABMetric(
            params=ArxDeltaR2LinearABParams(
                horizons_min=horizons,
                add_time_of_day=True,
                n_splits=5,
                alpha=1.0,
                min_samples=200,
            )
        ),

        # Granger block F-test (multi-horizon)
        GrangerBlockFTestMetric(
            params=GrangerBlockParams(
                horizons_min=horizons,
                freq_min=5,
                add_time_of_day=True,
                match_n=True,
                min_samples=200,
                clip_partial_r2_at_zero=True,
                flatten_all_horizons=True,
            ),
            name="granger_block_f_test",
        ),
        GrangerABDecompositionMetric(
            params=GrangerABDecompositionParams(
                horizons_min=horizons,
                freq_min=5,
                add_time_of_day=True,
                match_n=True,
                min_samples=200,
                clip_partial_r2_at_zero=True,
            )
        ),

        # Nonlinear ΔR² (multi-horizon)
        DeltaR2NonlinearMetric(
            params=DeltaR2NonlinearParams(
                horizons_min=horizons,
                freq_min=5,
                add_time_of_day=True,
                cv_spec=TemporalCVSpec(n_splits=5, test_size=1000, min_train_size=2000, purge_gap=0),
                regressor_factory=lambda rs: _build_default_regressor(rs),
                random_state=42,
                min_samples=None,
                flatten_all_horizons=True,
            ),
            # IMPORTANT: provide your regressor_factory here (example placeholder)
            # regressor_factory must be a callable: (random_state: int) -> model with fit/predict

            name="delta_r2_nonlinear",
        ),

        NonlinearDeltaR2ABDecompositionMetric(
            params=NonlinearDeltaR2ABParams(
                horizons_min=horizons,
                freq_min=5,
                add_time_of_day=True,
                cv_spec=TemporalCVSpec(n_splits=5, test_size=1000, min_train_size=2000, purge_gap=0),
                regressor_factory=lambda rs: _build_default_regressor(rs),
                random_state=42,
                min_samples=None,
            )
        ),

        # CMI KSG (multi-horizon)
        CmiKsgMetric(
            params=CmiKsgParams(
                horizons_min=horizons,
                freq_min=5,
                add_time_of_day=True,
                k=10,
                metric="chebyshev",
                jitter=1e-6,
                random_state=0,
                min_samples=100,
                clip_at_zero=False,
                flatten_all_horizons=True,
            ),
            name="cmi_ksg",
        ),

        CmiKsgDecompositionMetric(
            params=CmiKsgDecompositionParams(
                horizons_min=horizons,
                freq_min=5,
                add_time_of_day=True,
                k=10,
                metric="chebyshev",
                jitter=1e-6,
                random_state=0,
                min_samples=100,
                clip_at_zero=False,
            )
        )
    ]

    # -----------------------------
    # Run evaluation
    # -----------------------------
    evaluator = Evaluator(cfg, metrics)
    result = evaluator.evaluate(series)

    # -----------------------------
    # Save outputs
    # -----------------------------
    per_subject_path = out_dir / "per_subject.csv"
    summary_path = out_dir / "summary.csv"

    result.per_subject.to_csv(per_subject_path, index=False)
    result.summary.to_csv(summary_path, index=False)

    print(f"[OK] Saved per-subject results to: {per_subject_path}")
    print(f"[OK] Saved summary results to:     {summary_path}")

    # -----------------------------
    # Save per-subject tables
    # -----------------------------
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    for subject_id, metric_dict in result.tables.per_subject.items():
        subj_dir = tables_dir / f"subject_{subject_id}"
        subj_dir.mkdir(parents=True, exist_ok=True)

        for metric_name, tables in metric_dict.items():
            for table_name, df_table in tables.items():
                path = subj_dir / f"{metric_name}__{table_name}.parquet"
                df_table.to_parquet(path)

    print(f"[OK] Saved per-subject tables under: {tables_dir}")

    # -----------------------------
    # Save artifact manifest
    # -----------------------------
    artifacts_dir = out_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    for subject_id, metric_dict in result.artifacts.per_subject.items():
        for metric_name, art in metric_dict.items():
            for art_name, art_path in art.items():
                manifest_rows.append(
                    {
                        "subject_id": str(subject_id),
                        "metric": metric_name,
                        "artifact": art_name,
                        "path": art_path,
                    }
                )

    if manifest_rows:
        manifest = pd.DataFrame(manifest_rows)
        manifest_path = artifacts_dir / "artifact_manifest.csv"
        manifest.to_csv(manifest_path, index=False)
        print(f"[OK] Saved artifact manifest to:   {manifest_path}")
    else:
        print("[INFO] No artifacts were produced.")


# -----------------------------
# Example regressor factory
# -----------------------------
def _build_default_regressor(random_state: int):
    """
    Build a default nonlinear regressor for DeltaR2NonlinearMetric.

    Replace this with your actual choice (e.g., LightGBM, RandomForest, XGBoost).
    """
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(
        n_estimators=300,
        random_state=int(random_state),
        n_jobs=-1,
    )


if __name__ == "__main__":
    main()
