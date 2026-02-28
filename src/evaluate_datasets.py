"""
evaluate_real_baseline.py
=========================
Calculates conditional metrics (Delta R², Granger, CMI) on REAL data only,
per dataset, to establish the theoretical baseline of how much the conditional
features actually predict glucose in each dataset.

This answers the fundamental question: "Is glucose predictable from these
features in my data?"

Usage:
    python evaluate_real_baseline.py

Output:
    ../reports/real_baselines/ with per-dataset results and comparison.
"""

import sys
from pathlib import Path

import pandas as pd
import yaml
import matplotlib


matplotlib.use("Agg")

from sklearn.ensemble import RandomForestRegressor

# Add src to path if needed
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from data_prep import AZT1D2025Dataset, BrisT1DDataset, HUPA_UCMDataset
from evaluation import CmiKsgMetric, DeltaR2NonlinearMetric, DeltaR2NonlinearParams, CmiKsgParams, \
    ArxDeltaR2LinearABMetric, ArxDeltaR2LinearABParams, GrangerABDecompositionMetric, GrangerABDecompositionParams, \
    NonlinearDeltaR2ABDecompositionMetric, NonlinearDeltaR2ABParams, TemporalCVSpec, CmiKsgDecompositionMetric, \
    CmiKsgDecompositionParams
from evaluation.evaluator import Evaluator
from evaluation.types import EvaluationConfig
from evaluation.wrappers import (
    ClinicalStatsMetric, ClinicalStatsParams,
    AgpMetric, AgpParams,
    ArxDeltaR2LinearMetric, ArxDeltaR2LinearParams,
    GrangerBlockFTestMetric, GrangerBlockParams,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

GLOBAL_CONFIG_PATH = Path("../global_config.yaml")
OUTPUT_DIR = Path("../reports/real_baselines_complete")
HORIZONS = [15, 30, 45, 60, 75, 90, 105, 120]

# Dataset definitions
DATASETS = {
    "AZT1D": {
        "class": AZT1D2025Dataset,
        "params": {
            "dataset_root": Path("../datasets/AZT1D2025/CGM Records"),
            "config_file": Path("../datasets/AZT1D2025/CGM Records/azt1d2025.yaml"),
            "patient_metadata_path": Path("../datasets/AZT1D2025/CGM Records/patient_metadata.yaml"),
            "logging_dir": Path("../datasets/AZT1D2025/prep_logs"),
        }
    },
    "HUPA": {
        "class": HUPA_UCMDataset,
        "params": {
            "dataset_root": Path("../datasets/HUPA-UCM Diabetes Dataset/Preprocessed"),
            "config_file": Path("../datasets/HUPA-UCM Diabetes Dataset/hupa-ucm.yaml"),
            "patient_metadata_path": Path("../datasets/HUPA-UCM Diabetes Dataset/patient_metadata.yaml"),
            "logging_dir": Path("../datasets/HUPA-UCM Diabetes Dataset/prep_logs"),
        }
    },
    "BrisT1D": {
        "class": BrisT1DDataset,
        "params": {
            "dataset_root": Path("../datasets/BrisT1D_open_dataset/device_data/processed_state"),
            "config_file": Path("../datasets/BrisT1D_open_dataset/brist1d.yaml"),
            "patient_metadata_path": Path("../datasets/BrisT1D_open_dataset/patient_metadata.yaml"),
            "logging_dir": Path("../datasets/BrisT1D_open_dataset/prep_logs"),
        }
    },
}


# =============================================================================
# HELPERS
# =============================================================================

def load_global_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_metrics() -> list:
    """Same metrics as run_evaluation.py but fresh instances."""
    return [
        ClinicalStatsMetric(
            params=ClinicalStatsParams(hypo_threshold=70.0, hyper_threshold=180.0)
        ),
        AgpMetric(
            params=AgpParams(freq="5min"),
            name="agp"
        ),
        # ArxDeltaR2LinearMetric(
        #     params=ArxDeltaR2LinearParams(
        #         horizons_min=HORIZONS,
        #         flatten_all_horizons=True,
        #         min_samples=200
        #     ),
        #     name="arx_linear"
        # ),
        ArxDeltaR2LinearABMetric(
            params=ArxDeltaR2LinearABParams(
                horizons_min=HORIZONS,
                add_time_of_day=True,
                n_splits=5,
                alpha=1.0,
                min_samples=200,
            )
        ),
        # GrangerBlockFTestMetric(
        #     params=GrangerBlockParams(
        #         horizons_min=HORIZONS,
        #         freq_min=5,
        #         match_n=True
        #     ),
        #     name="granger"
        # ),
        GrangerABDecompositionMetric(
            params=GrangerABDecompositionParams(
                horizons_min=HORIZONS,
                freq_min=5,
                add_time_of_day=True,
                match_n=True,
                min_samples=200,
                clip_partial_r2_at_zero=True,
            )
        ),
        # DeltaR2NonlinearMetric(
        #     params=DeltaR2NonlinearParams(
        #         horizons_min=HORIZONS,
        #         freq_min=5,
        #         add_time_of_day=True,
        #         min_samples=200,
        #         flatten_all_horizons=True,
        #         regressor_factory=lambda seed: RandomForestRegressor(
        #             n_estimators=100,
        #             max_depth=10,
        #             n_jobs=-1,
        #             random_state=42
        #         )
        #     ),
        #    name="delta_r2_nonlinear"
        #),
        NonlinearDeltaR2ABDecompositionMetric(
            params=NonlinearDeltaR2ABParams(
                horizons_min=HORIZONS,
                freq_min=5,
                add_time_of_day=True,
                cv_spec=TemporalCVSpec(n_splits=5, test_size=1000, min_train_size=2000, purge_gap=0),
                regressor_factory=lambda seed: RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    n_jobs=-1,
                    random_state=42
                ),
                random_state=42,
                min_samples=None,
            )
        ),
        # CmiKsgMetric(
        #     params=CmiKsgParams(
        #         horizons_min=HORIZONS,
        #         freq_min=5,
        #         k=5,
        #         min_samples=100,
        #         flatten_all_horizons=True
        #     ),
        #     name="cmi_ksg"
        # ),
        CmiKsgDecompositionMetric(
            params=CmiKsgDecompositionParams(
                horizons_min=HORIZONS,
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


def load_dataset(name: str, ds_config: dict, global_config_path: Path) -> list[pd.DataFrame]:
    """Load, clean, and standardize a dataset. Returns list of DataFrames."""
    print(f"\n{'='*60}")
    print(f"  Loading dataset: {name}")
    print(f"{'='*60}")

    cls = ds_config["class"]
    params = ds_config["params"].copy()
    params["global_config"] = global_config_path

    ds = cls(**params)
    ds.clean()
    ds.standardize()

    print(f"  -> {len(ds.all_data)} subjects loaded")
    return ds.all_data


def evaluate_dataset(
    name: str,
    dfs: list[pd.DataFrame],
    schema: dict,
    output_dir: Path,
) -> pd.DataFrame:
    """Run evaluation metrics on a single dataset's real data."""

    col_target = schema.get("target_col", "glucose")
    all_cond_cols = schema.get("cond_cols", [])

    # Filter cond_cols to only those present in this dataset
    sample_df = dfs[0]
    available_cond_cols = [c for c in all_cond_cols if c in sample_df.columns]

    print(f"\n  Target: {col_target}")
    print(f"  Available cond cols: {len(available_cond_cols)}/{len(all_cond_cols)}")
    print(f"  Missing: {set(all_cond_cols) - set(available_cond_cols)}")

    ds_output = output_dir / name
    ds_output.mkdir(parents=True, exist_ok=True)

    cond_a =  schema["cond_a"]
    cond_b = schema["cond_b"]

    cfg = EvaluationConfig(
        target_col=col_target,
        cond_cols=available_cond_cols,
        masked_dataframes=True,
        lag_minutes=HORIZONS,
        ensure_time_of_day=True,
        output_dir=ds_output,
        per_subject_plots=True,
        feature_groups={
            "A": cond_a,  # block A
            "B": cond_b,  # block B
        },
    )

    evaluator = Evaluator(cfg, get_metrics(), verbose=True)
    results = evaluator.evaluate(dfs)

    # Save per-dataset results
    per_subject_path = ds_output / "per_subject.csv"
    summary_path = ds_output / "summary.csv"

    results.per_subject.to_csv(per_subject_path, index=False)
    results.summary.to_csv(summary_path, index=False)

    print(f"[OK] Saved per-subject results to: {per_subject_path}")
    print(f"[OK] Saved summary results to:     {summary_path}")

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  REAL DATA BASELINE EVALUATION")
    print("  Evaluating conditional predictability per dataset")
    print("=" * 60)

    # Load global config
    config_data = load_global_config(GLOBAL_CONFIG_PATH)
    schema = config_data.get("schema", {})

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Key metrics to extract for the summary
    key_metric_cols = [
        "arx_linear__delta_r2_mean__avg_over_horizons",
        "arx_linear__r2_base_mean__avg_over_horizons",
        "arx_linear__r2_aug_mean__avg_over_horizons",
        "delta_r2_nonlinear__delta_r2_mean__avg_over_horizons",
        "delta_r2_nonlinear__r2_base_mean__avg_over_horizons",
        "delta_r2_nonlinear__r2_aug_mean__avg_over_horizons",
        "granger__partial_r2_is__avg_over_horizons",
        "cmi_ksg__cmi_bits__avg_over_horizons",
        "basic_stats__mean",
        "basic_stats__std",
        "basic_stats__tir",
        "basic_stats__tbr",
        "basic_stats__tar",
    ]

    all_summaries = {}

    # Evaluate each dataset independently
    for ds_name, ds_config in DATASETS.items():
        try:
            dfs = load_dataset(ds_name, ds_config, GLOBAL_CONFIG_PATH)
            results = evaluate_dataset(ds_name, dfs, schema, OUTPUT_DIR)

            # Extract from summary DataFrame (has columns: metric, mean, median, std)
            summary_df = results.summary.set_index("metric")

            row = {}
            for col in key_metric_cols:
                if col in summary_df.index:
                    row[col] = summary_df.loc[col, "mean"]

            all_summaries[ds_name] = row

        except Exception as e:
            print(f"\n  [ERROR] Failed to evaluate {ds_name}: {e}")
            import traceback
            traceback.print_exc()

    # Build comparison table
    if all_summaries:
        comparison = pd.DataFrame(all_summaries).T
        comparison.to_csv(OUTPUT_DIR / "dataset_comparison.csv")

        print("\n" + "=" * 60)
        print("  CROSS-DATASET COMPARISON (Real Data)")
        print("=" * 60)
        print(comparison.to_string())
        print(f"\n  -> Saved to {OUTPUT_DIR / 'dataset_comparison.csv'}")

    print("\n  Done.")


if __name__ == "__main__":
    main()