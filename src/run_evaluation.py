import pandas as pd
import yaml
import copy
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

from sklearn.ensemble import RandomForestRegressor

from config_utils import deep_merge
from evaluation import CmiKsgMetric, DeltaR2NonlinearMetric, DeltaR2NonlinearParams, CmiKsgParams, CohortComparator
from evaluation.evaluator import Evaluator
from evaluation.types import EvaluationConfig
from evaluation.wrappers import (
    ClinicalStatsMetric, ClinicalStatsParams,
    AgpMetric, AgpParams,
    ArxDeltaR2LinearMetric, ArxDeltaR2LinearParams,
    GrangerBlockFTestMetric, GrangerBlockParams,
)

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

# Run Identifiers
RUN_NAME = "block1_baselines/20260225_092323_B1_conditional_timegan"

# Paths
RUN_DIR = Path(f"../runs/{RUN_NAME}")
DATA_DIR = RUN_DIR / "val" / "csv_data"
OUTPUT_DIR = Path(f"../reports/{RUN_NAME}/val")

# Synthetic Column Name
COL_TARGET_SYNTH = "glucose_synth"

HORIZONS = [15, 30, 45, 60, 75, 90, 105, 120]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def resolve_global_config(run_dir: Path) -> dict:
    """
    Resolves the global config from the experiment, applying any overrides.
    Mirrors the runner's _resolve_global_config logic.
    """
    experiment_config_path = run_dir / "experiment_config.yaml"
    if not experiment_config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {experiment_config_path}")

    with open(experiment_config_path, "r", encoding="utf-8") as f:
        experiment_config = yaml.safe_load(f)

    # Resolve base global config (path or inline dict)
    gc_section = experiment_config.get("global_config")
    if gc_section is None:
        base = {}
    elif isinstance(gc_section, dict):
        base = copy.deepcopy(gc_section)
    else:
        # It's a path — resolve relative to experiment YAML's original location
        # The experiment was run from the experiments/ directory
        p = Path(gc_section)
        if not p.is_absolute():
            # Try relative to run_dir first, then experiments/
            candidates = [
                run_dir / p,
                run_dir.parent.parent / "experiments" / p,
                Path("..") / p,  # fallback: relative to cwd
            ]
            for candidate in candidates:
                if candidate.exists():
                    p = candidate
                    break
        with open(p, "r", encoding="utf-8") as f:
            base = yaml.safe_load(f)

    # Apply overrides if present
    overrides = experiment_config.get("global_config_overrides")
    if overrides:
        base = deep_merge(base, overrides)
        print(f"[Config] Applied global_config_overrides from experiment")

    return base


def get_metrics() -> list:
    return [
        ClinicalStatsMetric(
            params=ClinicalStatsParams(hypo_threshold=70.0, hyper_threshold=180.0)
        ),
        AgpMetric(
            params=AgpParams(freq="5min"),
            name="agp"
        ),
        ArxDeltaR2LinearMetric(
            params=ArxDeltaR2LinearParams(
                horizons_min=HORIZONS,
                flatten_all_horizons=True,
                min_samples=200
            ),
            name="arx_linear"
        ),
        GrangerBlockFTestMetric(
            params=GrangerBlockParams(
                horizons_min=HORIZONS,
                freq_min=5,
                match_n=True
            ),
            name="granger"
        ),
        DeltaR2NonlinearMetric(
            params=DeltaR2NonlinearParams(
                horizons_min=HORIZONS,
                freq_min=5,
                add_time_of_day=True,
                min_samples=200,
                flatten_all_horizons=True,
                regressor_factory=lambda seed: RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    n_jobs=-1,
                    random_state=42
                )
            ),
            name="delta_r2_nonlinear"
        ),
        CmiKsgMetric(
            params=CmiKsgParams(
                horizons_min=HORIZONS,
                freq_min=5,
                k=5,
                min_samples=100,
                flatten_all_horizons=True
            ),
            name="cmi_ksg"
        ),
    ]


def load_series(path: Path, time_col: str | None) -> list[pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"Data directory not found: {path}")

    csv_files = sorted(list(path.glob("*.csv")))
    print(f"[Loader] Found {len(csv_files)} files in {path}")

    series = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            if time_col and time_col in df.columns:
                df[time_col] = pd.to_datetime(df[time_col])
            series.append(df)
        except Exception as e:
            print(f"[Loader] Error loading {f.name}: {e}")

    return series


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline():
    # 1. Load Configuration — resolved from experiment, with overrides
    print(f"[Config] Resolving global config from experiment: {RUN_DIR}...")
    config_data = resolve_global_config(RUN_DIR)
    schema = config_data.get("schema", {})

    col_time = schema.get("time_col", "timestamp")
    col_target_real = schema.get("target_col", "glucose")
    all_cond_cols = schema.get("cond_cols", [])
    metadata_cols = schema.get("metadata_cols", [])
    col_subject = metadata_cols[0] if metadata_cols else "patient_id"

    print(f"[Config] Time Col: {col_time}")
    print(f"[Config] Real Target: {col_target_real}")
    print(f"[Config] Synthetic Target: {COL_TARGET_SYNTH}")
    print(f"[Config] Subject ID Col: {col_subject}")
    print(f"[Config] Conditional Columns ({len(all_cond_cols)}): {all_cond_cols}")

    # 2. Load Data
    series = load_series(DATA_DIR, time_col=col_time)
    if not series:
        print("[Error] No data loaded. Exiting.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # PASS 1: REAL DATA EVALUATION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print(" STARTING PASS 1: REAL DATA (Baseline)")
    print("=" * 40)

    cfg_real = EvaluationConfig(
        target_col=col_target_real,
        cond_cols=all_cond_cols,
        time_col=col_time,
        subject_id_col=col_subject,
        masked_dataframes=True,
        lag_minutes=HORIZONS,
        ensure_time_of_day=True,
        output_dir=OUTPUT_DIR / "real",
        per_subject_plots=True
    )

    evaluator_real = Evaluator(cfg_real, get_metrics(), verbose=True)
    results_real = evaluator_real.evaluate(series)

    # -------------------------------------------------------------------------
    # PASS 2: SYNTHETIC DATA EVALUATION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print(" STARTING PASS 2: SYNTHETIC DATA (Model)")
    print("=" * 40)

    cfg_synth = EvaluationConfig(
        target_col=COL_TARGET_SYNTH,
        cond_cols=all_cond_cols,
        time_col=col_time,
        subject_id_col=col_subject,
        masked_dataframes=True,
        lag_minutes=HORIZONS,
        ensure_time_of_day=True,
        output_dir=OUTPUT_DIR / "synth",
        per_subject_plots=True
    )

    evaluator_synth = Evaluator(cfg_synth, get_metrics(), verbose=True)
    results_synth = evaluator_synth.evaluate(series)

    # -------------------------------------------------------------------------
    # COMPARE AND FINALIZE
    # -------------------------------------------------------------------------
    comparator = CohortComparator(
        real_res=results_real,
        synth_res=results_synth,
    )

    print(f"Found feature groups: {comparator.get_feature_groups()}")

    summary = comparator.get_summary_by_group()
    print(f"Summary:\n{summary}")
    summary.to_csv(OUTPUT_DIR / "comparison_by_feature_group.csv")

    key_metrics = [
        "basic_stats__mean",
        "basic_stats__std",
        "basic_stats__tir",
        "basic_stats__tbr",
        "basic_stats__tar",
        "agp__iqr_mean",
        "agp__median_mean",
        "arx_linear__delta_r2_mean__avg_over_horizons",
        "delta_r2_nonlinear__delta_r2_mean__avg_over_horizons",
        "granger__partial_r2_is__avg_over_horizons",
        "cmi_ksg__cmi_bits__avg_over_horizons"
    ]

    print("[Comparator] Generating distribution plots...")
    comparator.plot_metric_distributions(key_metrics, output_dir=OUTPUT_DIR)

    print("[Comparator] Generating error analysis plots...")
    comparator.plot_errors_by_group(key_metrics, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    run_pipeline()