import pandas as pd
import yaml
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

from sklearn.ensemble import RandomForestRegressor

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
RUN_NAME = "block1_baselines/20260225_201047_B1_diffwave"

# Paths
# Assuming global_config.yaml is in the parent directory or the run directory.
# Adjust this path to where your YAML file actually lives.
GLOBAL_CONFIG_PATH = Path("../global_config.yaml")

DATA_DIR = Path(f"../runs/{RUN_NAME}/val/csv_data")  # Folder containing per-subject CSVs
OUTPUT_DIR = Path(f"../reports/{RUN_NAME}/val")  # Folder where results/plots will be saved

# Synthetic Column Name
# This is usually specific to the model output and not in the global schema.
COL_TARGET_SYNTH = "glucose_synth"

HORIZONS = [15, 30, 45, 60, 75, 90, 105, 120]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_global_config(path: Path) -> dict:
    """
    Loads the YAML configuration file.
    """
    if not path.exists():
        raise FileNotFoundError(f"Global config file not found at: {path}")

    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_metrics() -> list:
    """
    Factory function to instantiate the list of metrics.
    We re-instantiate metrics for each run (Real vs Synth) to ensure
    no internal state leaks between the two passes.
    """


    return [
        # 1. Clinical Statistics (Mean, TIR, Hypo/Hyper)
        ClinicalStatsMetric(
            params=ClinicalStatsParams(hypo_threshold=70.0, hyper_threshold=180.0)
        ),

        # 2. Ambulatory Glucose Profile (AGP)
        AgpMetric(
            params=AgpParams(freq="5min"),
            name="agp"
        ),

        # 3. Linear Predictability (ARX)
        ArxDeltaR2LinearMetric(
            params=ArxDeltaR2LinearParams(
                horizons_min=HORIZONS,
                flatten_all_horizons=True,
                min_samples=200
            ),
            name="arx_linear"
        ),

        # 4. Granger Causality (Block F-Test)
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
                regressor_factory = lambda seed: RandomForestRegressor(
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
                k=5,  # Nearest neighbors for KSG
                min_samples=100,
                flatten_all_horizons=True
            ),
            name="cmi_ksg"
        ),
    ]


def load_series(path: Path, time_col: str | None) -> list[pd.DataFrame]:
    """
    Loads all CSV files from the specified directory into a list of DataFrames.
    Parses time columns if they exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Data directory not found: {path}")

    csv_files = sorted(list(path.glob("*.csv")))
    print(f"[Loader] Found {len(csv_files)} files in {path}")

    series = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            # Ensure time is datetime for accurate lag/AGP calculation
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
    # 1. Load Configuration
    print(f"[Config] Loading configuration from {GLOBAL_CONFIG_PATH}...")
    config_data = load_global_config(GLOBAL_CONFIG_PATH)
    schema = config_data.get("schema", {})

    # Extract Column Definitions from YAML Schema
    col_time = schema.get("time_col", "timestamp")
    col_target_real = schema.get("target_col", "glucose")

    # Chimera/Conditional Columns:
    # We take the list directly from the YAML 'cond_cols'
    all_cond_cols = schema.get("cond_cols", [])

    # Subject ID:
    # YAML defines 'metadata_cols', usually the first one is the ID (e.g., patient_id)
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

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # PASS 1: REAL DATA EVALUATION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print(" STARTING PASS 1: REAL DATA (Baseline)")
    print("=" * 40)

    cfg_real = EvaluationConfig(
        target_col=col_target_real,  # Loaded from YAML
        cond_cols=all_cond_cols,  # Loaded from YAML
        time_col=col_time,  # Loaded from YAML
        subject_id_col=col_subject,  # Loaded from YAML

        # Mask handling for Chimera datasets
        masked_dataframes=True,

        # Feature Engineering controls
        lag_minutes=HORIZONS,
        ensure_time_of_day=True,

        # Output controls
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

    print(all_cond_cols)

    cfg_synth = EvaluationConfig(
        target_col=COL_TARGET_SYNTH,  # Hardcoded/Defined in script
        cond_cols=all_cond_cols,  # Same conditions as Real
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
        real_res= results_real,
        synth_res= results_synth,
    )

    print(f"Found feature groups: {comparator.get_feature_groups()}")

    summary = comparator.get_summary_by_group()

    print(f"Summary:\n{summary}")

    summary.to_csv(OUTPUT_DIR / "comparison_by_feature_group.csv")

    key_metrics = [
        "basic_stats__mean",  # Il glucosio medio è realistico?
        "basic_stats__std",  # La variabilità è corretta?
        "basic_stats__tir",  # Time in Range (70-180): Fondamentale
        "basic_stats__tbr",  # Time Below Range (<70): Critico per la sicurezza (Ipoglicemia)
        "basic_stats__tar",  # Time Above Range (>180)

        "agp__iqr_mean",  # Variabilità intra-giornaliera media
        "agp__median_mean",  # Livello mediano del glucosio

        "arx_linear__delta_r2_mean__avg_over_horizons",  # Quanto linearmente predicibile è il segnale grazie all'input?
        "delta_r2_nonlinear__delta_r2_mean__avg_over_horizons",  # Quanto NON-linearmente predicibile è?
        "granger__partial_r2_is__avg_over_horizons",  # Causalità di Granger media
        "cmi_ksg__cmi_bits__avg_over_horizons"  # Conditional Mutual Information (Information flow)
    ]

    print("[Comparator] Generating distribution plots...")
    comparator.plot_metric_distributions(key_metrics, output_dir=OUTPUT_DIR)

    print("[Comparator] Generating error analysis plots...")
    comparator.plot_errors_by_group(key_metrics, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    run_pipeline()