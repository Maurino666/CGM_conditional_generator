import pandas as pd
import numpy as np

def compute_clinical_stats(
        values: pd.Series,
        hypo_threshold: float = 70.0,
        hyper_threshold: float = 180.0,
) -> dict[str, float]:
    """
    Computes standard clinical glucose metrics:
    - Mean, Std, CV (Coefficient of Variation)
    - TIR (Time In Range): % values within [hypo, hyper]
    - TBR (Time Below Range): % values < hypo
    - TAR (Time Above Range): % values > hyper
    - GMI (Glucose Management Indicator): estimated HbA1c based on mean glucose.

    Args:
        values: Glucose time series (mg/dL).
        hypo_threshold: Threshold for hypoglycemia (default 70 mg/dL).
        hyper_threshold: Threshold for hyperglycemia (default 180 mg/dL).

    Returns:
        Dictionary of scalar metrics. Returns NaNs if the series is empty.
    """
    # Ensure we work with clean data (drop NaNs)
    clean = values.dropna()
    total = len(clean)

    if total == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "cv": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "tir": float("nan"),
            "tbr": float("nan"),
            "tar": float("nan"),
            "gmi": float("nan"),
        }

    # 1. Basic Distributional Stats
    mean_val = float(clean.mean())
    std_val = float(clean.std(ddof=1))  # Sample std
    cv_val = std_val / mean_val if mean_val > 1e-6 else 0.0
    min_val = float(clean.min())
    max_val = float(clean.max())

    # 2. Clinical Ranges (TIR/TBR/TAR)
    # Using numpy for speed
    arr = clean.to_numpy(dtype=float)

    # Count occurrences
    n_hypo = np.sum(arr < hypo_threshold)
    n_hyper = np.sum(arr > hyper_threshold)
    # TIR is the remainder (includes boundaries typically,
    # but strict logic is: 100% - TBR - TAR ensures sum is 1.0)

    tbr = float(n_hypo) / total
    tar = float(n_hyper) / total
    tir = 1.0 - tbr - tar

    # 3. GMI (Glucose Management Indicator)
    # Formula: 3.31 + 0.02392 * mean_glucose_mgdl
    gmi = 3.31 + 0.02392 * mean_val

    return {
        "mean": mean_val,
        "std": std_val,
        "cv": cv_val,
        "min": min_val,
        "max": max_val,
        "tir": tir,
        "tbr": tbr,
        "tar": tar,
        "gmi": gmi,
    }