from scipy.stats import wasserstein_distance
import numpy as np
import pandas as pd


def compute_distributional_distance(
        df: pd.DataFrame,
        target_col: str,
        synth_col: str,
        method: str = "wasserstein"
) -> dict[str, float]:
    """
    Computes distance between Real and Synthetic distributions.

    Args:
        df: DataFrame containing both columns.
        target_col: Real data column name.
        synth_col: Synthetic data column name.
        method: 'wasserstein' (default) or 'ks' (Kolmogorov-Smirnov).
    """
    # Clean data (drop NaNs in either column independently or jointly)
    # For distributional comparison, we usually treat them as two bags of numbers,
    # so we can drop NaNs independently to maximize sample size.
    real = df[target_col].dropna().values
    synth = df[synth_col].dropna().values

    if len(real) == 0 or len(synth) == 0:
        return {f"{method}_dist": float("nan")}

    if method == "wasserstein":
        dist = wasserstein_distance(real, synth)
    elif method == "energy":
        # Energy distance is a multivariate generalization, good for 1D too
        from scipy.stats import energy_distance
        dist = energy_distance(real, synth)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Basic statistics difference
    mean_diff = np.abs(np.mean(real) - np.mean(synth))
    std_diff = np.abs(np.std(real) - np.std(synth))

    return {
        f"{method}_dist": float(dist),
        "mean_abs_diff": float(mean_diff),
        "std_abs_diff": float(std_diff)
    }