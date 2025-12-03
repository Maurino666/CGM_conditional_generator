from scipy import stats
import numpy as np
import pandas as pd

def build_arx_matrices(
    df: pd.DataFrame,
    target_col: str,
    candidate_cols: list[str],
    lag_minutes: list[int],
    horizon_min: int,
    add_time_of_day: bool = True,
    step_minutes: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build AR (base) and ARX (augmented) design matrices for a given horizon.

    Parameters
    ----------
    df : DataFrame with target, target lags, and candidate features.
    target_col : Name of the target column.
    candidate_cols : Block of candidate features to test (X at time t).
    lag_minutes : List of lags (in minutes) that already exist as '<target>_lag_{m}m'.
    horizon_min : Forecast horizon (minutes).
    add_time_of_day : If True, add 'tod_sin_24h' and 'tod_cos_24h' if present.
    step_minutes : Sampling step in minutes (default 5). Used to compute the shift.

    Returns
    -------
    X_base, X_aug, y : numpy arrays ready for estimation.
    """
    # Compute how many rows to shift for the horizon
    if horizon_min % step_minutes != 0:
        raise ValueError(f"horizon_min={horizon_min} not multiple of step_minutes={step_minutes}.")
    h_steps = horizon_min // step_minutes

    # Future target at t + h
    y = df[target_col].shift(-h_steps)

    # Base regressors: target lags (only those present)
    base_cols = [f"{target_col}_lag_{m}m" for m in lag_minutes if f"{target_col}_lag_{m}m" in df.columns]

    # Optional time-of-day harmonics, if available
    if add_time_of_day:
        for c in ("tod_sin_24h", "tod_cos_24h"):
            if c in df.columns:
                base_cols.append(c)

    # Augmented = base + candidate block (only those present)
    cand_cols = [c for c in candidate_cols if c in df.columns]

    # Extract matrices
    X_base_df = df[base_cols]
    X_aug_df  = df[base_cols + cand_cols]

    # Align and drop any row with NaNs in X_aug or y (X_base is a subset of columns)
    valid = (~X_aug_df.isna().any(axis=1)) & (~y.isna())
    Xb = X_base_df.loc[valid].to_numpy()
    Xa = X_aug_df.loc[valid].to_numpy()
    yy = y.loc[valid].to_numpy()

    return Xb, Xa, yy


def granger_block_f_test(X_base: np.ndarray, X_aug: np.ndarray, y: np.ndarray) -> dict:
    """
    Nested OLS F-test: H0 = all coefficients of (X_aug \ X_base) are zero.

    Returns
    -------
    dict with: n, F, pval, partial_r2_is, df_num, df_den
    """
    n = len(y)
    if n == 0:
        return {"n": 0, "F": np.nan, "pval": np.nan, "partial_r2_is": np.nan, "df_num": 0, "df_den": 0}

    # Add intercept
    Xb = np.column_stack([np.ones(n), X_base])
    Xa = np.column_stack([np.ones(n), X_aug])

    # OLS via least squares
    beta_b, _, _, _ = np.linalg.lstsq(Xb, y, rcond=None)
    beta_a, _, _, _ = np.linalg.lstsq(Xa, y, rcond=None)

    # Residual sum of squares
    rss_b = float(np.sum((y - Xb @ beta_b) ** 2))
    rss_a = float(np.sum((y - Xa @ beta_a) ** 2))

    k_b = Xb.shape[1]        # includes intercept
    k_a = Xa.shape[1]
    q   = k_a - k_b          # number of added parameters
    df_den = n - k_a
    df_num = q

    if q <= 0 or df_den <= 0 or rss_a <= 0:
        return {"n": n, "F": np.nan, "pval": np.nan, "partial_r2_is": np.nan,
                "df_num": int(max(q, 0)), "df_den": int(df_den)}

    # Classical nested-model F statistic
    F = ((rss_b - rss_a) / q) / (rss_a / df_den)
    pval = stats.f.sf(F, df_num, df_den)

    # In-sample partial R^2 of the added block
    partial_r2 = max(0.0, 1.0 - (rss_a / rss_b)) if rss_b > 0 else np.nan

    return {"n": n, "F": float(F), "pval": float(pval),
            "partial_r2_is": float(partial_r2), "df_num": int(df_num), "df_den": int(df_den)}


def compute_granger_block_over_horizons(
    df: pd.DataFrame,
    target_col: str,
    candidate_cols: list[str],
    lag_minutes: list[int],
    horizons_min: list[int],
    add_time_of_day: bool = True,
    step_minutes: int = 5,
    min_samples: int = 200,
) -> pd.DataFrame:
    """
    Multi-horizon Granger (block) F-test on an ARX with time-of-day option.

    Returns
    -------
    DataFrame with columns:
      ['horizon_min', 'n', 'F', 'pval', 'partial_r2_is', 'df_num', 'df_den']
    """
    rows = []
    for h in horizons_min:
        Xb, Xa, y = build_arx_matrices(
            df=df,
            target_col=target_col,
            candidate_cols=candidate_cols,
            lag_minutes=lag_minutes,
            horizon_min=h,
            add_time_of_day=add_time_of_day,
            step_minutes=step_minutes,
        )
        if len(y) < min_samples:
            rows.append({"horizon_min": h, "n": len(y), "F": np.nan, "pval": np.nan,
                         "partial_r2_is": np.nan, "df_num": 0, "df_den": 0})
            continue

        out = granger_block_f_test(Xb, Xa, y)
        out["horizon_min"] = h
        rows.append(out)

    cols = ["horizon_min", "n", "F", "pval", "partial_r2_is", "df_num", "df_den"]
    return pd.DataFrame(rows)[cols].sort_values("horizon_min").reset_index(drop=True)
