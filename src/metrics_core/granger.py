from __future__ import annotations


from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


# =========================
# Low-level core: OLS + RSS
# =========================
def ols_rss(X: np.ndarray, y: np.ndarray, *, add_intercept: bool = True) -> tuple[float, int]:
    """
    Compute OLS residual sum of squares (RSS) via least squares.

    Requirements
    ------------
    - X: shape (n, p), y: shape (n,)
    - No NaNs (must be filtered upstream).
    - If add_intercept=True, intercept is prepended.

    Returns
    -------
    (rss, p_effective) where p_effective includes intercept if added.
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    y = np.asarray(y).reshape(-1)
    n = int(len(y))
    if n == 0:
        return (np.nan, 0)

    X_ = np.column_stack([np.ones(n), X]) if add_intercept else X
    beta, _, _, _ = np.linalg.lstsq(X_, y, rcond=None)
    resid = y - X_ @ beta
    rss = float(resid.T @ resid)
    p_eff = int(X_.shape[1])
    return rss, p_eff


def nested_ols_block_f_test(
    X_reduced: np.ndarray,
    X_full: np.ndarray,
    y: np.ndarray,
    *,
    add_intercept: bool = True,
    clip_partial_r2_at_zero: bool = True,
) -> dict[str, float]:
    """
    Nested OLS block F-test:
      H0: coefficients of the added block (X_full \\ X_reduced) are zero.

    This is the classical in-sample Granger-style block test:
      reduced: y ~ X_reduced
      full:    y ~ X_full

    Requirements
    ------------
    - X_reduced and X_full share the same row order and have same number of rows.
    - X_full contains all columns in X_reduced plus the block columns (i.e. nested models).
    - No NaNs (must be removed upstream).
    - n must be > p_full (after intercept) to have positive df_den.

    Returns
    -------
    dict with:
      n, F, pval, partial_r2_is, df_num, df_den,
      rss_reduced, rss_full

    Notes
    -----
    partial_r2_is here is in-sample partial R² of the added block:
      1 - RSS_full / RSS_reduced
    """
    y = np.asarray(y).reshape(-1)
    Xr = np.asarray(X_reduced)
    Xf = np.asarray(X_full)

    n = int(len(y))
    if n == 0:
        return {
            "n": 0,
            "F": np.nan,
            "pval": np.nan,
            "partial_r2_is": np.nan,
            "df_num": 0,
            "df_den": 0,
            "rss_reduced": np.nan,
            "rss_full": np.nan,
        }
    if Xr.shape[0] != n or Xf.shape[0] != n:
        raise ValueError("Row count mismatch between X and y.")
    if Xr.shape[1] > Xf.shape[1]:
        raise ValueError("Reduced model has more columns than full model (not nested).")

    rss_r, p_r = ols_rss(Xr, y, add_intercept=add_intercept)
    rss_f, p_f = ols_rss(Xf, y, add_intercept=add_intercept)

    k = int(p_f - p_r)  # number of added parameters (intercept cancels out)
    df_den = int(n - p_f)
    df_num = int(k)

    if k <= 0 or df_den <= 0 or not np.isfinite(rss_r) or not np.isfinite(rss_f):
        return {
            "n": n,
            "F": np.nan,
            "pval": np.nan,
            "partial_r2_is": np.nan,
            "df_num": max(k, 0),
            "df_den": df_den,
            "rss_reduced": rss_r,
            "rss_full": rss_f,
        }

    # Guard against numerical issues
    if rss_f <= 0:
        rss_f = 1e-12

    F = ((rss_r - rss_f) / k) / (rss_f / df_den)
    pval = float(stats.f.sf(F, df_num, df_den))

    partial_r2 = (1.0 - (rss_f / rss_r)) if rss_r > 0 else np.nan
    if clip_partial_r2_at_zero and np.isfinite(partial_r2):
        partial_r2 = max(0.0, float(partial_r2))

    return {
        "n": n,
        "F": float(F),
        "pval": float(pval),
        "partial_r2_is": float(partial_r2) if np.isfinite(partial_r2) else np.nan,
        "df_num": df_num,
        "df_den": df_den,
        "rss_reduced": float(rss_r),
        "rss_full": float(rss_f),
    }


# ==========================================
# Core helpers: build sample + design matrices
# ==========================================
def _horizon_steps(horizon_min: int, freq_min: int) -> int:
    step = int(round(horizon_min / float(freq_min)))
    if step <= 0:
        raise ValueError(f"horizon_min={horizon_min} with freq_min={freq_min} gives non-positive step.")
    return step


def build_granger_design_matrices(
    df: pd.DataFrame,
    *,
    target_col: str,
    base_cols: list[str],
    block_cols: list[str],
    horizon_min: int,
    freq_min: int = 5,
    add_time_of_day: bool = True,
    match_n: bool = True,
) -> dict[str, object]:
    """
    Build reduced/full design matrices (X_reduced, X_full) and y_future for a given horizon.

    This function is pure plumbing:
    - it does NOT create lags or TOD; assumes they exist already.
    - it does NOT fit models; only returns aligned series/dataframes.

    Returns
    -------
    dict with:
      horizon_min, step, n,
      y (pd.Series),
      X_reduced (pd.DataFrame), X_full (pd.DataFrame),
      used_cols_reduced, used_cols_full
    """
    step = _horizon_steps(horizon_min, freq_min)
    y = df[target_col].shift(-step).rename("_y_")

    # Reduced cols = base + optional TOD
    used_reduced = list(base_cols)
    if add_time_of_day:
        for c in ("tod_sin_24h", "tod_cos_24h"):
            if c in df.columns and c not in used_reduced:
                used_reduced.append(c)

    missing_base = [c for c in used_reduced if c not in df.columns]
    if missing_base:
        raise ValueError(f"Missing base/TOD columns: {missing_base}")

    missing_block = [c for c in block_cols if c not in df.columns]
    if missing_block:
        raise ValueError(f"Missing block columns: {missing_block}")

    used_full = used_reduced + list(block_cols)

    Xr = df[used_reduced]
    Xf = df[used_full]

    if match_n:
        Z = pd.concat([y, Xf], axis=1).dropna()
        y2 = Z["_y_"]
        Xf2 = Z.drop(columns=["_y_"])
        Xr2 = Xf2[used_reduced]  # same rows
        return {
            "horizon_min": int(horizon_min),
            "step": int(step),
            "y": y2,
            "X_reduced": Xr2,
            "X_full": Xf2,
            "used_cols_reduced": used_reduced,
            "used_cols_full": used_full,
            "n": int(len(y2)),
        }

    return {
        "horizon_min": int(horizon_min),
        "step": int(step),
        "y": y,
        "X_reduced": Xr,
        "X_full": Xf,
        "used_cols_reduced": used_reduced,
        "used_cols_full": used_full,
        "n": int(len(df)),
    }


# ==================================
# Single-horizon computations (internal)
# ==================================
def _compute_granger_block_single_horizon(
    df: pd.DataFrame,
    *,
    target_col: str,
    base_cols: list[str],
    block_cols: list[str],
    horizon_min: int,
    freq_min: int = 5,
    add_time_of_day: bool = True,
    match_n: bool = True,
    min_samples: int = 200,
    clip_partial_r2_at_zero: bool = True,
) -> dict[str, Any]:
    """
    Internal helper: compute in-sample block F-test + partial R² for one horizon.
    """
    pack = build_granger_design_matrices(
        df,
        target_col=target_col,
        base_cols=base_cols,
        block_cols=block_cols,
        horizon_min=horizon_min,
        freq_min=freq_min,
        add_time_of_day=add_time_of_day,
        match_n=match_n,
    )

    n = int(pack["n"])
    if n < int(min_samples):
        return {
            "horizon_min": int(horizon_min),
            "step": int(pack["step"]),
            "n": n,
            "F": np.nan,
            "pval": np.nan,
            "partial_r2_is": np.nan,
            "df_num": 0,
            "df_den": 0,
            "rss_reduced": np.nan,
            "rss_full": np.nan,
            "used_cols_reduced": pack["used_cols_reduced"],
            "used_cols_full": pack["used_cols_full"],
        }

    if match_n:
        yv = pack["y"].to_numpy()
        Xr = pack["X_reduced"].to_numpy()
        Xf = pack["X_full"].to_numpy()
        out = nested_ols_block_f_test(
            X_reduced=Xr,
            X_full=Xf,
            y=yv,
            clip_partial_r2_at_zero=clip_partial_r2_at_zero,
        )
        out.update(
            {
                "horizon_min": int(horizon_min),
                "step": int(pack["step"]),
                "used_cols_reduced": pack["used_cols_reduced"],
                "used_cols_full": pack["used_cols_full"],
            }
        )
        return out

    # match_n=False: do local dropna for this single test
    y = pack["y"].rename("_y_")
    Xf_df = pack["X_full"]
    Z = pd.concat([y, Xf_df], axis=1).dropna()
    n2 = int(len(Z))
    if n2 < int(min_samples):
        return {
            "horizon_min": int(horizon_min),
            "step": int(pack["step"]),
            "n": n2,
            "F": np.nan,
            "pval": np.nan,
            "partial_r2_is": np.nan,
            "df_num": 0,
            "df_den": 0,
            "rss_reduced": np.nan,
            "rss_full": np.nan,
            "used_cols_reduced": pack["used_cols_reduced"],
            "used_cols_full": pack["used_cols_full"],
        }

    yv = Z["_y_"].to_numpy()
    Xf2 = Z.drop(columns=["_y_"])
    Xr2 = Xf2[pack["used_cols_reduced"]].to_numpy()

    out = nested_ols_block_f_test(
        X_reduced=Xr2,
        X_full=Xf2.to_numpy(),
        y=yv,
        clip_partial_r2_at_zero=clip_partial_r2_at_zero,
    )
    out.update(
        {
            "horizon_min": int(horizon_min),
            "step": int(pack["step"]),
            "used_cols_reduced": pack["used_cols_reduced"],
            "used_cols_full": pack["used_cols_full"],
        }
    )
    return out


def _compute_granger_ab_decomposition_single_horizon(
    df: pd.DataFrame,
    *,
    target_col: str,
    base_cols: list[str],
    features_A: list[str],
    features_B: list[str],
    horizon_min: int,
    freq_min: int = 5,
    add_time_of_day: bool = True,
    min_samples: int = 200,
    match_n: bool = True,
    clip_shared_at_zero: bool = True,
    clip_partial_r2_at_zero: bool = True,
) -> dict[str, Any]:
    """
    Internal helper: AB decomposition for one horizon (match_n must be True for strict fairness).
    """
    if not match_n:
        raise ValueError("AB decomposition requires match_n=True for fair comparisons.")

    step = _horizon_steps(horizon_min, freq_min)

    A_cols = list(features_A)
    B_cols = list(features_B)
    AB_cols = list(dict.fromkeys(A_cols + B_cols))

    pack_AB = build_granger_design_matrices(
        df,
        target_col=target_col,
        base_cols=base_cols,
        block_cols=AB_cols,
        horizon_min=horizon_min,
        freq_min=freq_min,
        add_time_of_day=add_time_of_day,
        match_n=True,
    )

    n = int(pack_AB["n"])
    if n < int(min_samples):
        return {
            "horizon_min": int(horizon_min),
            "step": int(step),
            "n": n,
            "partial_r2_A": np.nan,
            "partial_r2_B": np.nan,
            "partial_r2_AB": np.nan,
            "unique_A": np.nan,
            "unique_B": np.nan,
            "shared": np.nan,
            "unique_A_delta": np.nan,
            "unique_B_delta": np.nan,
            "shared_delta": np.nan,
            "F_A": np.nan,
            "p_A": np.nan,
            "df_num_A": np.nan,
            "df_den_A": np.nan,
            "F_B": np.nan,
            "p_B": np.nan,
            "df_num_B": np.nan,
            "df_den_B": np.nan,
            "F_AB": np.nan,
            "p_AB": np.nan,
            "df_num_AB": np.nan,
            "df_den_AB": np.nan,
            "F_A_given_B": np.nan,
            "p_A_given_B": np.nan,
            "F_B_given_A": np.nan,
            "p_B_given_A": np.nan,
        }

    # Base (+TOD) and full base+AB on the SAME rows
    yv = pack_AB["y"].to_numpy()
    X_base_df = pack_AB["X_reduced"]
    X_AB_df = pack_AB["X_full"]

    # Ensure A/B are aligned on the same row set
    XA_df = pd.concat([X_base_df, df.loc[X_base_df.index, A_cols]], axis=1)
    XB_df = pd.concat([X_base_df, df.loc[X_base_df.index, B_cols]], axis=1)

    # Robustness: align to intersection if something becomes NaN unexpectedly
    Z_A = pd.concat([pd.Series(yv, index=X_base_df.index, name="_y_"), XA_df], axis=1).dropna()
    Z_B = pd.concat([pd.Series(yv, index=X_base_df.index, name="_y_"), XB_df], axis=1).dropna()
    Z_AB = pd.concat([pd.Series(yv, index=X_base_df.index, name="_y_"), X_AB_df], axis=1).dropna()

    common_idx = Z_A.index.intersection(Z_B.index).intersection(Z_AB.index)
    if len(common_idx) < int(min_samples):
        n2 = int(len(common_idx))
        return {
            "horizon_min": int(horizon_min),
            "step": int(step),
            "n": n2,
            "partial_r2_A": np.nan,
            "partial_r2_B": np.nan,
            "partial_r2_AB": np.nan,
            "unique_A": np.nan,
            "unique_B": np.nan,
            "shared": np.nan,
            "unique_A_delta": np.nan,
            "unique_B_delta": np.nan,
            "shared_delta": np.nan,
            "F_A": np.nan,
            "p_A": np.nan,
            "df_num_A": np.nan,
            "df_den_A": np.nan,
            "F_B": np.nan,
            "p_B": np.nan,
            "df_num_B": np.nan,
            "df_den_B": np.nan,
            "F_AB": np.nan,
            "p_AB": np.nan,
            "df_num_AB": np.nan,
            "df_den_AB": np.nan,
            "F_A_given_B": np.nan,
            "p_A_given_B": np.nan,
            "F_B_given_A": np.nan,
            "p_B_given_A": np.nan,
        }

    y = Z_AB.loc[common_idx, "_y_"].to_numpy()
    X_base = X_base_df.loc[common_idx].to_numpy()
    X_A = XA_df.loc[common_idx].to_numpy()
    X_B = XB_df.loc[common_idx].to_numpy()
    X_AB = X_AB_df.loc[common_idx].to_numpy()

    # Unconditional nested tests vs base
    out_A = nested_ols_block_f_test(X_base, X_A, y, clip_partial_r2_at_zero=clip_partial_r2_at_zero)
    out_B = nested_ols_block_f_test(X_base, X_B, y, clip_partial_r2_at_zero=clip_partial_r2_at_zero)
    out_AB = nested_ols_block_f_test(X_base, X_AB, y, clip_partial_r2_at_zero=clip_partial_r2_at_zero)

    pr2_A = out_A["partial_r2_is"]
    pr2_B = out_B["partial_r2_is"]
    pr2_AB = out_AB["partial_r2_is"]

    # Conditional uniques via RSS ratios
    rss_baseB, _ = ols_rss(X_B, y, add_intercept=True)
    rss_baseA, _ = ols_rss(X_A, y, add_intercept=True)
    rss_fullAB, _ = ols_rss(X_AB, y, add_intercept=True)

    unique_A = (1.0 - (rss_fullAB / rss_baseB)) if np.isfinite(rss_fullAB) and np.isfinite(rss_baseB) and rss_baseB > 0 else np.nan
    unique_B = (1.0 - (rss_fullAB / rss_baseA)) if np.isfinite(rss_fullAB) and np.isfinite(rss_baseA) and rss_baseA > 0 else np.nan
    if np.isfinite(unique_A) and clip_partial_r2_at_zero:
        unique_A = max(0.0, float(unique_A))
    if np.isfinite(unique_B) and clip_partial_r2_at_zero:
        unique_B = max(0.0, float(unique_B))

    shared = pr2_AB - unique_A - unique_B if all(np.isfinite([pr2_AB, unique_A, unique_B])) else np.nan
    if np.isfinite(shared) and clip_shared_at_zero:
        shared = max(0.0, float(shared))

    # Conditional F-tests
    out_A_given_B = nested_ols_block_f_test(X_B, X_AB, y, clip_partial_r2_at_zero=clip_partial_r2_at_zero)
    out_B_given_A = nested_ols_block_f_test(X_A, X_AB, y, clip_partial_r2_at_zero=clip_partial_r2_at_zero)

    # Additive delta-style decomposition on partial R²
    unique_A_delta = (pr2_AB - pr2_B) if all(np.isfinite([pr2_AB, pr2_B])) else np.nan
    unique_B_delta = (pr2_AB - pr2_A) if all(np.isfinite([pr2_AB, pr2_A])) else np.nan
    shared_delta = (pr2_A + pr2_B - pr2_AB) if all(np.isfinite([pr2_A, pr2_B, pr2_AB])) else np.nan

    if clip_shared_at_zero and np.isfinite(shared_delta):
        shared_delta = max(0.0, float(shared_delta))
    if clip_partial_r2_at_zero:
        if np.isfinite(unique_A_delta):
            unique_A_delta = max(0.0, float(unique_A_delta))
        if np.isfinite(unique_B_delta):
            unique_B_delta = max(0.0, float(unique_B_delta))

    return {
        "horizon_min": int(horizon_min),
        "step": int(step),
        "n": int(len(common_idx)),
        "partial_r2_A": float(pr2_A) if np.isfinite(pr2_A) else np.nan,
        "partial_r2_B": float(pr2_B) if np.isfinite(pr2_B) else np.nan,
        "partial_r2_AB": float(pr2_AB) if np.isfinite(pr2_AB) else np.nan,
        "unique_A": float(unique_A) if np.isfinite(unique_A) else np.nan,
        "unique_B": float(unique_B) if np.isfinite(unique_B) else np.nan,
        "shared": float(shared) if np.isfinite(shared) else np.nan,
        "unique_A_delta": float(unique_A_delta) if np.isfinite(unique_A_delta) else np.nan,
        "unique_B_delta": float(unique_B_delta) if np.isfinite(unique_B_delta) else np.nan,
        "shared_delta": float(shared_delta) if np.isfinite(shared_delta) else np.nan,
        # Unconditional tests
        "F_A": out_A["F"],
        "p_A": out_A["pval"],
        "df_num_A": out_A["df_num"],
        "df_den_A": out_A["df_den"],
        "F_B": out_B["F"],
        "p_B": out_B["pval"],
        "df_num_B": out_B["df_num"],
        "df_den_B": out_B["df_den"],
        "F_AB": out_AB["F"],
        "p_AB": out_AB["pval"],
        "df_num_AB": out_AB["df_num"],
        "df_den_AB": out_AB["df_den"],
        # Conditional tests
        "F_A_given_B": out_A_given_B["F"],
        "p_A_given_B": out_A_given_B["pval"],
        "F_B_given_A": out_B_given_A["F"],
        "p_B_given_A": out_B_given_A["pval"],
    }


# ==================================
# Public API (multi-horizon): compute_*
# ==================================
def compute_granger_block(
    df: pd.DataFrame,
    *,
    target_col: str,
    base_cols: list[str],
    block_cols: list[str],
    horizons_min: list[int],
    freq_min: int = 5,
    add_time_of_day: bool = True,
    match_n: bool = True,
    min_samples: int = 200,
    clip_partial_r2_at_zero: bool = True,
) -> dict[str, Any]:
    """
    Compute Granger-style in-sample block F-test over multiple horizons.

    Returns
    -------
    dict with:
      - by_horizon: pd.DataFrame (one row per horizon)
      - used_cols_reduced: list[str]
      - used_cols_full: list[str]
      - meta: dict[str, object]
    """
    if len(horizons_min) == 0:
        raise ValueError("horizons_min must be non-empty.")

    rows: list[dict[str, Any]] = []
    used_reduced: list[str] | None = None
    used_full: list[str] | None = None

    for h in horizons_min:
        res = _compute_granger_block_single_horizon(
            df,
            target_col=target_col,
            base_cols=base_cols,
            block_cols=block_cols,
            horizon_min=int(h),
            freq_min=int(freq_min),
            add_time_of_day=bool(add_time_of_day),
            match_n=bool(match_n),
            min_samples=int(min_samples),
            clip_partial_r2_at_zero=bool(clip_partial_r2_at_zero),
        )

        used_reduced = res.get("used_cols_reduced", used_reduced)
        used_full = res.get("used_cols_full", used_full)

        rows.append(
            {
                "horizon_min": int(res.get("horizon_min", h)),
                "step": int(res.get("step", _horizon_steps(int(h), int(freq_min)))),
                "n": int(res.get("n", 0)),
                "F": float(res.get("F", np.nan)),
                "pval": float(res.get("pval", np.nan)),
                "partial_r2_is": float(res.get("partial_r2_is", np.nan)),
                "df_num": float(res.get("df_num", np.nan)),
                "df_den": float(res.get("df_den", np.nan)),
                "rss_reduced": float(res.get("rss_reduced", np.nan)),
                "rss_full": float(res.get("rss_full", np.nan)),
            }
        )

    by_horizon = pd.DataFrame(rows).sort_values("horizon_min").reset_index(drop=True)

    return {
        "by_horizon": by_horizon,
        "used_cols_reduced": used_reduced or [],
        "used_cols_full": used_full or [],
        "meta": {
            "freq_min": int(freq_min),
            "add_time_of_day": bool(add_time_of_day),
            "match_n": bool(match_n),
            "min_samples": int(min_samples),
            "clip_partial_r2_at_zero": bool(clip_partial_r2_at_zero),
        },
    }


def compute_granger_ab_decomposition(
    df: pd.DataFrame,
    *,
    target_col: str,
    base_cols: list[str],
    features_A: list[str],
    features_B: list[str],
    horizons_min: list[int],
    freq_min: int = 5,
    add_time_of_day: bool = True,
    min_samples: int = 200,
    match_n: bool = True,
    clip_shared_at_zero: bool = True,
    clip_partial_r2_at_zero: bool = True,
) -> dict[str, Any]:
    """
    Compute Granger-style AB decomposition over multiple horizons.

    Returns
    -------
    dict with:
      - by_horizon: pd.DataFrame (one row per horizon) containing:
          partial_r2_A, partial_r2_B, partial_r2_AB,
          unique_A, unique_B, shared,
          unique_A_delta, unique_B_delta, shared_delta,
          F/p/df for A, B, AB, and conditional tests A|B, B|A
      - meta: dict[str, object]
    """
    if len(horizons_min) == 0:
        raise ValueError("horizons_min must be non-empty.")

    rows: list[dict[str, Any]] = []

    for h in horizons_min:
        res = _compute_granger_ab_decomposition_single_horizon(
            df,
            target_col=target_col,
            base_cols=base_cols,
            features_A=features_A,
            features_B=features_B,
            horizon_min=int(h),
            freq_min=int(freq_min),
            add_time_of_day=bool(add_time_of_day),
            min_samples=int(min_samples),
            match_n=bool(match_n),
            clip_shared_at_zero=bool(clip_shared_at_zero),
            clip_partial_r2_at_zero=bool(clip_partial_r2_at_zero),
        )
        rows.append(res)

    by_horizon = pd.DataFrame(rows).sort_values("horizon_min").reset_index(drop=True)

    return {
        "by_horizon": by_horizon,
        "meta": {
            "freq_min": int(freq_min),
            "add_time_of_day": bool(add_time_of_day),
            "match_n": bool(match_n),
            "min_samples": int(min_samples),
            "clip_shared_at_zero": bool(clip_shared_at_zero),
            "clip_partial_r2_at_zero": bool(clip_partial_r2_at_zero),
        },
    }
