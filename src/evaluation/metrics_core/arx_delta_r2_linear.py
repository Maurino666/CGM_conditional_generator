from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



class SafeStandardScaler(StandardScaler):
    """
    StandardScaler with a minimum scale threshold to avoid numerical explosions
    when a feature has near-zero variance in a training fold.
    """

    def __init__(self, *, min_scale: float = 1e-6):
        super().__init__()
        self.min_scale = float(min_scale)

    def fit(self, X, y=None, **fit_params):
        super().fit(X, y, **fit_params)

        # Clip extremely small scales to 1.0 (feature becomes effectively unscaled)
        scale = self.scale_
        scale[~np.isfinite(scale) | (scale < self.min_scale)] = 1.0
        self.scale_ = scale

        return self

def _infer_dt_minutes(df: pd.DataFrame, *, time_col: str | None) -> float:
    """
    Infer the sampling step (in minutes) as the median delta between consecutive timestamps.
    """
    if time_col is None:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Provide time_col or use a DatetimeIndex.")
        t = df.index.to_series()
    else:
        t = pd.to_datetime(df[time_col], errors="coerce")
        if t.isna().any():
            raise ValueError("time_col contains invalid datetimes.")

    dt_series = t.diff().dt.total_seconds().div(60.0).bfill()
    dt_min = float(dt_series.median())
    if not np.isfinite(dt_min) or dt_min <= 0:
        raise ValueError("Cannot infer a positive sampling step.")
    return dt_min


def _select_baseline_cols(
    df: pd.DataFrame,
    *,
    target_col: str,
    lag_minutes: list[int],
    add_time_of_day: bool,
) -> list[str]:
    """
    Select baseline columns (target lags + optional time-of-day).
    Assumes lag columns are already present (created upstream).
    """
    base_cols: list[str] = []
    for m in lag_minutes:
        col = f"{target_col}_lag_{m}m"
        if col in df.columns:
            base_cols.append(col)

    if add_time_of_day:
        for c in ("tod_sin_24h", "tod_cos_24h"):
            if c in df.columns and c not in base_cols:
                base_cols.append(c)

    if not base_cols:
        raise ValueError("No baseline columns found (target lags / tod_*).")

    return base_cols


def _select_candidate_cols(df: pd.DataFrame, *, candidate_cols: list[str]) -> list[str]:
    """
    Keep only candidate columns that exist in df.
    """
    cand_cols = [c for c in candidate_cols if c in df.columns]
    if not cand_cols:
        raise ValueError("No candidate columns found in DataFrame.")
    return cand_cols


def _compute_arx_delta_r2_linear_single_horizon(
    df: pd.DataFrame,
    *,
    target_col: str,
    candidate_cols: list[str],
    lag_minutes: list[int],
    horizon_min: int,
    add_time_of_day: bool = True,
    time_col: str | None = None,
    n_splits: int = 5,
    alpha: float = 1.0,
    min_samples: int = 200,
) -> dict[str, Any]:
    """
    Internal helper: compute ARX ΔR² (linear) for a single horizon.

    Returns a dict with (at least):
      - r2_base_mean, r2_aug_mean, delta_r2_mean
      - r2_base_folds, r2_aug_folds, delta_folds
      - n_samples, n_features_base, n_features_aug
      - used_cols_base, used_cols_aug
    """
    dt_min = _infer_dt_minutes(df, time_col=time_col)

    # Future target
    h_steps = max(1, int(round(float(horizon_min) / dt_min)))
    y = df[target_col].shift(-h_steps).rename("y_future")

    # Columns
    base_cols = _select_baseline_cols(df, target_col=target_col, lag_minutes=lag_minutes, add_time_of_day=add_time_of_day)
    cand_cols = _select_candidate_cols(df, candidate_cols=candidate_cols)

    X_base = df[base_cols]
    X_aug = df[base_cols + cand_cols]

    # Align to common sample (fair baseline vs augmented)
    data_base = pd.concat([y, X_base], axis=1).dropna()
    data_aug = pd.concat([y, X_aug], axis=1).dropna()
    common_idx = data_base.index.intersection(data_aug.index)

    if len(common_idx) < int(min_samples):
        return {
            "horizon_min": int(horizon_min),
            "r2_base_mean": np.nan,
            "r2_aug_mean": np.nan,
            "delta_r2_mean": np.nan,
            "r2_base_folds": [],
            "r2_aug_folds": [],
            "delta_folds": [],
            "n_samples": int(len(common_idx)),
            "n_features_base": int(len(base_cols)),
            "n_features_aug": int(len(base_cols) + len(cand_cols)),
            "used_cols_base": base_cols,
            "used_cols_aug": base_cols + cand_cols,
            "dt_min": float(dt_min),
            "h_steps": int(h_steps),
        }

    yv = y.loc[common_idx].to_numpy(dtype=float)
    Xb = X_base.loc[common_idx].to_numpy(dtype=float)
    Xa = X_aug.loc[common_idx].to_numpy(dtype=float)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    def fold_r2(X: np.ndarray) -> np.ndarray:
        scores: list[float] = []
        for tr, te in tscv.split(X):
            model = Pipeline([("scaler", SafeStandardScaler()), ("ridge", Ridge(alpha=alpha))])
            model.fit(X[tr], yv[tr])
            pred = model.predict(X[te])
            scores.append(float(r2_score(yv[te], pred)))
        return np.asarray(scores, dtype=float)

    r2b = fold_r2(Xb)
    r2a = fold_r2(Xa)
    delta = r2a - r2b

    return {
        "horizon_min": int(horizon_min),
        "r2_base_mean": float(np.nanmean(r2b)),
        "r2_aug_mean": float(np.nanmean(r2a)),
        "delta_r2_mean": float(np.nanmean(delta)),
        "r2_base_folds": r2b.tolist(),
        "r2_aug_folds": r2a.tolist(),
        "delta_folds": delta.tolist(),
        "n_samples": int(len(common_idx)),
        "n_features_base": int(len(base_cols)),
        "n_features_aug": int(len(base_cols) + len(cand_cols)),
        "used_cols_base": base_cols,
        "used_cols_aug": base_cols + cand_cols,
        "dt_min": float(dt_min),
        "h_steps": int(h_steps),
    }


def compute_arx_delta_r2_linear(
    df: pd.DataFrame,
    *,
    target_col: str,
    candidate_cols: list[str],
    lag_minutes: list[int],
    horizons_min: list[int],
    add_time_of_day: bool = True,
    time_col: str | None = None,
    n_splits: int = 5,
    alpha: float = 1.0,
    min_samples: int = 200,
) -> dict[str, Any]:
    """
    Compute ARX ΔR² (linear) over multiple horizons using time-series CV (Ridge).

    Baseline model (AR):
      y(t+h) ~ target lags [+ time-of-day]
    Augmented model (ARX):
      y(t+h) ~ target lags [+ time-of-day] + candidate_cols(t)

    This function is multi-horizon: it runs the single-horizon computation for each horizon in
    `horizons_min` and returns tabular outputs.

    Returns
    -------
    dict with:
      - by_horizon: pd.DataFrame (one row per horizon)
      - folds: pd.DataFrame (long format: horizon_min, fold, r2_base, r2_aug, delta_r2)
      - used_cols_base: list[str]
      - used_cols_aug: list[str]
      - meta: dict[str, object] (dt_min, add_time_of_day, n_splits, alpha, min_samples)
    """
    if len(horizons_min) == 0:
        raise ValueError("horizons_min must be non-empty.")

    # Compute once so column selection is consistent (and we can expose them)
    base_cols = _select_baseline_cols(df, target_col=target_col, lag_minutes=lag_minutes, add_time_of_day=add_time_of_day)
    cand_cols = _select_candidate_cols(df, candidate_cols=candidate_cols)

    rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []

    for h in horizons_min:
        res = _compute_arx_delta_r2_linear_single_horizon(
            df,
            target_col=target_col,
            candidate_cols=cand_cols,
            lag_minutes=lag_minutes,
            horizon_min=int(h),
            add_time_of_day=add_time_of_day,
            time_col=time_col,
            n_splits=n_splits,
            alpha=alpha,
            min_samples=min_samples,
        )

        rows.append(
            {
                "horizon_min": int(res["horizon_min"]),
                "dt_min": float(res.get("dt_min", np.nan)),
                "h_steps": int(res.get("h_steps", 0)),
                "r2_base_mean": float(res.get("r2_base_mean", np.nan)),
                "r2_aug_mean": float(res.get("r2_aug_mean", np.nan)),
                "delta_r2_mean": float(res.get("delta_r2_mean", np.nan)),
                "n_samples": int(res.get("n_samples", 0)),
                "n_features_base": int(res.get("n_features_base", len(base_cols))),
                "n_features_aug": int(res.get("n_features_aug", len(base_cols) + len(cand_cols))),
            }
        )

        r2b = res.get("r2_base_folds", [])
        r2a = res.get("r2_aug_folds", [])
        dlt = res.get("delta_folds", [])

        def to_float_list(x: object) -> list[float]:
            if not isinstance(x, list):
                return []
            out: list[float] = []
            for v in x:
                try:
                    out.append(float(v))
                except Exception:
                    out.append(float("nan"))
            return out

        r2b_f = to_float_list(r2b)
        r2a_f = to_float_list(r2a)
        dlt_f = to_float_list(dlt)
        n_folds = max(len(r2b_f), len(r2a_f), len(dlt_f))

        for k in range(n_folds):
            fold_rows.append(
                {
                    "horizon_min": int(res["horizon_min"]),
                    "fold": int(k),
                    "r2_base": r2b_f[k] if k < len(r2b_f) else float("nan"),
                    "r2_aug": r2a_f[k] if k < len(r2a_f) else float("nan"),
                    "delta_r2": dlt_f[k] if k < len(dlt_f) else float("nan"),
                }
            )

    by_horizon = pd.DataFrame(rows).sort_values("horizon_min").reset_index(drop=True)
    folds = pd.DataFrame(fold_rows).sort_values(["horizon_min", "fold"]).reset_index(drop=True)

    return {
        "by_horizon": by_horizon,
        "folds": folds,
        "used_cols_base": base_cols,
        "used_cols_aug": base_cols + cand_cols,
        "meta": {
            "add_time_of_day": bool(add_time_of_day),
            "time_col": time_col,
            "n_splits": int(n_splits),
            "alpha": float(alpha),
            "min_samples": int(min_samples),
        },
    }


def compute_arx_delta_r2_linear_ab_decomposition(
    df: pd.DataFrame,
    *,
    target_col: str,
    lag_minutes: list[int],
    horizons_min: list[int],
    features_A: list[str],
    features_B: list[str],
    add_time_of_day: bool = True,
    time_col: str | None = None,
    n_splits: int = 5,
    alpha: float = 1.0,
    min_samples: int = 200,
    clip_shared_at_zero: bool = True,
) -> dict[str, Any]:
    """
    Compute AB decomposition for ARX ΔR² (linear) over multiple horizons, on a *common sample* per horizon.

    For each horizon h:
      - Compute ΔR²(A), ΔR²(B), ΔR²(A+B) on the SAME filtered rows (fairness).
      - Decomposition in ΔR² space:
          delta_total = ΔR²(A+B)
          unique_A = ΔR²(A+B) - ΔR²(B)
          unique_B = ΔR²(A+B) - ΔR²(A)
          shared   = delta_total - unique_A - unique_B
        Optionally clip shared at >= 0.

    Returns
    -------
    dict with:
      - by_horizon: pd.DataFrame with columns:
          horizon_min, delta_total, unique_A, unique_B, shared, n_rows_common
          plus (optional) the mean scores for A/B/AB (r2_base_mean, r2_aug_mean, delta_r2_mean) suffixed
      - details: dict[int, dict[str, object]] mapping horizon_min -> {"results_A","results_B","results_AB",...}
      - used_cols_baseline, used_cols_A, used_cols_B, used_cols_AB
    """
    if len(horizons_min) == 0:
        raise ValueError("horizons_min must be non-empty.")

    # Baseline selection once (stable naming)
    base_cols = _select_baseline_cols(df, target_col=target_col, lag_minutes=lag_minutes, add_time_of_day=add_time_of_day)

    # Present A/B columns
    A_cols = [c for c in features_A if c in df.columns]
    B_cols = [c for c in features_B if c in df.columns]
    AB_cols = list(dict.fromkeys(A_cols + B_cols))  # keep order, drop dup
    if not AB_cols:
        raise ValueError("No A/B candidate columns found in DataFrame.")

    dt_min = _infer_dt_minutes(df, time_col=time_col)

    rows: list[dict[str, Any]] = []
    details: dict[int, dict[str, Any]] = {}

    for h in horizons_min:
        horizon_min = int(h)
        h_steps = max(1, int(round(float(horizon_min) / dt_min)))
        y_future = df[target_col].shift(-h_steps)

        # Common mask: y_future + baseline + (A+B)
        needed = base_cols + AB_cols
        mask = y_future.notna()
        for c in needed:
            mask &= df[c].notna()
        df_same = df.loc[mask]

        res_A = _compute_arx_delta_r2_linear_single_horizon(
            df_same,
            target_col=target_col,
            candidate_cols=A_cols,
            lag_minutes=lag_minutes,
            horizon_min=horizon_min,
            add_time_of_day=add_time_of_day,
            time_col=time_col,
            n_splits=n_splits,
            alpha=alpha,
            min_samples=min_samples,
        )
        res_B = _compute_arx_delta_r2_linear_single_horizon(
            df_same,
            target_col=target_col,
            candidate_cols=B_cols,
            lag_minutes=lag_minutes,
            horizon_min=horizon_min,
            add_time_of_day=add_time_of_day,
            time_col=time_col,
            n_splits=n_splits,
            alpha=alpha,
            min_samples=min_samples,
        )
        res_AB = _compute_arx_delta_r2_linear_single_horizon(
            df_same,
            target_col=target_col,
            candidate_cols=AB_cols,
            lag_minutes=lag_minutes,
            horizon_min=horizon_min,
            add_time_of_day=add_time_of_day,
            time_col=time_col,
            n_splits=n_splits,
            alpha=alpha,
            min_samples=min_samples,
        )

        dA = float(res_A.get("delta_r2_mean", np.nan))
        dB = float(res_B.get("delta_r2_mean", np.nan))
        dAB = float(res_AB.get("delta_r2_mean", np.nan))

        delta_total = dAB
        unique_A = dAB - dB if np.isfinite(dAB) and np.isfinite(dB) else np.nan
        unique_B = dAB - dA if np.isfinite(dAB) and np.isfinite(dA) else np.nan
        shared = delta_total - unique_A - unique_B if all(np.isfinite([delta_total, unique_A, unique_B])) else np.nan
        if clip_shared_at_zero and np.isfinite(shared):
            shared = max(0.0, float(shared))

        rows.append(
            {
                "horizon_min": horizon_min,
                "dt_min": float(dt_min),
                "h_steps": int(h_steps),
                "n_rows_common": int(len(df_same)),
                "delta_total": float(delta_total) if np.isfinite(delta_total) else np.nan,
                "unique_A": float(unique_A) if np.isfinite(unique_A) else np.nan,
                "unique_B": float(unique_B) if np.isfinite(unique_B) else np.nan,
                "shared": float(shared) if np.isfinite(shared) else np.nan,
                # Useful diagnostics (mean scores) for A/B/AB
                "dA_mean": float(dA) if np.isfinite(dA) else np.nan,
                "dB_mean": float(dB) if np.isfinite(dB) else np.nan,
                "dAB_mean": float(dAB) if np.isfinite(dAB) else np.nan,
            }
        )

        details[horizon_min] = {
            "results_A": res_A,
            "results_B": res_B,
            "results_AB": res_AB,
            "n_rows_common": int(len(df_same)),
            "used_cols_baseline": base_cols,
            "used_cols_A": A_cols,
            "used_cols_B": B_cols,
            "used_cols_AB": AB_cols,
        }

    by_horizon = pd.DataFrame(rows).sort_values("horizon_min").reset_index(drop=True)

    return {
        "by_horizon": by_horizon,
        "details": details,
        "used_cols_baseline": base_cols,
        "used_cols_A": A_cols,
        "used_cols_B": B_cols,
        "used_cols_AB": AB_cols,
        "meta": {
            "add_time_of_day": bool(add_time_of_day),
            "time_col": time_col,
            "n_splits": int(n_splits),
            "alpha": float(alpha),
            "min_samples": int(min_samples),
            "clip_shared_at_zero": bool(clip_shared_at_zero),
        },
    }
